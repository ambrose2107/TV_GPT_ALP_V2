"""
research/ollama_analysis.py
LLM-written technical analysis, grounded on REAL computed indicators (no
hallucinated prices/RSI/etc — core/analyzer.py computes those first, the
LLM only interprets them in plain English).

Supports two interchangeable providers, chosen with AI_PROVIDER:

  AI_PROVIDER=ollama (default) — runs 100% locally via Ollama.
    Requires Ollama installed and running: https://ollama.com
    Good for: privacy, zero cost, no API key. Bad for: needs a fairly
    capable local machine, doesn't work when deployed to Render (no LLM
    runs there).
      OLLAMA_URL   = http://localhost:11434
      OLLAMA_MODEL = llama3.1   (must match `ollama list` exactly)

  AI_PROVIDER=groq — calls Groq's hosted API (free tier, no credit card,
    https://console.groq.com). Runs on dedicated inference hardware, so
    responses come back in under a second — no more 180-300s timeouts.
    Works from Render since it's just an outbound HTTPS call, not a
    locally-running model.
      GROQ_API_KEY = gsk_...              (from console.groq.com)
      GROQ_MODEL   = llama-3.1-8b-instant (fast/cheap default; try
                                            llama-3.3-70b-versatile for
                                            better quality, still free tier)

Set AI_PROVIDER=groq in Render's environment variables (your bot can't run
Ollama there), and keep AI_PROVIDER=ollama locally if you prefer to stay
fully local/private when testing on your own PC.
"""
import os
from datetime import datetime
import requests
from core.logger import get_logger
from core.analyzer import analyze_symbol

logger = get_logger(__name__)

AI_PROVIDER = os.environ.get("AI_PROVIDER", "ollama").lower()

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"


# ── Ollama backend ──────────────────────────────────────────────────────────
def _ollama_reachable() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_ollama(messages: list, max_tokens: int) -> str:
    if not _ollama_reachable():
        return (
            f"⚠️ Ollama is not reachable at {OLLAMA_URL}. Start it with "
            f"`ollama serve` and make sure a model is pulled "
            f"(`ollama pull {OLLAMA_MODEL}`). Or set AI_PROVIDER=groq to "
            f"use the hosted API instead (works on Render too)."
        )
    body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        return f"⚠️ Could not connect to Ollama at {OLLAMA_URL}. Is `ollama serve` running?"
    except requests.exceptions.ReadTimeout:
        return "⚠️ Ollama timed out (>300s). Your machine may be too slow for this model — try a smaller tag, or set AI_PROVIDER=groq."
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"⚠️ Ollama error: {e}"


# ── Groq backend (hosted, free tier, OpenAI-compatible) ────────────────────
def _call_groq(messages: list, max_tokens: int) -> str:
    if not GROQ_API_KEY:
        return (
            "⚠️ GROQ_API_KEY is not set. Get a free key (no card) at "
            "https://console.groq.com/keys and set it in your env vars."
        )
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=30)
        if resp.status_code == 401:
            return "⚠️ Groq rejected the API key (401). Check GROQ_API_KEY is correct."
        if resp.status_code == 429:
            return "⚠️ Groq free-tier rate limit hit (429). Wait a bit and retry."
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "⚠️ Could not reach Groq API. Check your internet connection."
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return f"⚠️ Groq error: {e}"


def _call_llm(prompt: str, system: str, max_tokens: int = 350) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if AI_PROVIDER == "groq":
        return _call_groq(messages, max_tokens)
    return _call_ollama(messages, max_tokens)


def _current_model_name() -> str:
    return GROQ_MODEL if AI_PROVIDER == "groq" else OLLAMA_MODEL


def _format_indicators_for_prompt(analysis: dict) -> str:
    """Turn analyze_symbol() output into a compact, factual block for the LLM."""
    lines = [
        f"Symbol: {analysis.get('symbol')}",
        f"Last price: {analysis.get('price')} (source: {analysis.get('source')})",
        f"Overall technical score: {analysis.get('overall_label')} "
        f"(score {analysis.get('overall_score')}/10, lower = more bullish)",
        "",
        "Per-timeframe indicators (REAL, already computed — do not recompute or invent):",
    ]
    for tf, row in analysis.get("timeframes", {}).items():
        if "error" in row:
            lines.append(f"  [{tf}] no data: {row['error']}")
            continue
        rsi  = row.get("rsi", {})
        macd = row.get("macd", {})
        adx  = row.get("adx", {})
        bb   = row.get("bb", {})
        ema  = row.get("ema50", {})
        vwap = row.get("vwap", {})
        res  = row.get("result", {})
        lines.append(
            f"  [{tf}] RSI={rsi.get('v')} ({rsi.get('l')}) | "
            f"MACD={macd.get('l')} | ADX={adx.get('v')} ({adx.get('l')}) | "
            f"BollingerBand={bb.get('l')} | EMA50={ema.get('v')} ({ema.get('l')}) | "
            f"VWAP={vwap.get('v')} ({vwap.get('l')}) | "
            f"=> {res.get('l')} (score {res.get('score')})"
        )
    return "\n".join(lines)


def get_local_ai_analysis(symbol: str, timeframes: list = None) -> dict:
    """
    Main entry point. Computes real indicators, then asks the configured
    LLM provider (Ollama or Groq) to write a plain-English technical
    summary grounded in those numbers.
    """
    analysis = analyze_symbol(symbol, timeframes)
    indicator_block = _format_indicators_for_prompt(analysis)

    system = (
        "You are a disciplined technical-analysis assistant. You are given "
        "REAL, already-computed indicator values for a stock. Never invent "
        "prices, indicator values, news, or filings you were not given. "
        "Summarize what the numbers imply, note any conflicting signals "
        "across timeframes, and end with a short risk note. Do not give "
        "financial advice or a buy/sell instruction — describe the setup "
        "only."
    )
    prompt = (
        f"{indicator_block}\n\n"
        "Write a concise (100-150 word) analysis:\n"
        "1) What the current technical picture suggests\n"
        "2) Where timeframes agree or disagree\n"
        "3) Key level/momentum risk to watch\n"
        "Be brief and direct.\n"
    )

    text = _call_llm(prompt, system=system, max_tokens=350)

    return {
        "symbol": analysis.get("symbol"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": _current_model_name(),
        "source": f"{AI_PROVIDER}-{'hosted' if AI_PROVIDER == 'groq' else 'local'}",
        "indicators": analysis,       # raw grounded numbers, for the UI
        "analysis": text,             # LLM's plain-English interpretation
    }
