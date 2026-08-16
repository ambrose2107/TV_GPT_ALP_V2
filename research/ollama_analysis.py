"""
research/ollama_analysis.py
Local, free LLM analysis using Ollama (no API key, no internet LLM calls).

Replaces research/ai_research.py's Anthropic+web_search approach with a
local model. Instead of asking the LLM to "search" and invent numbers
(which a local model cannot do reliably), we compute REAL indicators first
with core/analyzer.py + core/market_data.py, then ask Ollama to only
*interpret* those grounded numbers in plain English. This avoids
hallucinated prices/RSI/etc.

Requires:
  - Ollama installed and running locally: https://ollama.com
  - `ollama serve` running (default http://localhost:11434)
  - a model pulled, e.g.:  ollama pull llama3.1
                            ollama pull qwen2.5:7b   (good for finance/JSON)
                            ollama pull phi4         (small, fast, low RAM)

Env vars (all optional, sensible defaults):
  OLLAMA_URL    = http://localhost:11434   (Ollama REST API base)
  OLLAMA_MODEL  = llama3.1                 (any locally pulled model tag)
"""
import os
import requests
from datetime import datetime
from core.logger import get_logger
from core.analyzer import analyze_symbol

logger = get_logger(__name__)

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def _ollama_reachable() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_ollama(prompt: str, system: str = None, max_tokens: int = 700) -> str:
    """Call local Ollama /api/chat (non-streaming)."""
    if not _ollama_reachable():
        return (
            "⚠️ Ollama is not reachable at "
            f"{OLLAMA_URL}. Start it with `ollama serve` and make sure a "
            f"model is pulled (`ollama pull {OLLAMA_MODEL}`)."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        return f"⚠️ Could not connect to Ollama at {OLLAMA_URL}. Is `ollama serve` running?"
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"⚠️ Ollama error: {e}"


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
    Main entry point. Computes real indicators, then asks the local
    Ollama model to write a plain-English technical/fundamental-style
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

    text = _call_ollama(prompt, system=system, max_tokens=350)

    return {
        "symbol": analysis.get("symbol"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": OLLAMA_MODEL,
        "source": "ollama-local",
        "indicators": analysis,       # raw grounded numbers, for the UI
        "analysis": text,             # LLM's plain-English interpretation
    }
