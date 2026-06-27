"""
mirrorfish/engine.py — Multi-LLM AI market prediction engine  v8
Providers (all free tier): Groq → OpenRouter → HuggingFace
No new pip packages needed — uses requests (already in requirements.txt).
"""
import os, json, requests
from datetime import datetime
from core.logger import get_logger

logger = get_logger(__name__)

PROVIDERS = {
    "groq": {
        "name":          "Groq (LLaMA 3.3 70B — free)",
        "url":           "https://api.groq.com/openai/v1/chat/completions",
        "env_key":       "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "models":        ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"],
        "headers_extra": {},
    },
    "openrouter": {
        "name":          "OpenRouter (free tier)",
        "url":           "https://openrouter.ai/api/v1/chat/completions",
        "env_key":       "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "models":        ["meta-llama/llama-3.3-70b-instruct:free","mistralai/mistral-7b-instruct:free","google/gemma-2-9b-it:free"],
        "headers_extra": {"HTTP-Referer":"https://optitrade-ai.railway.app","X-Title":"OptiTrade MirrorFish"},
    },
    "huggingface": {
        "name":          "HuggingFace Inference (free)",
        "url":           "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        "env_key":       "HUGGINGFACE_API_KEY",
        "default_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "models":        ["mistralai/Mistral-7B-Instruct-v0.3"],
        "headers_extra": {},
        "hf_mode":       True,
    },
}

SYSTEM_PROMPT = (
    "You are MirrorFish, an AI market analyst in OptiTrade, an algorithmic trading system. "
    "You analyze technical data and give concise, structured insights. "
    "Always respond in valid JSON when asked. Never give financial advice — note analysis is educational only."
)

def _get_provider():
    for name in ["groq", "openrouter", "huggingface"]:
        p = PROVIDERS[name]
        if os.environ.get(p["env_key"]):
            return name, p
    return None, None

def _call_openai(p, model, messages, max_tokens=600, temperature=0.3):
    key = os.environ.get(p["env_key"], "")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **p.get("headers_extra", {})}
    resp = requests.post(p["url"], headers=headers,
                         json={"model": model, "messages": messages,
                               "max_tokens": max_tokens, "temperature": temperature},
                         timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def _call_hf(p, prompt, max_tokens=400):
    key = os.environ.get(p["env_key"], "")
    headers = {"Authorization": f"Bearer {key}"}
    resp = requests.post(p["url"], headers=headers,
                         json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.3}},
                         timeout=45)
    resp.raise_for_status()
    data = resp.json()
    return data[0].get("generated_text", "") if isinstance(data, list) else str(data)

def _parse_json(raw):
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                raw = p
                break
    return json.loads(raw)

def get_provider_status():
    result = {}
    for name, cfg in PROVIDERS.items():
        key = os.environ.get(cfg["env_key"])
        result[name] = {
            "name":       cfg["name"],
            "configured": bool(key),
            "key_hint":   f"...{key[-4:]}" if key else "not set",
            "models":     cfg["models"],
            "env_var":    cfg["env_key"],
        }
    return result

def analyze_symbol(symbol: str, market_data: dict, signals: dict) -> dict:
    pname, p = _get_provider()
    if not p:
        return {"error": "No LLM provider configured. Add GROQ_API_KEY to Railway env vars.",
                "prediction": "NEUTRAL", "confidence": 0, "reasoning": "No provider.",
                "key_levels": {"support": 0, "resistance": 0}, "sentiment": "unknown",
                "risk_note": "Add a free API key.", "provider": "none"}

    prompt = f"""Analyze this stock and respond ONLY with valid JSON (no markdown):

Symbol: {symbol}
Price: ${market_data.get('price','N/A')}  Change: {market_data.get('change_pct','N/A')}%
RSI: {market_data.get('rsi','N/A')}  MACD: {market_data.get('macd_signal','N/A')}
EMA50: {market_data.get('ema50','N/A')}  VWAP: {market_data.get('vwap','N/A')}
BB: {market_data.get('bb_position','N/A')}  Volume: {market_data.get('volume','N/A')}
Overall Signal: {signals.get('overall','N/A')}
Timeframe signals: {json.dumps(signals.get('timeframes',{}))}

Required JSON format:
{{
  "prediction": "BULLISH" or "BEARISH" or "NEUTRAL",
  "confidence": 0-100,
  "reasoning": "2-3 sentences",
  "key_levels": {{"support": float, "resistance": float}},
  "sentiment": "strong_bull" or "bull" or "neutral" or "bear" or "strong_bear",
  "short_term_bias": "UP" or "DOWN" or "SIDEWAYS",
  "suggested_action": "BUY" or "SELL" or "HOLD" or "WATCH",
  "risk_note": "one sentence"
}}"""

    try:
        if p.get("hf_mode"):
            raw = _call_hf(p, SYSTEM_PROMPT + "\n\n" + prompt)
            start = raw.find("{"); end = raw.rfind("}") + 1
            raw = raw[start:end] if start != -1 else "{}"
            result = json.loads(raw)
        else:
            messages = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
            raw = _call_openai(p, p["default_model"], messages, max_tokens=512)
            result = _parse_json(raw)
        result["provider"] = p["name"]
        result["model"]    = p["default_model"]
        result["timestamp_utc"] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        logger.error(f"MirrorFish analyze error: {e}")
        return {"prediction":"NEUTRAL","confidence":0,"reasoning":str(e),
                "key_levels":{"support":0,"resistance":0},"sentiment":"unknown",
                "risk_note":"API error.","provider":p["name"],"error":str(e)}

def analyze_portfolio(positions: list, closed_trades: list) -> dict:
    pname, p = _get_provider()
    if not p:
        return {"error": "No provider configured.", "summary": "Add GROQ_API_KEY.", "provider": "none"}

    pos_text = "\n".join(
        f"  {pos.get('symbol')}: {pos.get('qty')} shares, ${float(pos.get('market_value') or 0):.2f}, "
        f"PnL {float(pos.get('unrealized_plpc') or 0)*100:.2f}%"
        for pos in positions[:10]
    ) or "  (no open positions)"

    wins     = [t for t in closed_trades if float(t.get("pnl") or 0) > 0]
    losses   = [t for t in closed_trades if float(t.get("pnl") or 0) <= 0]
    total_pnl= sum(float(t.get("pnl") or 0) for t in closed_trades)
    win_rate = round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0

    prompt = f"""Analyze this trading portfolio and respond ONLY with valid JSON:

Open Positions:
{pos_text}

Recent Performance ({len(closed_trades)} closed trades):
  Win rate: {win_rate}%  Total P&L: ${total_pnl:.2f}
  Wins: {len(wins)}, Losses: {len(losses)}

Required JSON:
{{
  "portfolio_health": "STRONG" or "HEALTHY" or "MIXED" or "WEAK",
  "summary": "2-3 sentence portfolio commentary",
  "concentration_risk": "brief note",
  "strategy_assessment": "is the algo strategy working?",
  "top_suggestion": "one actionable suggestion",
  "win_rate_comment": "brief comment on {win_rate}% win rate"
}}"""

    try:
        messages = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
        raw = _call_openai(p, p["default_model"], messages, max_tokens=400)
        result = _parse_json(raw)
        result["provider"] = p["name"]
        return result
    except Exception as e:
        logger.error(f"MirrorFish portfolio error: {e}")
        return {"portfolio_health":"UNKNOWN","summary":str(e),"provider":p["name"],"error":str(e)}

def chat(message: str, context: dict = None) -> str:
    pname, p = _get_provider()
    if not p:
        return ("MirrorFish is not configured. Add GROQ_API_KEY (free at console.groq.com) "
                "to your Railway environment variables.")
    ctx = f"\n\nContext: {json.dumps(context, default=str)[:400]}" if context else ""
    messages = [{"role":"system","content":SYSTEM_PROMPT + ctx},{"role":"user","content":message}]
    try:
        if p.get("hf_mode"):
            return _call_hf(p, SYSTEM_PROMPT + "\n\nUser: " + message)
        return _call_openai(p, p["default_model"], messages, max_tokens=600, temperature=0.5)
    except Exception as e:
        logger.error(f"MirrorFish chat error: {e}")
        return f"MirrorFish error: {str(e)}"
