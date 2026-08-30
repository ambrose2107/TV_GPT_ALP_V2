"""
research/signal_engine.py
Batch signal scanner: for a set of symbols (open positions + a watchlist +
insider/options-flow candidates), runs AI analysis + Kronos forecast, scores
their agreement ("confluence"), and stores results via core/database.py.

Designed to run as a BACKGROUND job (see core/scheduler.py) -- a full scan
can take many minutes (Kronos alone can take 30-100s per symbol on a free
CPU tier), and this bot's gunicorn config uses a single worker, so running
this synchronously inside a request would freeze webhook handling for the
whole scan. Never call run_full_scan() directly from a Flask route handler;
always hand it to a background thread (core/scheduler.py does this).
"""
import time
import json
from datetime import datetime

from core.logger import get_logger
from core.database import save_signal
from research.ollama_analysis import get_local_ai_analysis
from research.kronos_predictor import get_kronos_forecast

logger = get_logger(__name__)

# Kept short deliberately: each symbol costs one LLM call + one Kronos call
# (the slow part). A long default watchlist turns a "scan" into a
# 20+ minute job on free-tier hardware.
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
]

# 1-2 week-ahead forecast: ~10 trading days.
KRONOS_PRED_LEN = 10


def _ai_bucket(ai_score: float) -> str:
    if ai_score is None:
        return "Unknown"
    if ai_score <= 3.5:
        return "Bull"
    if ai_score >= 6.5:
        return "Bear"
    return "Neutral"


def _kronos_bucket(implied_pct: float) -> str:
    if implied_pct is None:
        return "Unknown"
    if implied_pct >= 1.5:
        return "Bull"
    if implied_pct <= -1.5:
        return "Bear"
    if implied_pct >= 0.3:
        return "Mild Bull"
    if implied_pct <= -0.3:
        return "Mild Bear"
    return "Neutral"


def _confluence(ai_score: float, kronos_pct: float):
    """
    Combine the analyzer's quantitative technical score (0-10, lower =
    more bullish) with Kronos's implied % move into one label + numeric
    score. Uses the analyzer's own number, not the LLM's prose, for the
    AI side -- prose isn't reliable to parse for sentiment, the underlying
    indicator score already is.
    """
    ai_b = _ai_bucket(ai_score)
    kr_b = _kronos_bucket(kronos_pct)

    ai_component = (5 - ai_score) if ai_score is not None else 0        # ~ -5..+5, + = bullish
    kr_component = max(-5, min(5, kronos_pct)) if kronos_pct is not None else 0
    score = round((ai_component + kr_component) / 2, 2)

    if ai_b == "Bull" and kr_b in ("Bull",):
        label = "Strong Buy"
    elif ai_b == "Bull" and kr_b in ("Mild Bull",) or (ai_b == "Neutral" and kr_b == "Bull"):
        label = "Buy"
    elif ai_b == "Bear" and kr_b in ("Bear",):
        label = "Strong Sell"
    elif ai_b == "Bear" and kr_b in ("Mild Bear",) or (ai_b == "Neutral" and kr_b == "Bear"):
        label = "Sell"
    elif (ai_b == "Bull" and kr_b in ("Bear", "Mild Bear")) or (ai_b == "Bear" and kr_b in ("Bull", "Mild Bull")):
        label = "Mixed"   # AI and Kronos disagree -- lower-conviction setup
    else:
        label = "Neutral"

    return label, score


def scan_one_symbol(symbol: str, category: str, note: str = None) -> dict:
    """Run AI + Kronos for one symbol and return a row ready for save_signal()."""
    symbol = symbol.upper()
    row = {
        "symbol": symbol, "category": category, "price": None,
        "ai_label": None, "ai_score": None, "ai_text": None,
        "kronos_last_close": None, "kronos_implied_pct": None, "kronos_forecast_json": None,
        "confluence_label": "Unknown", "confluence_score": 0, "note": note,
    }

    try:
        ai = get_local_ai_analysis(symbol, ["1D"])
        indicators = ai.get("indicators", {})
        row["price"] = indicators.get("price")
        row["ai_score"] = indicators.get("overall_score")
        row["ai_label"] = indicators.get("overall_label")
        row["ai_text"] = ai.get("analysis")
    except Exception as e:
        logger.warning(f"signal_engine AI step failed for {symbol}: {e}")
        row["ai_text"] = f"AI analysis failed: {e}"

    try:
        kr = get_kronos_forecast(symbol, period="1D", pred_len=KRONOS_PRED_LEN, lookback=300)
        if "error" not in kr:
            row["kronos_last_close"] = kr.get("last_close")
            row["kronos_implied_pct"] = kr.get("implied_move_pct")
            row["kronos_forecast_json"] = json.dumps(kr.get("forecast", []))
        else:
            row["note"] = ((row["note"] + " | ") if row["note"] else "") + f"Kronos: {kr['error']}"
    except Exception as e:
        logger.warning(f"signal_engine Kronos step failed for {symbol}: {e}")
        row["note"] = ((row["note"] + " | ") if row["note"] else "") + f"Kronos failed: {e}"

    row["confluence_label"], row["confluence_score"] = _confluence(row["ai_score"], row["kronos_implied_pct"])
    return row


def _get_open_position_symbols() -> list:
    try:
        from brokers.alpaca_adapter import AlpacaAdapter
        alpaca = AlpacaAdapter()
        positions = alpaca.get_positions() or []
        return [p["symbol"].upper() for p in positions if p.get("symbol")]
    except Exception as e:
        logger.warning(f"signal_engine: could not fetch open positions: {e}")
        return []


def _get_candidate_symbols(limit: int = 5) -> list:
    """Symbols flagged by unusual options flow / recent insider Form 4 buys."""
    try:
        from research.insider_flow import get_confluence_stocks
        data = get_confluence_stocks()
        return [c["symbol"] for c in data.get("confluence_stocks", [])[:limit]]
    except Exception as e:
        logger.warning(f"signal_engine: candidate scan failed: {e}")
        return []


def run_full_scan(watchlist: list = None, include_positions: bool = True,
                   include_candidates: bool = True, throttle_seconds: float = 2.0) -> dict:
    """
    The actual batch job. SLOW by design (real LLM + real Kronos calls per
    symbol) -- always run this from a background thread, never inline in a
    request handler. throttle_seconds paces requests to stay under Groq's
    free-tier rate limit when AI_PROVIDER=groq.
    """
    started = time.time()
    watchlist = watchlist or DEFAULT_WATCHLIST

    plan = []  # list of (symbol, category, note)
    if include_positions:
        for sym in _get_open_position_symbols():
            plan.append((sym, "position", None))
    for sym in watchlist:
        if sym not in [p[0] for p in plan]:
            plan.append((sym, "watchlist", None))
    if include_candidates:
        for sym in _get_candidate_symbols():
            if sym not in [p[0] for p in plan]:
                plan.append((sym, "candidate", "Flagged by options flow / insider buys"))

    results = []
    for i, (sym, category, note) in enumerate(plan):
        logger.info(f"signal_engine: scanning {sym} ({category}) [{i+1}/{len(plan)}]")
        row = scan_one_symbol(sym, category, note)
        save_signal(row)
        results.append(row)
        if i < len(plan) - 1:
            time.sleep(throttle_seconds)

    duration = round(time.time() - started, 1)
    summary = {
        "scanned": len(plan),
        "positions": sum(1 for r in results if r["category"] == "position"),
        "watchlist": sum(1 for r in results if r["category"] == "watchlist"),
        "candidates": sum(1 for r in results if r["category"] == "candidate"),
        "duration_sec": duration,
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    logger.info(f"signal_engine: scan complete — {summary}")
    return summary
