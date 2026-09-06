"""
research/ai_stock_picker.py
"Research tracker": for a universe of symbols, pulls REAL recent news
headlines + REAL insider Form 3/4/5 transactions from Finnhub, then asks the
configured LLM (Groq or Ollama, via research/ollama_analysis._call_llm) to
weigh that real evidence into a verdict. The LLM never invents headlines or
transactions -- it only interprets what was actually fetched.

This is a genuinely different job from research/signal_engine.py: that one
tracks known positions/watchlist technicals, this one surfaces NEW ideas
from news/insider flow you might not already be watching.

Does NOT use Finnhub's institutional/13F ownership endpoint -- that's
premium-only on Finnhub (confirmed against their docs). If you want real
13F "smart money" tracking, that needs either a paid Finnhub tier or a
separate SEC EDGAR 13F parser (a real project on its own -- ask if you want
this built next).
"""
import time
from datetime import datetime

from core.logger import get_logger
from core.database import save_signal
from research.finnhub_client import get_company_news, get_insider_transactions, summarize_insider_activity
from research.ollama_analysis import _call_llm, _current_model_name, AI_PROVIDER

logger = get_logger(__name__)

# A liquid, well-covered universe to scan for new ideas. Kept short on
# purpose -- each symbol costs 2 Finnhub calls + 1 LLM call.
DEFAULT_RESEARCH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "NFLX", "CRM", "ORCL",
]

SYSTEM_PROMPT = (
    "You are a research analyst assistant. You are given REAL recent news "
    "headlines and REAL insider Form 4 transaction data for a stock -- "
    "never invent headlines, names, or numbers beyond what's given. If no "
    "news or insider data is provided, say so plainly rather than "
    "speculating. Weigh the evidence into one of exactly these verdicts: "
    "BUY, WATCH, or AVOID, with a one-line reason. This is not financial "
    "advice -- describe the setup only."
)


def _build_prompt(symbol: str, news: list, insider_summary: dict) -> str:
    lines = [f"Symbol: {symbol}", ""]
    if news:
        lines.append(f"Recent headlines ({len(news)} in the last 7 days):")
        for n in news[:8]:
            headline = (n.get("headline") or "").strip()
            if headline:
                lines.append(f"  - {headline}")
    else:
        lines.append("Recent headlines: none found in the last 7 days.")

    lines.append("")
    if insider_summary["buy_count"] or insider_summary["sell_count"]:
        lines.append(
            f"Insider activity (last 90 days): {insider_summary['buy_count']} buy filing(s), "
            f"{insider_summary['sell_count']} sell filing(s), net shares changed: {insider_summary['net_shares']:+d}."
        )
        if insider_summary["recent_buyers"]:
            lines.append(f"  Recent buyers: {', '.join(insider_summary['recent_buyers'])}")
        if insider_summary["recent_sellers"]:
            lines.append(f"  Recent sellers: {', '.join(insider_summary['recent_sellers'])}")
    else:
        lines.append("Insider activity (last 90 days): none filed.")

    lines.append("")
    lines.append("Give your verdict (BUY / WATCH / AVOID) and a one-line reason, based only on the above.")
    return "\n".join(lines)


def research_one_symbol(symbol: str) -> dict:
    symbol = symbol.upper()
    news = get_company_news(symbol, days=7)
    insider_txns = get_insider_transactions(symbol, days=90)
    insider_summary = summarize_insider_activity(insider_txns)

    prompt = _build_prompt(symbol, news, insider_summary)
    verdict_text = _call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=150)

    verdict = "WATCH"
    upper = verdict_text.upper()
    if upper.startswith("BUY") or "\nBUY" in upper or "VERDICT: BUY" in upper:
        verdict = "BUY"
    elif upper.startswith("AVOID") or "\nAVOID" in upper or "VERDICT: AVOID" in upper:
        verdict = "AVOID"

    return {
        "symbol": symbol, "category": "research_pick",
        "price": None, "ai_label": verdict, "ai_score": None, "ai_text": verdict_text,
        "kronos_last_close": None, "kronos_implied_pct": None, "kronos_forecast_json": None,
        "confluence_label": verdict, "confluence_score": {"BUY": 1, "WATCH": 0, "AVOID": -1}[verdict],
        "note": f"{len(news)} headlines, {insider_summary['buy_count']} insider buys / "
                f"{insider_summary['sell_count']} sells (last 90d)",
        "strategy_label": None, "strategy_reason": None,
    }


def run_research_scan(symbols: list = None, throttle_seconds: float = 1.5) -> dict:
    """
    SLOW by design (news + insider fetch + one LLM call per symbol) -- run
    from a background thread only, same as signal_engine.run_full_scan.
    """
    started = time.time()
    symbols = symbols or DEFAULT_RESEARCH_UNIVERSE
    results = []
    for i, sym in enumerate(symbols):
        logger.info(f"ai_stock_picker: researching {sym} [{i+1}/{len(symbols)}]")
        try:
            row = research_one_symbol(sym)
            save_signal(row)
            results.append(row)
        except Exception as e:
            logger.error(f"ai_stock_picker: {sym} failed: {e}")
        if i < len(symbols) - 1:
            time.sleep(throttle_seconds)

    duration = round(time.time() - started, 1)
    return {
        "scanned": len(results),
        "buys": sum(1 for r in results if r["ai_label"] == "BUY"),
        "watches": sum(1 for r in results if r["ai_label"] == "WATCH"),
        "avoids": sum(1 for r in results if r["ai_label"] == "AVOID"),
        "duration_sec": duration,
        "finished_at": datetime.utcnow().isoformat() + "Z",
        "model": _current_model_name(),
    }
