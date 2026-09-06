"""
research/finnhub_client.py
Thin wrapper around Finnhub's free-tier endpoints:
  - Company news (/company-news)
  - Insider transactions, Form 3/4/5 (/stock/insider-transactions)

NOTE: Finnhub's fund/institutional-ownership endpoint (13F/13D/13G data) is
PREMIUM-ONLY -- confirmed against their docs. This client deliberately does
NOT call that endpoint; a "your portfolio disclosures" tracker would need
either a paid Finnhub tier or a separate free source (e.g. parsing raw 13F
filings from SEC EDGAR directly).

Env var: FINNHUB_API_KEY (get one free, no card, at finnhub.io)
"""
import os
import time
from datetime import datetime, timedelta
import requests

from core.logger import get_logger

logger = get_logger(__name__)

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_URL = "https://finnhub.io/api/v1"

_last_call_at = 0.0
_MIN_INTERVAL = 0.15  # ~6-7 calls/sec, comfortably under the free 30-60/min-ish limits


def _throttled_get(path: str, params: dict):
    global _last_call_at
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY is not set. Get a free key at finnhub.io and set it in your env vars.")
    wait = _MIN_INTERVAL - (time.time() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    params = {**params, "token": FINNHUB_API_KEY}
    resp = requests.get(f"{FINNHUB_URL}{path}", params=params, timeout=15)
    _last_call_at = time.time()
    if resp.status_code == 429:
        raise RuntimeError("Finnhub rate limit hit (429). Slow down or wait a minute.")
    resp.raise_for_status()
    return resp.json()


def get_company_news(symbol: str, days: int = 7) -> list:
    """Recent headlines for a symbol. Returns [] on any failure (never raises
    into a caller that's mid-batch-scan -- one bad symbol shouldn't kill the run)."""
    try:
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=days)
        data = _throttled_get("/company-news", {
            "symbol": symbol.upper(),
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        })
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"finnhub news failed for {symbol}: {e}")
        return []


def get_insider_transactions(symbol: str, days: int = 90) -> list:
    """Form 3/4/5 insider transactions in the last `days`. Returns [] on failure."""
    try:
        data = _throttled_get("/stock/insider-transactions", {"symbol": symbol.upper()})
        rows = data.get("data", []) if isinstance(data, dict) else []
        cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
        return [r for r in rows if (r.get("transactionDate") or "") >= cutoff]
    except Exception as e:
        logger.warning(f"finnhub insider transactions failed for {symbol}: {e}")
        return []


def summarize_insider_activity(transactions: list) -> dict:
    """Turn raw transaction rows into a compact buy/sell summary (net shares,
    counts) -- this is what actually goes in the LLM prompt, not raw JSON."""
    buys = [t for t in transactions if (t.get("change") or 0) > 0]
    sells = [t for t in transactions if (t.get("change") or 0) < 0]
    net_shares = sum(t.get("change") or 0 for t in transactions)
    return {
        "buy_count": len(buys), "sell_count": len(sells),
        "net_shares": net_shares,
        "recent_buyers": [t.get("name") for t in buys[:5]],
        "recent_sellers": [t.get("name") for t in sells[:5]],
    }
