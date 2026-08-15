"""
research/kronos_predictor.py
Calls a separately-hosted Kronos Forecast API (deployed as a free Hugging
Face Space) instead of running torch/Kronos inside this app. This keeps
your main trading bot lightweight (works fine on Render's free 512MB tier)
while still getting real Kronos forecasts over HTTP.

Deploy the API first (see the kronos-api-space files provided separately),
then set KRONOS_API_URL below.

Env vars:
  KRONOS_API_URL = https://YOUR-USERNAME-kronos-forecast-api.hf.space
                   (no trailing slash; get this from your HF Space page)
"""
import os
import requests
from core.logger import get_logger
from core.market_data import get_bars

logger = get_logger(__name__)

KRONOS_API_URL = os.environ.get("KRONOS_API_URL", "").rstrip("/")


def get_kronos_forecast(symbol: str, period: str = "1D",
                         lookback: int = 400, pred_len: int = 20,
                         sample_count: int = 1, model: str = "mini") -> dict:
    """
    Fetch real OHLCV bars for `symbol` and forecast the next `pred_len`
    bars by calling the hosted Kronos API.
    """
    if not KRONOS_API_URL:
        return {
            "error": "KRONOS_API_URL is not set. Deploy the Kronos API to a "
                     "free Hugging Face Space (see README) and set "
                     "KRONOS_API_URL in your .env / Render env vars."
        }

    bars = get_bars(symbol, period)
    if not bars or len(bars) < 50:
        return {"error": f"Not enough bar data for {symbol} ({len(bars) if bars else 0} bars)"}

    if len(bars) > lookback:
        bars = bars[-lookback:]

    payload = {
        "bars": [{"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b.get("v", 0)} for b in bars],
        "pred_len": pred_len,
        "period": period if period in ("5m", "15m", "1h", "4h", "1D") else "1D",
        "model": model,
        "sample_count": sample_count,
    }

    try:
        resp = requests.post(f"{KRONOS_API_URL}/predict", json=payload, timeout=90)
    except requests.exceptions.ConnectionError:
        return {"error": f"Could not reach Kronos API at {KRONOS_API_URL}. "
                          f"Is the Hugging Face Space awake? (free Spaces sleep when idle — "
                          f"the first call after sleeping can take 20-30s while it wakes up, "
                          f"try again in a moment)"}
    except requests.exceptions.Timeout:
        return {"error": "Kronos API timed out — the Space may be waking from sleep. Try again."}

    if resp.status_code != 200:
        logger.error(f"Kronos API {resp.status_code}: {resp.text[:300]}")
        return {"error": f"Kronos API error {resp.status_code}: {resp.text[:300]}"}

    data = resp.json()
    data["symbol"] = symbol.upper()
    data["source"] = "kronos-api"
    data["note"] = ("Raw model forecast, not financial advice. Kronos is trained "
                     "for pattern continuation, not fundamentals or news.")
    return data
