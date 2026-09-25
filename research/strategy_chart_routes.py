"""Shared interactive OHLC chart data for strategy research pages.

Uses the same Alpaca loader as the GLD research strategies, with a selected
ticker. Indicators are computed from the returned bars so the browser can
render an interactive candlestick chart without shipping base64 PNGs.
"""

import math
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, session

from core.market_data import get_bars

strategy_chart_bp = Blueprint("strategy_chart", __name__)


def _safe(v):
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, np.generic):
        return _safe(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


@strategy_chart_bp.route("/api/strategy-chart", methods=["POST"])
def strategy_chart():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "GLD")).strip().upper()
    bars = max(200, min(20000, int(body.get("n_bars", 1560))))
    timeframe = str(body.get("timeframe", "5m")).lower()
    if timeframe not in {"5m", "15m", "1h", "1d"}:
        timeframe = "5m"

    # Reuse Analyzer Pro's existing market-data path instead of maintaining
    # a second chart-specific downloader. This keeps the strategy-analysis
    # chart on the same source used by the Analyzer Pro tab.
    period = {"5m": "1D", "15m": "1D", "1h": "1W", "1d": "3mo"}[timeframe]
    raw = get_bars(symbol, period)
    if not raw:
        return jsonify({"error": f"No chart data available for {symbol}"}), 404

    df = pd.DataFrame(raw)
    if df.empty:
        return jsonify({"error": f"No chart data available for {symbol}"}), 404
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.set_index("t").sort_index()
    df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    if timeframe != "5m":
        rule = {"15m":"15min","1h":"1h","1d":"1D"}[timeframe]
        df = df.resample(rule).agg(agg).dropna()
    df = df.tail(bars)

    close = df["Close"].astype(float)
    df["EMA20"] = _ema(close, 20)
    df["EMA50"] = _ema(close, 50)
    df["EMA200"] = _ema(close, 200)

    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    day = pd.Series(df.index.date, index=df.index)
    pv = typical * df["Volume"].astype(float)
    df["VWAP"] = pv.groupby(day).cumsum() / df["Volume"].astype(float).groupby(day).cumsum().replace(0, np.nan)

    source = raw[0].get("source", "unknown") if raw else "unknown"
    return jsonify(_safe({
        "symbol": symbol,
        "timeframe": timeframe,
        "data_source": source,
        "bars": int(len(df)),
        "data_start": df.index[0].isoformat() if len(df) else None,
        "data_end": df.index[-1].isoformat() if len(df) else None,
        "ohlcv": [
            {"t": idx.isoformat(), "o": float(row.Open), "h": float(row.High), "l": float(row.Low),
             "c": float(row.Close), "v": float(row.Volume)}
            for idx, row in df.iterrows()
        ],
        "indicators": {
            "ema20": [None if pd.isna(x) else float(x) for x in df["EMA20"]],
            "ema50": [None if pd.isna(x) else float(x) for x in df["EMA50"]],
            "ema200": [None if pd.isna(x) else float(x) for x in df["EMA200"]],
            "vwap": [None if pd.isna(x) else float(x) for x in df["VWAP"]],
        },
    }))



__all__ = ["strategy_chart_bp"]
