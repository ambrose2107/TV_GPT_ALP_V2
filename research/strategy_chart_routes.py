"""Shared interactive OHLC chart data for strategy research pages.

Uses Analyzer Pro's existing Alpaca market-data path. The endpoint deliberately
caps chart history so Plotly stays responsive on mobile while keeping enough
bars for multi-month failure analysis.
"""
import math
import time
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, session

from core.market_data import get_bars, alpaca_get_bars

strategy_chart_bp = Blueprint("strategy_chart", __name__)

_CHART_CACHE = {}
_CHART_CACHE_TTL = 20
_CHART_CAPS = {"5m": 6000, "15m": 6000, "1h": 3000, "1d": 1500}


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
    timeframe = str(body.get("timeframe", "15m")).lower()
    if timeframe not in _CHART_CAPS:
        timeframe = "15m"

    requested = max(200, min(10000, int(body.get("n_bars", 3000))))
    bars = min(requested, _CHART_CAPS[timeframe])

    cache_key = (symbol, timeframe, bars)
    cached = _CHART_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _CHART_CACHE_TTL:
        return jsonify(cached[1])

    tf_map = {"5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day"}
    raw = alpaca_get_bars(symbol, tf_map[timeframe], limit=bars)
    source = "Alpaca"
    if not raw:
        period = {"5m": "5m", "15m": "15m", "1h": "1h", "1d": "3mo"}[timeframe]
        raw = get_bars(symbol, period)
        source = "Analyzer Pro/Yahoo fallback"
    if not raw:
        return jsonify({"error": f"No chart data available for {symbol}"}), 404

    df = pd.DataFrame(raw)
    if df.empty:
        return jsonify({"error": f"No chart data available for {symbol}"}), 404

    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.set_index("t").sort_index()
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if timeframe != "5m":
        rule = {"15m": "15min", "1h": "1h", "1d": "1D"}[timeframe]
        df = df.resample(rule).agg(agg).dropna()
    df = df.tail(bars)

    close = df["Close"].astype(float)
    df["EMA20"] = _ema(close, 20)
    df["EMA50"] = _ema(close, 50)
    df["EMA200"] = _ema(close, 200)

    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    day = pd.Series(df.index.date, index=df.index)
    volume = df["Volume"].astype(float)
    pv = typical * volume
    df["VWAP"] = pv.groupby(day).cumsum() / volume.groupby(day).cumsum().replace(0, np.nan)

    payload = _safe({
        "symbol": symbol,
        "timeframe": timeframe,
        "data_source": source,
        "bars": int(len(df)),
        "data_start": df.index[0].isoformat() if len(df) else None,
        "data_end": df.index[-1].isoformat() if len(df) else None,
        "ohlcv": [
            {"t": idx.isoformat(), "o": float(row.Open), "h": float(row.High),
             "l": float(row.Low), "c": float(row.Close), "v": float(row.Volume)}
            for idx, row in df.iterrows()
        ],
        "indicators": {
            "ema20": [None if pd.isna(x) else float(x) for x in df["EMA20"]],
            "ema50": [None if pd.isna(x) else float(x) for x in df["EMA50"]],
            "ema200": [None if pd.isna(x) else float(x) for x in df["EMA200"]],
            "vwap": [None if pd.isna(x) else float(x) for x in df["VWAP"]],
        },
    })

    _CHART_CACHE[cache_key] = (time.time(), payload)
    if len(_CHART_CACHE) > 12:
        oldest = sorted(_CHART_CACHE.items(), key=lambda kv: kv[1][0])[:len(_CHART_CACHE) - 12]
        for key, _ in oldest:
            _CHART_CACHE.pop(key, None)

    return jsonify(payload)


__all__ = ["strategy_chart_bp"]
