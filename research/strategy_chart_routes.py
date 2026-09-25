"""Shared interactive OHLC chart data for strategy research pages.

Uses Analyzer Pro's existing Alpaca market-data path. The endpoint deliberately
caps chart history so Plotly stays responsive on mobile while keeping enough
bars for multi-month failure analysis.
"""
import math
import time
import csv
import io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, session, make_response

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


@strategy_chart_bp.route("/api/strategy-lab/export", methods=["POST"])
def strategy_lab_export():
    """Download raw Alpaca OHLCV plus all strategy trades in one CSV.

    The market-data rows and trade rows use record_type so the file remains
    easy to filter in Excel/Pandas while preserving the exact trade records
    returned by the browser's completed backtests.
    """
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "GLD")).strip().upper()
    timeframe = str(body.get("timeframe", "1Day")).strip()
    allowed = {"1Min", "2Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day", "1Week"}
    if timeframe not in allowed:
        return jsonify({"error": "Unsupported timeframe"}), 400

    try:
        bars_requested = max(100, min(100000, int(body.get("bars", 10000))))
    except (TypeError, ValueError):
        bars_requested = 10000

    raw = alpaca_get_bars(symbol, timeframe, limit=bars_requested)
    if not raw:
        return jsonify({
            "error": f"No Alpaca data available for {symbol} {timeframe}. "
                     "Export requires Alpaca market data."
        }), 404

    strategies = body.get("strategies") or {}
    headers = [
        "exported_at_utc", "record_type", "strategy", "symbol",
        "timeframe", "data_source", "timestamp",
        "open", "high", "low", "close", "volume",
        "entry_time", "exit_time", "side", "entry", "sl", "tp", "R", "equity",
        "reason"
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    exported = datetime.now(timezone.utc).isoformat()

    for b in raw:
        writer.writerow({
            "exported_at_utc": exported,
            "record_type": "market_data",
            "symbol": symbol,
            "timeframe": timeframe,
            "data_source": "Alpaca",
            "timestamp": b.get("t"),
            "open": b.get("o"),
            "high": b.get("h"),
            "low": b.get("l"),
            "close": b.get("c"),
            "volume": b.get("v"),
        })

    trade_count = 0
    for strategy, trades in strategies.items():
        for t in (trades or []):
            writer.writerow({
                "exported_at_utc": exported,
                "record_type": "trade",
                "strategy": strategy,
                "symbol": symbol,
                "timeframe": timeframe,
                "data_source": "strategy backtest",
                "entry_time": t.get("entry_time"),
                "exit_time": t.get("exit_time"),
                "side": t.get("side"),
                "entry": t.get("entry"),
                "sl": t.get("sl"),
                "tp": t.get("tp"),
                "R": t.get("R"),
                "equity": t.get("equity"),
                "reason": t.get("reason"),
            })
            trade_count += 1

    filename = f"strategy_lab_{symbol}_{timeframe}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    response = make_response(out.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["X-Market-Data-Bars"] = str(len(raw))
    response.headers["X-Strategy-Trades"] = str(trade_count)
    return response


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
