"""Authenticated API for the focused research candidates, using the known-working V1 engines as the baseline.

The first pass intentionally uses the exact known-working V1 strategy engines and
V4/Alpaca data loader. We will improve each engine only after baseline trades are
visible and validated.
"""
import math
import pandas as pd
import time
import io
import csv
import json
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, session, send_file

from core.market_data import alpaca_get_bars, get_bars
from research.xauusd_pullback_v1 import PullbackConfig, backtest as pullback_backtest
from research.xauusd_pullback_v2 import PullbackV2Config, backtest as pullback_v2_backtest
from research.xauusd_ema_retest_v2 import EMARetestV2Config, backtest as ema_v2_backtest
from research.xauusd_triple_rsi_v2 import TripleRSIV2Config, backtest as triple_v2_backtest
from research.xauusd_ema_retest_v1 import EMARetestConfig, backtest as ema_backtest
from research.xauusd_triple_rsi_v1 import TripleRSIConfig, backtest as triple_backtest
from research.xauusd_confluence_v4 import load_data as v4_load_data

bp = Blueprint("xauusd_research_v2", __name__)
_DATA_CACHE = {}
_CACHE_TTL = 300
_LAST_RUN = {"status": "never", "updated": None, "summary": {}}


def _safe(v):
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if hasattr(v, "item"):
        try:
            return _safe(v.item())
        except Exception:
            pass
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _cfg(cls, raw):
    raw = raw or {}
    return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def _bars_to_df(raw, symbol, timeframe):
    if not raw:
        raise ValueError(f"No {timeframe} market data returned for {symbol}")

    df = pd.DataFrame(raw)
    rename = {"t": "timestamp", "o": "Open", "h": "High", "l": "Low",
              "c": "Close", "v": "Volume"}
    df = df.rename(columns=rename)

    # Accept both Alpaca's compact keys and any normalized bars returned by a
    # fallback provider.
    required = {"timestamp", "Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Market-data format error for {symbol} {timeframe}: "
            f"missing {sorted(missing)}; columns={list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("Open", "High", "Low", "Close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    df = (
        df.dropna(subset=["timestamp", "Open", "High", "Low", "Close"])
          .set_index("timestamp")
          .sort_index()
    )
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _intraday(symbol, n):
    """Load enough 5m history for V2, with a proven paginated fallback."""
    n = max(500, min(30000, int(n)))
    key = ("5m", symbol, n)
    cached = _DATA_CACHE.get(key)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1].copy(), cached[2]
    try:
        raw = alpaca_get_bars(symbol, "5Min", limit=n)
        if raw:
            df = _bars_to_df(raw, symbol, "5m")
            if len(df) >= 500:
                out = df.tail(n)
                _DATA_CACHE[key] = (time.time(), out.copy(), "Alpaca")
                return out, "Alpaca"
    except Exception:
        pass

    try:
        df = v4_load_data(use_live=True, n_bars=n, symbol=symbol,
                          data_source="alpaca")["m5"]
        if df is not None and len(df) >= 500:
            out = df.tail(n)
            _DATA_CACHE[key] = (time.time(), out.copy(), "Alpaca/V4 paginated fallback")
            return out, "Alpaca/V4 paginated fallback"
        fallback_error = "fallback returned insufficient bars"
    except Exception as exc:
        fallback_error = str(exc)

    raise ValueError(
        f"V2 could not obtain enough 5m history for {symbol}. "
        f"Need at least 500 bars. {fallback_error}"
    )


def _daily(symbol, n):
    n = max(250, min(5000, int(n)))
    key = ("1d", symbol, n)
    cached = _DATA_CACHE.get(key)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1].copy(), cached[2]
    raw = alpaca_get_bars(symbol, "1Day", limit=n)
    source = "Alpaca"
    if not raw:
        raw = get_bars(symbol, "1y")
        source = "Analyzer Pro fallback"
    out = _bars_to_df(raw, symbol, "1Day").tail(n)
    _DATA_CACHE[key] = (time.time(), out.copy(), source)
    return out, source


@bp.route("/api/xauusd-research-v2/all", methods=["POST"])
def run_all():
    """Run all V2 candidates while sharing the expensive market-data fetch."""
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "GLD")).strip().upper()
    bars = int(body.get("n_bars", 5000))
    daily_bars = int(body.get("daily_bars", 250))
    risk = float(body.get("risk_pct", 0.5))
    try:
        m5, intraday_source = _intraday(symbol, bars)
        out = {}
        for name, cls, fn, extra in (
            ("pullback", PullbackConfig, pullback_backtest, {}),
            ("ema", EMARetestConfig, ema_backtest, {"rr": 2.5}),
        ):
            raw_cfg = dict(extra)
            raw_cfg["risk_pct"] = risk
            result = fn(m5, _cfg(cls, raw_cfg))
            trades = result.get("trades", [])
            if hasattr(trades, "to_dict"):
                trades = trades.to_dict(orient="records")
            signals = result.get("signals")
            if hasattr(signals, "tail"):
                signals = signals.tail(500).reset_index().to_dict(orient="records")
            out[name] = _safe({
                "strategy": name + " baseline", "symbol": symbol,
                "data_source": intraday_source, "bars": len(m5),
                "metrics": result.get("metrics", {}), "trades": trades,
                "signals": signals, "diagnostics": result.get("diagnostics", {}),
            })

        daily, daily_source = _daily(symbol, daily_bars)
        triple_cfg = _cfg(TripleRSIConfig, {
            "risk_pct": risk, "require_reversal": True
        })
        result = triple_backtest(daily, triple_cfg)
        trades = result.get("trades", [])
        if hasattr(trades, "to_dict"):
            trades = trades.to_dict(orient="records")
        signals = result.get("signals")
        if hasattr(signals, "tail"):
            signals = signals.tail(500).reset_index().to_dict(orient="records")
        out["triple"] = _safe({
            "strategy": "triple baseline", "symbol": symbol,
            "data_source": daily_source, "bars": len(daily),
            "metrics": result.get("metrics", {}), "trades": trades,
            "signals": signals, "diagnostics": result.get("diagnostics", {}),
        })
        return jsonify({"results": out})
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}",
                        "strategy": "all V2", "symbol": symbol}), 500


@bp.route("/api/xauusd-research-v2/status", methods=["GET"])
def status():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_safe(_LAST_RUN))




@bp.route("/api/strategy-lab/export", methods=["POST"])
def export_strategy_lab():
    """Export Alpaca OHLCV data plus the completed V2 strategy trades as CSV."""
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "GLD")).strip().upper()
    timeframe = str(body.get("timeframe", "5m")).strip()
    bars = max(100, min(100000, int(body.get("bars", 5000))))
    strategies = body.get("strategies", []) or []
    tf_map = {"5m":"5Min","15m":"15Min","1h":"1Hour","1d":"1Day","1D":"1Day","1w":"1Week","1W":"1Week"}
    alpaca_tf = tf_map.get(timeframe)
    if not alpaca_tf:
        return jsonify({"error": f"Unsupported export timeframe: {timeframe}"}), 400
    try:
        raw = alpaca_get_bars(symbol, alpaca_tf, limit=bars)
        if not raw:
            return jsonify({"error": f"No Alpaca {timeframe} data returned for {symbol}"}), 404
        exported = datetime.now(timezone.utc).isoformat()
        rows = [{
            "exported_at_utc": exported, "record_type": "market_data", "strategy": "",
            "symbol": symbol, "timeframe": timeframe, "timestamp": b.get("t", ""),
            "open": b.get("o"), "high": b.get("h"), "low": b.get("l"),
            "close": b.get("c"), "volume": b.get("v")
        } for b in raw]
        for item in strategies:
            if not isinstance(item, dict):
                continue
            name = str(item.get("strategy") or item.get("name") or "").strip()
            for trade in (item.get("trades") or []):
                if not isinstance(trade, dict):
                    continue
                row = {"exported_at_utc": exported, "record_type": "trade",
                       "strategy": name, "symbol": symbol, "timeframe": timeframe}
                for key, value in trade.items():
                    row[str(key)] = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                rows.append(row)
        base = ["exported_at_utc","record_type","strategy","symbol","timeframe",
                "timestamp","open","high","low","close","volume"]
        extra = sorted({k for row in rows for k in row} - set(base))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=base + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        payload = io.BytesIO(output.getvalue().encode("utf-8-sig"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return send_file(payload, mimetype="text/csv; charset=utf-8", as_attachment=True,
                         download_name=f"{symbol}_strategy_lab_v2_{timeframe}_{stamp}.csv")
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

@bp.route("/api/xauusd-research-v2/<strategy>", methods=["POST"])
def run(strategy):
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    strategy = str(strategy).strip().lower()
    symbol = str(body.get("symbol", "GLD")).strip().upper()

    try:
        if strategy == "triple":
            daily_bars = int(body.get("daily_bars", 250))
            df, source = _daily(symbol, daily_bars)
            cfg = _cfg(TripleRSIV2Config, body.get("config"))
            result = triple_v2_backtest(df, cfg)
        elif strategy == "pullback":
            bars = int(body.get("n_bars", 5000))
            df, source = _intraday(symbol, bars)
            cfg = _cfg(PullbackV2Config, body.get("config"))
            result = pullback_v2_backtest(df, cfg)
        elif strategy == "ema":
            bars = int(body.get("n_bars", 15600))
            df, source = _intraday(symbol, bars)
            cfg = _cfg(EMARetestV2Config, body.get("config"))
            result = ema_v2_backtest(df, cfg)
        else:
            return jsonify({"error": f"Unknown V2 strategy: {strategy}"}), 400

        trades = result.get("trades", [])
        if hasattr(trades, "to_dict"):
            trades = trades.to_dict(orient="records")

        signals = result.get("signals")
        if hasattr(signals, "tail"):
            signals = (
                signals.tail(500)
                .reset_index()
                .to_dict(orient="records")
            )

        return jsonify(_safe({
            "strategy": strategy + " baseline",
            "symbol": symbol,
            "data_source": source,
            "bars": len(df),
            "metrics": result.get("metrics", {}),
            "trades": trades,
            "signals": signals,
            "diagnostics": result.get("diagnostics", {}),
        }))
    except Exception as exc:
        # Always return JSON so the V2 page can display the real backend
        # failure instead of showing a generic "not running" message.
        return jsonify({
            "error": f"{type(exc).__name__}: {exc}",
            "strategy": strategy + " V2",
            "symbol": symbol,
        }), 500
