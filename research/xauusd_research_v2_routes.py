"""Authenticated API for the focused V2 research candidates.

V2 contains only the newer research engines. The older seven-strategy lab and
its legacy runner remain isolated in the V1 page/backend paths.
"""
import math
import pandas as pd
import time
import io
import csv
import json
import uuid
import itertools
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, session, send_file

from core.market_data import alpaca_get_bars, get_bars
from research.xauusd_pullback_v2 import PullbackV2Config, backtest as pullback_v2_backtest
from research.xauusd_ema_retest_v2 import EMARetestV2Config, backtest as ema_v2_backtest
from research.xauusd_daily_research_v2 import (
    WilliamsRConfig, CCIConfig, MultiHorizonRSIConfig,
    backtest_williams_r, backtest_cci, backtest_multi_rsi,
)
from research.xauusd_confluence_v4 import load_data as v4_load_data

bp = Blueprint("xauusd_research_v2", __name__)
_DATA_CACHE = {}
_CACHE_TTL = 300
_LAST_RUN = {"status": "never", "updated": None, "summary": {}}
_RUN_DATA = {}


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


@bp.route("/api/xauusd-research-v2/status", methods=["GET"])
def status():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_safe(_LAST_RUN))




@bp.route("/api/xauusd-research-v2/optimize", methods=["POST"])
def optimize():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    strategy = str(body.get("strategy", "pullback")).strip().lower()
    symbol = str(body.get("symbol", "SPY")).strip().upper()
    bars = max(1000, min(30000, int(body.get("n_bars", 15600))))
    train_pct = min(0.8, max(0.5, float(body.get("train_pct", 0.7))))
    min_train = max(5, int(body.get("min_train_trades", 10)))
    min_test = max(3, int(body.get("min_test_trades", 5)))
    try:
        df, source = _intraday(symbol, bars)
        split = max(500, min(len(df)-100, int(len(df) * train_pct)))
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()
        rows = []
        if strategy == "pullback":
            grid = list(itertools.product([0.05, 0.10], [0.15, 0.25, 0.35], [1.5, 2.0, 2.5], [6, 8, 12], [18, 19, 20, 21]))
            for buf, body_min, rr, cooldown, end_hour in grid:
                cfg = PullbackV2Config(breakout_buffer_atr=buf, min_body_atr=body_min, rr=rr, cooldown_bars=cooldown, session_end_utc=end_hour)
                a = pullback_v2_backtest(train, cfg)["metrics"]; b = pullback_v2_backtest(test, cfg)["metrics"]
                if a["num_trades"] < min_train or b["num_trades"] < min_test: continue
                ap, bp = float(a["profit_factor"] or 0), float(b["profit_factor"] or 0)
                rows.append({"breakout_buffer_atr":buf,"min_body_atr":body_min,"rr":rr,"cooldown_bars":cooldown,"session_end_utc":end_hour,"train_trades":a["num_trades"],"train_pf":ap,"train_expectancy_R":a["expectancy_R"],"test_trades":b["num_trades"],"test_pf":bp,"test_expectancy_R":b["expectancy_R"],"test_total_R":b["total_R"],"test_return_pct":b["total_return_pct"],"test_max_dd_pct":b["max_drawdown_pct"],"robust_score":min(ap,bp)+0.25*float(b["expectancy_R"])})
        elif strategy == "ema":
            grid = list(itertools.product([0.05,0.10,0.20], [0.05,0.10,0.20], [2.0,2.5,3.0], [1,2]))
            for gap, rej, rr, sep in grid:
                cfg = EMARetestV2Config(min_gap_atr=gap,min_rejection_body_atr=rej,rr=rr,separation_bars=sep)
                a = ema_v2_backtest(train,cfg)["metrics"]; b = ema_v2_backtest(test,cfg)["metrics"]
                if a["num_trades"] < min_train or b["num_trades"] < min_test: continue
                ap,bp=float(a["profit_factor"] or 0),float(b["profit_factor"] or 0)
                rows.append({"min_gap_atr":gap,"min_rejection_body_atr":rej,"rr":rr,"separation_bars":sep,"train_trades":a["num_trades"],"train_pf":ap,"train_expectancy_R":a["expectancy_R"],"test_trades":b["num_trades"],"test_pf":bp,"test_expectancy_R":b["expectancy_R"],"test_total_R":b["total_R"],"test_return_pct":b["total_return_pct"],"test_max_dd_pct":b["max_drawdown_pct"],"robust_score":min(ap,bp)+0.25*float(b["expectancy_R"])})
        else:
            return jsonify({"error":"Optimizer supports Pullback V2 and EMA 20/50 V2."}),400
        rows.sort(key=lambda x:(x["robust_score"],x["test_pf"],x["test_total_R"]),reverse=True)
        return jsonify(_safe({"strategy":strategy,"symbol":symbol,"data_source":source,"bars":len(df),"train_bars":len(train),"test_bars":len(test),"train_pct":train_pct,"tested":len(grid),"passed":len(rows),"results":rows[:25],"note":"Chronological train/test research only; not a future-performance guarantee."}))
    except Exception as exc:
        return jsonify({"error":f"{type(exc).__name__}: {exc}"}),500


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
        run_id = str(body.get("run_id", "")).strip()
        raw = None
        if run_id and run_id in _RUN_DATA:
            saved = _RUN_DATA[run_id]["df"]
            if timeframe == "5m":
                z = saved
            elif timeframe == "15m":
                z = saved.resample("15min", label="left", closed="left").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            elif timeframe == "1h":
                z = saved.resample("1h", label="left", closed="left").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            elif timeframe in ("1d","1D"):
                z = saved.resample("1D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            else:
                z = None
            if z is not None:
                raw = [{"t":ts.isoformat(),"o":float(row.Open),"h":float(row.High),"l":float(row.Low),"c":float(row.Close),"v":float(row.Volume)} for ts,row in z.iterrows()]
        if raw is None:
            raw = alpaca_get_bars(symbol, alpaca_tf, limit=bars)
        if not raw:
            return jsonify({"error": f"No Alpaca {timeframe} data returned for {symbol}"}), 404
        raw = raw[-bars:]
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
            daily_bars = int(body.get("daily_bars", 1000))
            df, source = _daily(symbol, daily_bars)
            cfg = _cfg(MultiHorizonRSIConfig, body.get("config"))
            result = backtest_multi_rsi(df, cfg)
        elif strategy in ("williams", "cci", "multi_rsi"):
            daily_bars = int(body.get("daily_bars", 1000))
            df, source = _daily(symbol, daily_bars)
            if strategy == "williams":
                cfg = _cfg(WilliamsRConfig, body.get("config"))
                result = backtest_williams_r(df, cfg)
            elif strategy == "cci":
                cfg = _cfg(CCIConfig, body.get("config"))
                result = backtest_cci(df, cfg)
            else:
                cfg = _cfg(MultiHorizonRSIConfig, body.get("config"))
                result = backtest_multi_rsi(df, cfg)
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

        run_id = uuid.uuid4().hex
        _RUN_DATA[run_id] = {"df":df.copy(),"symbol":symbol,"source":source,"created":time.time()}
        if len(_RUN_DATA) > 20:
            for key,_ in sorted(_RUN_DATA.items(),key=lambda kv:kv[1]["created"])[:5]:
                _RUN_DATA.pop(key,None)
        return jsonify(_safe({
            "run_id": run_id,
            "strategy": strategy + " V2",
            "symbol": symbol,
            "data_source": source,
            "bars": len(df),
            "data_start": df.index.min().isoformat(),
            "data_end": df.index.max().isoformat(),
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
