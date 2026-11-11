"""TradingView-style historical data and bar-replay workspace.

Replay datasets are retained in the user's browser IndexedDB and can be exported
as CSV. Server endpoints are stateless; a replay request only analyzes the
dataset supplied by the browser.
"""
import io
import csv
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from flask import Blueprint, jsonify, request, session, render_template
from research.strategies import list_strategies, get_strategy

replay_bp = Blueprint("backtest_replay", __name__)

def _auth():
    if not session.get("logged_in"):
        return jsonify({"error":"Unauthorized"}), 401
    return None

def _safe_float(v):
    try:
        x=float(v)
        return x if pd.notna(x) else None
    except Exception:
        return None

def _bars_payload(df):
    out=[]
    for ts,row in df.iterrows():
        out.append({"time":pd.Timestamp(ts).isoformat(),
                    "open":float(row["Open"]),"high":float(row["High"]),
                    "low":float(row["Low"]),"close":float(row["Close"]),
                    "volume":(_safe_float(row.get("Volume",0)) or 0)})
    return out

def _dataset_frame(raw):
    if not isinstance(raw,list) or len(raw)<100:
        raise ValueError("Dataset must contain at least 100 OHLCV bars.")
    if len(raw)>12000:
        raise ValueError("Replay dataset is capped at 12,000 bars.")
    df=pd.DataFrame(raw)
    aliases={"o":"open","h":"high","l":"low","c":"close","v":"volume","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume","timestamp":"time","t":"time"}
    df=df.rename(columns=aliases)
    required={"time","open","high","low","close"}
    if not required.issubset(df.columns):
        raise ValueError("CSV/data needs time, open, high, low, close columns.")
    df.index=pd.to_datetime(df["time"],utc=True,errors="coerce")
    for key in ("open","high","low","close","volume"):
        if key not in df: df[key]=0
        df[key]=pd.to_numeric(df[key],errors="coerce")
    df=df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    df=df.dropna(subset=["Open","High","Low","Close"]).sort_index()
    if len(df)<100: raise ValueError("Not enough valid OHLCV rows after parsing.")
    return df[["Open","High","Low","Close","Volume"]]

@replay_bp.route("/backtest-replay")
def backtest_replay_page():
    if not session.get("logged_in"):
        return "Unauthorized",401
    return render_template("backtest_replay.html")

@replay_bp.route("/api/backtest/replay/strategies")
def replay_strategies():
    err=_auth()
    if err:return err
    return jsonify({"strategies":list_strategies()})

@replay_bp.route("/api/backtest/replay/data",methods=["POST"])
def replay_download_data():
    err=_auth()
    if err:return err
    body=request.get_json(silent=True) or {}
    symbol=str(body.get("symbol","GLD")).strip().upper()
    interval=str(body.get("interval","5m"))
    period=str(body.get("period","60d"))
    if not symbol or interval not in {"1m","2m","5m","15m","30m","1h","4h","1d"}:
        return jsonify({"error":"Select a valid symbol and interval."}),400
    try:
        from research.backtest_engine import fetch_yf_data
        df=fetch_yf_data(symbol,interval=interval,period=period)
        df=df.tail(12000)
        return jsonify({"symbol":symbol,"interval":interval,"period":period,
                        "source":"Alpaca-first / Yahoo fallback",
                        "bars":_bars_payload(df),
                        "data_range":{"start":str(df.index[0]),"end":str(df.index[-1])},
                        "count":len(df)})
    except Exception as exc:
        return jsonify({"error":str(exc)}),400

@replay_bp.route("/api/backtest/replay/analyze",methods=["POST"])
def replay_analyze():
    err=_auth()
    if err:return err
    body=request.get_json(silent=True) or {}
    strategy_id=str(body.get("strategy_id","fibonacci_golden_box"))
    try:
        df=_dataset_frame(body.get("bars"))
        strat=get_strategy(strategy_id)
        if not strat: return jsonify({"error":"Unknown strategy."}),400
        params=body.get("params") or {}
        if not isinstance(params,dict): return jsonify({"error":"Strategy params must be an object."}),400
        allowed=set((strat.get("param_schema") or {}).keys())
        params={k:v for k,v in params.items() if k in allowed}
        signaled=strat["generate_signals"](df,params)
        cols=["signal","custom_stop_price","custom_target_price","golden_box_low","golden_box_high","swing_a","swing_b","swing_c","fib_a_bar","fib_b_bar","fib_c_bar","fib_05","fib_0618","fib_0786","fib_1272","fib_1414","fib_1618","fib_2000","entry_box_low","entry_box_high","target_box_low","target_box_high","target_1272","target_1414","target_1618","target_2000","setup_direction"]
        rows=[]
        for i,ts in enumerate(signaled.index):
            item={"time":pd.Timestamp(ts).isoformat()}
            for col in cols:
                if col in signaled.columns:
                    v=signaled[col].iloc[i]
                    item[col]=int(v) if col=="signal" else _safe_float(v)
            rows.append(item)
        return jsonify({"strategy_id":strategy_id,"signals":rows,"count":len(rows),
                        "causal_note":"Pivots are exposed only after right-bar confirmation; each signal is computed using data available at that bar close."})
    except Exception as exc:
        return jsonify({"error":str(exc)}),400

@replay_bp.route("/api/backtest/replay/run",methods=["POST"])
def replay_run_backtest():
    err=_auth()
    if err:return err
    body=request.get_json(silent=True) or {}
    try:
        df=_dataset_frame(body.get("bars"))
        strategy_id=str(body.get("strategy_id","fibonacci_golden_box"))
        params=body.get("params") or {}
        from research.backtest_engine import run_backtest
        result=run_backtest(df,strategy_id,params=params,
                            initial_equity=float(body.get("initial_equity",10000)),
                            risk_pct=float(body.get("risk_pct",0.5)),
                            commission_pct=float(body.get("commission_pct",0.0005)),
                            slippage_points=float(body.get("slippage_points",0)))
        result["symbol"]=str(body.get("symbol","CUSTOM")).upper()
        result["interval"]=str(body.get("interval","5m"))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error":str(exc)}),400

@replay_bp.route("/api/backtest/replay/dukascopy", methods=["POST"])
def replay_dukascopy_data():
    """Fetch a bounded XAUUSD date slice directly from Dukascopy for browser replay."""
    err = _auth()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    interval_map = {
        "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
        "1h": "H1", "4h": "H4", "1d": "D1",
    }
    interval = str(body.get("interval", "5m"))
    if interval not in interval_map:
        return jsonify({"error": "Dukascopy supports 1m, 5m, 15m, 30m, 1h, 4h and 1d. 2m is not available."}), 400
    side = str(body.get("price_side", "BID")).upper()
    if side not in {"BID", "ASK"}:
        return jsonify({"error": "price_side must be BID or ASK."}), 400
    try:
        start = date.fromisoformat(str(body.get("start", "")))
        end = date.fromisoformat(str(body.get("end", "")))
        if end < start:
            return jsonify({"error": "End date must be on or after start date."}), 400
        if end > datetime.utcnow().date():
            return jsonify({"error": "End date cannot be in the future (UTC)."}), 400
        max_days = 8 if interval == "1m" else 31
        if (end - start).days + 1 > max_days:
            return jsonify({
                "error": f"Keep this request to {max_days} calendar days or fewer. "
                         "Fetch the historical range in smaller date chunks."
            }), 400

        from scripts.download_dukascopy_xauusd import download_range
        from scripts.xauusd_data_pipeline import RULES, resample_ohlc

        frames, missing = download_range(start, end, side, workers=1)
        if not frames:
            return jsonify({"error": "Dukascopy returned no candles for this date range."}), 404
        raw = pd.concat(frames, ignore_index=True).drop_duplicates("time", keep="last")
        raw = raw.sort_values("time").set_index("time")
        timeframe = interval_map[interval]
        if timeframe != "M1":
            raw = resample_ohlc(raw, RULES[timeframe])
        if raw.empty:
            return jsonify({"error": "No candles remain after timeframe aggregation."}), 404
        if len(raw) > 12000:
            return jsonify({"error": "The selected range exceeds the 12,000-bar Replay limit. Reduce the date range."}), 400
        replay_df = raw.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        bars = _bars_payload(replay_df)
        return jsonify({
            "symbol": "XAUUSD",
            "interval": interval,
            "source": f"Dukascopy {side} historical candles",
            "price_side": side,
            "bars": bars,
            "data_range": {"start": str(replay_df.index[0]), "end": str(replay_df.index[-1])},
            "count": len(bars),
            "missing_or_empty_days": missing,
            "storage_note": "Fetched on demand; saved only in this browser's Replay IndexedDB.",
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({
            "error": f"Dukascopy is temporarily unavailable: {exc}",
            "retryable": True,
            "suggestion": "Retry later or select a completed UTC date range ending yesterday."
        }), 503
    except Exception as exc:
        return jsonify({"error": f"Dukascopy download failed: {exc}"}), 502



@replay_bp.route("/api/backtest/replay/sdb", methods=["POST"])
def replay_run_sdb():
    """Run SDB on the exact XAUUSD M5 bars supplied by the Replay browser."""
    err = _auth()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    try:
        symbol = str(body.get("symbol", "")).strip().upper()
        interval = str(body.get("interval", ""))
        if symbol != "XAUUSD":
            return jsonify({"error": "SDB is restricted to XAUUSD. Set the Replay symbol to XAUUSD; GLD proxy data is not accepted."}), 400
        if interval != "5m":
            return jsonify({"error": "ICT SDB requires 5-minute entry bars. Select 5m and load XAUUSD data first."}), 400
        df = _dataset_frame(body.get("bars"))
        if len(df) < 500:
            return jsonify({"error": f"SDB requires at least 500 valid 5-minute XAUUSD bars; received {len(df)}."}), 400
        params = body.get("params") or {}
        if not isinstance(params, dict):
            return jsonify({"error": "SDB parameters must be an object."}), 400
        allowed = {"sweep_lookback", "mss_lookback", "mss_window", "h1_ema_len", "atr_len", "stop_buffer_atr", "h1_bias_mode"}
        params = {k: v for k, v in params.items() if k in allowed}
        initial_equity = float(body.get("initial_equity", 10000))
        risk_pct = float(body.get("risk_pct", 0.5))
        spread = float(body.get("spread_points", 0.30))
        slippage = float(body.get("slippage_points", 0.05))
        commission = float(body.get("commission_pct", 0.0005))
        if initial_equity <= 0 or not 0 < risk_pct <= 2:
            return jsonify({"error": "Initial equity must be positive and risk must be greater than 0 and at most 2%."}), 400
        if not 0 <= spread <= 20 or not 0 <= slippage <= 10 or not 0 <= commission <= 0.02:
            return jsonify({"error": "Spread, slippage or commission is outside the supported range."}), 400
        from research.xauusd_sdb import run_backtest
        result = run_backtest(df, initial_equity=initial_equity, risk_pct=risk_pct,
                              spread_points=spread, slippage_points=slippage,
                              commission_pct=commission, **params)
        result["symbol"] = "XAUUSD"
        result["interval"] = "5m"
        result["data_source"] = str(body.get("source", "Replay browser dataset"))
        result["disclaimer"] = (
            "Uses the exact 5m bars submitted from this Replay browser dataset. "
            "Deviation levels remain a documented range-projection approximation, not a verified "
            "replica of a proprietary ICT/TradingView indicator. Calibrate spread, slippage, "
            "commission and contract sizing to your broker."
        )
        def clean(value):
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                value = float(value)
            if isinstance(value, float):
                return value if np.isfinite(value) else None
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            return value
        return jsonify(clean(result))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"SDB Replay backtest failed: {exc}", "type": type(exc).__name__}), 500
