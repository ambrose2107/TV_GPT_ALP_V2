"""TradingView-style historical data and bar-replay workspace.

Replay datasets are retained in the user's browser IndexedDB and can be exported
as CSV. Server endpoints are stateless; a replay request only analyzes the
dataset supplied by the browser.
"""
import io
import csv
from datetime import datetime
import pandas as pd
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
        cols=["signal","custom_stop_price","custom_target_price","golden_box_low","golden_box_high","swing_a","swing_b","swing_c"]
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
