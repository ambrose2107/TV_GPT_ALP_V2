"""Authenticated API for the focused V2 research candidates."""
import math
from flask import Blueprint, jsonify, request, session
from core.market_data import alpaca_get_bars, get_bars
from research.xauusd_v3.data_loader import get_data as v3_get_data
from research.xauusd_confluence_v4 import load_data
from research.xauusd_pullback_v2 import PullbackV2Config, backtest as pullback_backtest
from research.xauusd_ema_retest_v2 import EMARetestV2Config, backtest as ema_backtest
from research.xauusd_triple_rsi_v2 import TripleRSIV2Config, backtest as triple_backtest

bp = Blueprint("xauusd_research_v2", __name__)

def _safe(v):
    if isinstance(v, dict): return {str(k): _safe(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [_safe(x) for x in v]
    if hasattr(v, "item"):
        try: return _safe(v.item())
        except Exception: pass
    if isinstance(v, float) and not math.isfinite(v): return None
    return v

def _cfg(cls, raw):
    raw = raw or {}
    return cls(**{k:v for k,v in raw.items() if k in cls.__dataclass_fields__})

def _daily(symbol, n):
    raw = alpaca_get_bars(symbol, "1Day", limit=max(250,min(5000,n)))
    source = "Alpaca"
    if not raw:
        raw = get_bars(symbol, "1y")
        source = "Analyzer Pro/Yahoo fallback"
    if not raw:
        raise ValueError(f"No daily data available for {symbol}")
    import pandas as pd
    df = pd.DataFrame(raw)
    if {"t","o","h","l","c"}.issubset(df.columns):
        df["t"]=pd.to_datetime(df["t"],utc=True); df=df.set_index("t").sort_index().rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    return df,source

@bp.route("/api/xauusd-research-v2/<strategy>", methods=["POST"])
def run(strategy):
    if not session.get("logged_in"): return jsonify({"error":"Unauthorized"}),401
    body=request.get_json(silent=True) or {}
    symbol=str(body.get("symbol","GLD")).strip().upper()
    bars=max(250,min(60000,int(body.get("n_bars",15600))))
    try:
        if strategy=="triple":
            df,source=_daily(symbol,int(body.get("daily_bars",max(250,min(5000,bars//78 or 250)))))
            cfg=_cfg(TripleRSIV2Config,body.get("config"))
            r=triple_backtest(df,cfg)
            data_bars=len(df)
        else:
            data=load_data(use_live=True,n_bars=bars,symbol=symbol,data_source="alpaca")
            if not data or data.get("m5") is None or len(data["m5"]) < 100:
                raise ValueError(f"Alpaca returned insufficient 5m data for {symbol}")
            source="Alpaca"
            if strategy=="pullback":
                cfg=_cfg(PullbackV2Config,body.get("config")); r=pullback_backtest(data["m5"],cfg)
            elif strategy=="ema":
                cfg=_cfg(EMARetestV2Config,body.get("config")); r=ema_backtest(data["m5"],cfg)
            else:
                return jsonify({"error":"Unknown V2 strategy"}),400
            data_bars=len(data["m5"])
        trades=r.get("trades",[])
        if hasattr(trades,"to_dict"): trades=trades.to_dict(orient="records")
        sig=r.get("signals")
        if hasattr(sig,"tail"): sig=sig.tail(500).reset_index().to_dict(orient="records")
        return jsonify(_safe({"strategy":strategy+" V2","symbol":symbol,"data_source":source,"bars":data_bars,
            "metrics":r["metrics"],"trades":trades,"signals":sig,"diagnostics":r.get("diagnostics",{})}))
    except Exception as e:
        return jsonify({"error":f"{type(e).__name__}: {e}"}),500
