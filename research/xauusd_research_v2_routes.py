"""Routes for the three focused V2 research candidates."""
from flask import Blueprint, jsonify, request, session
from research.xauusd_confluence_v4 import load_data
from research.xauusd_pullback_v2 import PullbackV2Config, backtest as pullback_backtest
from research.xauusd_ema_retest_v2 import EMARetestV2Config, backtest as ema_backtest
from research.xauusd_triple_rsi_v2 import TripleRSIV2Config, backtest as triple_backtest

bp=Blueprint("xauusd_research_v2",__name__)

@bp.route("/api/xauusd-research-v2/<strategy>",methods=["POST"])
def run(strategy):
    if not session.get("logged_in"): return jsonify({"error":"Unauthorized"}),401
    body=request.get_json(silent=True) or {}
    symbol=str(body.get("symbol","GLD")).upper()
    bars=max(500,min(60000,int(body.get("n_bars",15600))))
    cfgbody=body.get("config") or {}
    try:
        data=load_data(use_live=True,n_bars=bars,symbol=symbol,data_source="alpaca")
        if strategy=="pullback":
            cfg=PullbackV2Config(**{k:v for k,v in cfgbody.items() if k in PullbackV2Config.__annotations__})
            r=pullback_backtest(data["m5"],cfg)
        elif strategy=="ema":
            cfg=EMARetestV2Config(**{k:v for k,v in cfgbody.items() if k in EMARetestV2Config.__annotations__})
            r=ema_backtest(data["m5"],cfg)
        elif strategy=="triple":
            cfg=TripleRSIV2Config(**{k:v for k,v in cfgbody.items() if k in TripleRSIV2Config.__annotations__})
            r=triple_backtest(data["m5"],cfg)
        else: return jsonify({"error":"Unknown V2 strategy"}),400
        trades=r.get("trades",[])
        if hasattr(trades,"to_dict"): trades=trades.to_dict(orient="records")
        sig=r.get("signals")
        if hasattr(sig,"tail"): sig=sig.tail(500).reset_index().to_dict(orient="records")
        return jsonify({"strategy":strategy+" V2","symbol":symbol,"data_source":"Alpaca","bars":len(data["m5"]),"metrics":r["metrics"],"trades":trades,"signals":sig})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@bp.route("/xauusd-research-v2")
def page():
    if not session.get("logged_in"):
        from flask import redirect,url_for
        return redirect(url_for("dashboard.login"))
    return jsonify({"strategies":["Pullback V2","EMA Retest V2","Triple RSI V2"],"note":"Research candidates; validate out-of-sample."})
