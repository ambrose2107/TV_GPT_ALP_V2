from flask import Blueprint, render_template, request, jsonify, session
import math
import numpy as np

from research.xauusd_confluence_v4 import load_data
from research.xauusd_ema_retest_v1 import EMARetestConfig, backtest

xauusd_ema_retest_v1_bp = Blueprint("xauusd_ema_retest_v1", __name__)


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


@xauusd_ema_retest_v1_bp.route("/xauusd-ema-retest-v1")
def page():
    if not session.get("logged_in"):
        from flask import redirect, url_for
        return redirect(url_for("dashboard.login"))
    return render_template("xauusd_ema_retest_v1.html")


@xauusd_ema_retest_v1_bp.route("/api/xauusd-ema-retest-v1/run", methods=["POST"])
def run():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    try:
        cfg = EMARetestConfig(**{
            k: v for k, v in (body.get("config") or {}).items()
            if k in EMARetestConfig.__dataclass_fields__
        })
        bars = int(body.get("n_bars", 7800))
        symbol = str(body.get("symbol", "GLD")).upper()
        data = load_data(
            use_live=True,
            n_bars=bars,
            symbol=symbol,
            data_source="alpaca",
        )
        result = backtest(data["m5"], cfg)
        trades = result["trades"].tail(300).copy()
        return jsonify(_safe({
            "symbol": symbol,
            "data_source": "Alpaca/GLD proxy" if symbol == "GLD" else "Alpaca",
            "bars": len(data["m5"]),
            "data_start": str(data["m5"].index.min()),
            "data_end": str(data["m5"].index.max()),
            "metrics": result["metrics"],
            "diagnostics": result["diagnostics"],
            "trades": trades.to_dict("records"),
            "config": cfg.__dict__,
        }))
    except Exception as ex:
        return jsonify({"error": str(ex), "type": type(ex).__name__}), 400
