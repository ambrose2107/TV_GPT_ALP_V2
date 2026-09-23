"""Routes for the Gold Quad-RSI mean-reversion research strategy."""

import math
from flask import Blueprint, jsonify, request, render_template, session

from research.xauusd_confluence_v4 import load_data
from research.xauusd_quad_rsi_v1 import QuadRSIConfig, backtest


xauusd_quad_rsi_v1_bp = Blueprint("xauusd_quad_rsi_v1", __name__)


def _safe(v):
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
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


@xauusd_quad_rsi_v1_bp.route("/xauusd-quad-rsi-v1")
def xauusd_quad_rsi_v1_page():
    if not session.get("logged_in"):
        from flask import redirect, url_for
        return redirect(url_for("dashboard.login"))
    return render_template("xauusd_quad_rsi_v1.html")


@xauusd_quad_rsi_v1_bp.route("/api/xauusd-quad-rsi-v1/run", methods=["POST"])
def xauusd_quad_rsi_v1_run():
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", "GLD")
    n_bars = max(1000, min(60000, int(body.get("n_bars", 3120))))
    raw = body.get("config") or {}
    allowed = {k: raw[k] for k in raw if k in QuadRSIConfig.__dataclass_fields__}
    cfg = QuadRSIConfig(**allowed)

    data = load_data(
        use_live=True,
        n_bars=n_bars,
        symbol=symbol,
        data_source="alpaca",
    )
    result = backtest(data["m5"], cfg)
    d = result["data"]

    return jsonify(_safe({
        "strategy": "Gold Quad-RSI Mean Reversion V1",
        "symbol": symbol,
        "data_source": "Alpaca/GLD proxy",
        "bars": int(len(data["m5"])),
        "data_start": data["m5"].index[0].isoformat() if len(data["m5"]) else None,
        "data_end": data["m5"].index[-1].isoformat() if len(data["m5"]) else None,
        "metrics": result["metrics"],
        "diagnostics": result["diagnostics"],
        "trades": result["trades"],
        "signals": result["signals"].tail(200).to_dict("records"),
        "daily_bars": int(len(d)),
        "rules": {
            "rsi_len": cfg.rsi_len,
            "rsi_oversold": cfg.rsi_oversold,
            "decline_window": cfg.decline_window,
            "min_rsi_declines": cfg.min_rsi_declines,
            "ibs_max": cfg.ibs_max,
            "ma_len": cfg.ma_len,
            "entry": "next session open",
            "exit": "next session open after close > previous session high",
            "position_rule": "one position at a time",
        },
    }))


__all__ = ["xauusd_quad_rsi_v1_bp"]
