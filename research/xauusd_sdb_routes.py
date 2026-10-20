"""Authenticated routes for the isolated XAUUSD SDB research page."""
from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for
import logging
import math
import numpy as np

from research.xauusd_sdb import load_gold_data, run_backtest

xauusd_sdb_bp = Blueprint("xauusd_sdb", __name__)
log = logging.getLogger("xauusd_sdb")


def _authorized():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@xauusd_sdb_bp.route("/xauusd-sdb")
def page():
    if not session.get("logged_in"):
        return redirect(url_for("dashboard.login"))
    return render_template("xauusd_sdb.html")


@xauusd_sdb_bp.route("/api/xauusd-sdb/run", methods=["POST"])
def run():
    denied = _authorized()
    if denied:
        return denied
    body = request.get_json(silent=True) or {}
    try:
        risk_pct = float(body.get("risk_pct", 0.5))
        spread = float(body.get("spread_points", 0.30))
        slippage = float(body.get("slippage_points", 0.05))
        if not 0 < risk_pct <= 2:
            return jsonify({"error": "Risk must be greater than 0 and at most 2% per trade."}), 400
        if not 0 <= spread <= 20 or not 0 <= slippage <= 10:
            return jsonify({"error": "Spread/slippage values are outside safe bounds."}), 400
        data = load_gold_data("60d")
        result = run_backtest(data, risk_pct=risk_pct, spread_points=spread,
                              slippage_points=slippage)
        result["symbol"] = "GC=F"
        result["disclaimer"] = (
            "Research approximation only. Yahoo GC=F is a gold futures proxy, not spot XAUUSD. "
            "Deviation targets use the documented range-projection convention; they are not "
            "statistical standard deviations or a verified TradingView tool replica."
        )
        return jsonify(_safe(result))
    except Exception as exc:
        log.exception("SDB run failed")
        return jsonify({"error": str(exc), "type": type(exc).__name__}), 400
