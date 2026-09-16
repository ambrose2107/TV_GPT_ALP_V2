"""
research/xauusd_v3_routes.py
Standalone routes for the v3 institutional-confluences backtester.
Deliberately kept separate from dashboard/routes.py and the existing
Backtest tab - this is its own page at /xauusd-v3, with its own blueprint,
so it cannot collide with anything in the existing (large) dashboard.html.

Routes:
  GET  /xauusd-v3                 -> renders the standalone page
  POST /api/xauusd-v3/run         -> runs one backtest, returns metrics +
                                      funnel + trades + base64 chart images
"""
import base64
import os
import tempfile
import traceback

from flask import Blueprint, render_template, request, jsonify, session

from core.logger import get_logger
from research.xauusd_v3 import run_backtest, SignalConfig, EngineConfig

logger = get_logger(__name__)
xauusd_v3_bp = Blueprint("xauusd_v3", __name__)


def _auth():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None


@xauusd_v3_bp.route("/xauusd-v3")
def xauusd_v3_page():
    if not session.get("logged_in"):
        from flask import redirect, url_for
        return redirect(url_for("dashboard.login"))
    return render_template("xauusd_v3.html")


def _fig_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


@xauusd_v3_bp.route("/api/xauusd-v3/run", methods=["POST"])
def api_xauusd_v3_run():
    e = _auth()
    if e:
        return e

    body = request.json or {}
    interval = body.get("interval", "5m")
    n_bars = int(body.get("n_bars", 20000))
    use_live = bool(body.get("use_live", True))

    # only accept known SignalConfig / EngineConfig fields - never pass
    # arbitrary request JSON straight into a dataclass constructor
    sig_fields = SignalConfig.__dataclass_fields__.keys()
    eng_fields = EngineConfig.__dataclass_fields__.keys()
    sig_kwargs = {k: v for k, v in (body.get("signal") or {}).items() if k in sig_fields}
    eng_kwargs = {k: v for k, v in (body.get("engine") or {}).items() if k in eng_fields}

    try:
        signal_cfg = SignalConfig(**sig_kwargs)
        engine_cfg = EngineConfig(**eng_kwargs)
    except TypeError as ex:
        return jsonify({"error": f"Invalid parameter: {ex}"}), 400

    try:
        result = run_backtest(
            symbol_interval=interval, use_live=use_live, n_bars=n_bars,
            signal_cfg=signal_cfg, engine_cfg=engine_cfg,
        )
    except Exception as ex:
        logger.warning(f"xauusd_v3 backtest failed: {ex}\n{traceback.format_exc()}")
        return jsonify({"error": str(ex)}), 400

    # generate charts to a temp dir, then inline them as base64 so the
    # frontend needs zero extra chart-drawing code
    from research.xauusd_v3.visualize import (
        plot_overview, plot_trade_detail, plot_fibonacci_levels, plot_zones, plot_funnel,
    )

    charts = {}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = os.path.join(tmp, "overview.png")
            plot_overview(result["df"], result["trades"], result["equity_curve"], p,
                          title=f"XAUUSD ({interval})")
            charts["overview"] = _fig_to_base64(p)
        except Exception as ex:
            logger.warning(f"overview chart failed: {ex}")

        if not result["trades"].empty:
            try:
                p = os.path.join(tmp, "trade_detail.png")
                plot_trade_detail(result["df"], result["trades"], p)
                charts["trade_detail"] = _fig_to_base64(p)
            except Exception as ex:
                logger.warning(f"trade_detail chart failed: {ex}")

        try:
            p = os.path.join(tmp, "fib.png")
            plot_fibonacci_levels(result["df"], signal_cfg.fib_swing_lookback, p)
            charts["fibonacci_levels"] = _fig_to_base64(p)
        except Exception as ex:
            logger.warning(f"fib chart failed: {ex}")

        try:
            p = os.path.join(tmp, "zones.png")
            plot_zones(result["df"], result["confluence_data"], p)
            charts["institutional_zones"] = _fig_to_base64(p)
        except Exception as ex:
            logger.warning(f"zones chart failed: {ex}")

        try:
            p = os.path.join(tmp, "funnel.png")
            plot_funnel(result["funnel"], p)
            charts["funnel"] = _fig_to_base64(p)
        except Exception as ex:
            logger.warning(f"funnel chart failed: {ex}")

    trades_out = result["trades"].tail(200).copy()
    for col in ("entry_time", "exit_time", "breakeven_time"):
        if col in trades_out.columns:
            trades_out[col] = trades_out[col].astype(str)

    # funnel order matters for the frontend (it shows successive filter
    # drop-off) - Flask's default JSON provider may reorder dict keys, so
    # ship it as an ordered list of pairs instead of a dict
    funnel_ordered = [{"stage": k, "count": v} for k, v in result["funnel"].items()]

    return jsonify({
        "metrics": result["metrics"],
        "funnel": funnel_ordered,
        "trades": trades_out.to_dict("records"),
        "used_numba": result["used_numba"],
        "charts": charts,
    })
