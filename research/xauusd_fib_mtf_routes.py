"""
research/xauusd_fib_mtf_routes.py
Standalone routes for the 5m/15m Fibonacci+liquidity+structure backtester.
Same pattern as research/xauusd_v3_routes.py: its own page at
/xauusd-fib-mtf, its own blueprint, zero edits to the existing
dashboard.html required.

Routes:
  GET  /xauusd-fib-mtf                 -> renders the standalone page
  POST /api/xauusd-fib-mtf/run         -> runs one backtest, returns
                                           metrics + funnel + trades +
                                           base64 chart images
  POST /api/xauusd-fib-mtf/optimize    -> runs the grid-search optimizer,
                                           returns the ranked results table
"""
import base64
import os
import tempfile
import traceback

from flask import Blueprint, render_template, request, jsonify, session

from core.logger import get_logger
from research.xauusd_fib_mtf import run_backtest, MTFConfig, EngineConfig, grid_search, PARAM_GRID
from research.xauusd_fib_mtf.data_loader import get_mtf_data
# Imported at module load time (not inside the request handler) so any
# failure here surfaces in the deploy/startup logs, not as a confusing
# per-request 500 the first time someone clicks "Run".
from research.xauusd_fib_mtf.visualize import plot_overview, plot_fib_and_trade_detail, plot_funnel

logger = get_logger(__name__)
xauusd_fib_mtf_bp = Blueprint("xauusd_fib_mtf", __name__)


def _auth():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None


@xauusd_fib_mtf_bp.route("/xauusd-fib-mtf")
def xauusd_fib_mtf_page():
    if not session.get("logged_in"):
        from flask import redirect, url_for
        return redirect(url_for("dashboard.login"))
    return render_template("xauusd_fib_mtf.html")


def _fig_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _safe_mtf_kwargs(body: dict) -> dict:
    fields = MTFConfig.__dataclass_fields__.keys()
    return {k: v for k, v in (body or {}).items() if k in fields}


def _safe_engine_kwargs(body: dict) -> dict:
    fields = EngineConfig.__dataclass_fields__.keys()
    return {k: v for k, v in (body or {}).items() if k in fields}


@xauusd_fib_mtf_bp.route("/api/xauusd-fib-mtf/run", methods=["POST"])
def api_run():
    e = _auth()
    if e:
        return e
    try:
        return _run_impl()
    except Exception as ex:
        logger.error(f"xauusd_fib_mtf UNEXPECTED error: {ex}\n{traceback.format_exc()}")
        return jsonify({"error": f"Unexpected server error: {ex}. Check server logs for the full traceback."}), 500


def _run_impl():
    body = request.get_json(silent=True) or {}
    n_bars = int(body.get("n_bars", 30000))
    use_live = bool(body.get("use_live", True))

    try:
        mtf_cfg = MTFConfig(**_safe_mtf_kwargs(body.get("mtf") or {}))
        engine_cfg = EngineConfig(**_safe_engine_kwargs(body.get("engine") or {}))
    except TypeError as ex:
        return jsonify({"error": f"Invalid parameter: {ex}"}), 400

    try:
        result = run_backtest(use_live=use_live, n_bars=n_bars, mtf_cfg=mtf_cfg, engine_cfg=engine_cfg)
    except Exception as ex:
        logger.warning(f"xauusd_fib_mtf backtest failed: {ex}\n{traceback.format_exc()}")
        return jsonify({"error": str(ex)}), 400

    charts = {}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = os.path.join(tmp, "overview.png")
            plot_overview(result["df5"], result["trades"], result["equity_curve"], p)
            charts["overview"] = _fig_to_base64(p)
        except Exception as ex:
            logger.warning(f"overview chart failed: {ex}")

        if not result["trades"].empty:
            try:
                p = os.path.join(tmp, "trade_detail.png")
                plot_fib_and_trade_detail(result["df5"], result["df15"], result["trades"], p)
                charts["trade_detail"] = _fig_to_base64(p)
            except Exception as ex:
                logger.warning(f"trade_detail chart failed: {ex}")

        try:
            p = os.path.join(tmp, "funnel.png")
            plot_funnel(result["funnel"], p)
            charts["funnel"] = _fig_to_base64(p)
        except Exception as ex:
            logger.warning(f"funnel chart failed: {ex}")

    trades_out = result["trades"].tail(200).copy()
    for col in ("entry_time", "exit_time"):
        if col in trades_out.columns:
            trades_out[col] = trades_out[col].astype(str)

    funnel_ordered = [{"stage": k, "count": v} for k, v in result["funnel"].items()]

    return jsonify({
        "metrics": result["metrics"],
        "funnel": funnel_ordered,
        "trades": trades_out.to_dict("records"),
        "used_numba": result["used_numba"],
        "charts": charts,
    })


@xauusd_fib_mtf_bp.route("/api/xauusd-fib-mtf/optimize", methods=["POST"])
def api_optimize():
    e = _auth()
    if e:
        return e
    try:
        return _optimize_impl()
    except Exception as ex:
        logger.error(f"xauusd_fib_mtf optimizer UNEXPECTED error: {ex}\n{traceback.format_exc()}")
        return jsonify({"error": f"Unexpected server error: {ex}. Check server logs."}), 500


def _optimize_impl():
    body = request.get_json(silent=True) or {}
    n_bars = int(body.get("n_bars", 60000))
    use_live = bool(body.get("use_live", True))
    min_trades = int(body.get("min_trades", 20))

    try:
        base_cfg = MTFConfig(**_safe_mtf_kwargs(body.get("mtf") or {}))
        engine_cfg = EngineConfig(**_safe_engine_kwargs(body.get("engine") or {}))
    except TypeError as ex:
        return jsonify({"error": f"Invalid parameter: {ex}"}), 400

    # bound the grid so a single request can't run unboundedly long on a
    # shared web dyno - this endpoint is meant for exploratory tuning, not
    # a full research-grade sweep (use main.py locally for that)
    grid = {k: v[:2] for k, v in PARAM_GRID.items()}  # cap each dimension at 2 values
    import itertools
    combos = [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]
    if len(combos) > 32:
        combos = combos[:32]

    try:
        data = get_mtf_data(use_live=use_live, n_bars=n_bars)
        results_df = grid_search(data, combos, base_cfg, engine_cfg, min_trades=min_trades)
    except Exception as ex:
        logger.warning(f"xauusd_fib_mtf optimize failed: {ex}\n{traceback.format_exc()}")
        return jsonify({"error": str(ex)}), 400

    if results_df.empty:
        return jsonify({"results": [], "note": "No combinations produced a valid backtest."})

    return jsonify({
        "results": results_df.head(20).to_dict("records"),
        "n_combos_tried": len(combos),
        "min_trades_threshold": min_trades,
    })
