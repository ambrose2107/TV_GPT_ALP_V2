"""
research/backtest_routes.py
Flask blueprint powering the dashboard's Backtest tab.

Routes:
  GET  /api/backtest/strategies   -> list registered strategies + param schema
  POST /api/backtest/run          -> run ONE strategy, return metrics+equity+trades
  POST /api/backtest/compare      -> run MULTIPLE strategies on the same
                                      symbol/interval/period, return a
                                      side-by-side comparison

No arbitrary code execution: strategies are pre-registered Python modules
in research/strategies/, not user-submitted code strings.
"""
from flask import Blueprint, request, jsonify, session
import math

from core.logger import get_logger
from research.strategies import list_strategies
from research.backtest_engine import fetch_yf_data, run_backtest

logger = get_logger(__name__)
backtest_bp = Blueprint("backtest", __name__)


def _sanitize_for_json(obj):
    """
    Python's json module serializes float('nan')/inf as bare NaN/Infinity
    tokens, which are NOT valid JSON - browsers' JSON.parse() rejects them
    with "Unexpected token 'N'...". Recursively swap NaN/Infinity for None
    (-> JSON null) before jsonify.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _auth():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None


@backtest_bp.route("/api/backtest/strategies")
def api_backtest_strategies():
    e = _auth()
    if e:
        return e
    return jsonify({"strategies": list_strategies()})


@backtest_bp.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    e = _auth()
    if e:
        return e
    data = request.json or {}
    symbol = (data.get("symbol") or "").strip().upper()
    interval = data.get("interval", "5m")
    period = data.get("period") or None
    strategy_id = data.get("strategy_id")
    params = data.get("params") or {}
    initial_equity = float(data.get("initial_equity", 10000))
    risk_pct = float(data.get("risk_pct", 0.5))

    if not symbol or not strategy_id:
        return jsonify({"error": "symbol and strategy_id are required"}), 400

    try:
        df = fetch_yf_data(symbol, interval=interval, period=period)
        result = run_backtest(
            df, strategy_id, params=params,
            initial_equity=initial_equity, risk_pct=risk_pct,
        )
        result["symbol"] = symbol
        result["interval"] = interval
        return jsonify(_sanitize_for_json(result))
    except Exception as ex:
        logger.warning(f"Backtest run failed for {symbol}/{strategy_id}: {ex}")
        return jsonify({"error": str(ex)}), 400


@backtest_bp.route("/api/backtest/compare", methods=["POST"])
def api_backtest_compare():
    e = _auth()
    if e:
        return e
    data = request.json or {}
    symbol = (data.get("symbol") or "").strip().upper()
    interval = data.get("interval", "5m")
    period = data.get("period") or None
    strategy_ids = data.get("strategy_ids") or []
    params_by_strategy = data.get("params_by_strategy") or {}
    initial_equity = float(data.get("initial_equity", 10000))
    risk_pct = float(data.get("risk_pct", 0.5))

    if not symbol or not strategy_ids:
        return jsonify({"error": "symbol and strategy_ids are required"}), 400
    if len(strategy_ids) > 8:
        return jsonify({"error": "Max 8 strategies per comparison run"}), 400

    try:
        df = fetch_yf_data(symbol, interval=interval, period=period)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400

    results = []
    for sid in strategy_ids:
        try:
            result = run_backtest(
                df, sid, params=params_by_strategy.get(sid, {}),
                initial_equity=initial_equity, risk_pct=risk_pct,
            )
            results.append(result)
        except Exception as ex:
            results.append({"strategy_id": sid, "error": str(ex)})

    # Rank valid results by total_return_pct (simple, transparent default;
    # the UI itself shows profit factor / drawdown / win rate too so the
    # user can re-sort by whatever they care about most)
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["metrics"].get("total_return_pct", -1e9), reverse=True)
    ranked_ids = [r["strategy_id"] for r in valid]

    return jsonify(_sanitize_for_json({
        "symbol": symbol, "interval": interval,
        "results": results,
        "ranking_by_return": ranked_ids,
    }))
