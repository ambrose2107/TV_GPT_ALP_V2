"""AI Crash Early-Warning API."""
from flask import Blueprint, jsonify, request, session
from core.logger import get_logger
logger=get_logger(__name__)
ai_crash_bp=Blueprint("ai_crash",__name__)
@ai_crash_bp.route("/api/ai-crash/dashboard")
def ai_crash_dashboard():
    if not session.get("logged_in"): return jsonify({"error":"Unauthorized"}),401
    try:
        from research.ai_crash_warning import build_dashboard
        return jsonify(build_dashboard(force=request.args.get("refresh")=="1"))
    except Exception as ex:
        logger.warning("AI crash dashboard failed: %s",ex)
        return jsonify({"error":str(ex)}),503


@ai_crash_bp.route("/api/ai-crash/production-replay/start", methods=["POST"])
def ai_crash_production_replay_start():
    """Explicitly start the expensive production-score historical replay."""
    if not session.get("logged_in"): return jsonify({"error":"Unauthorized"}),401
    try:
        from research.ai_crash_warning import start_production_replay
        from research.ai_crash_warning import _fred, FRED
        f={k:_fred(v) for k,v in FRED.items()}
        started,state=start_production_replay(f)
        return jsonify({"started":started,"state":state})
    except Exception as ex:
        logger.warning("AI crash production replay start failed: %s",ex)
        return jsonify({"error":str(ex)}),503


@ai_crash_bp.route("/api/ai-crash/production-replay/status")
def ai_crash_production_replay_status():
    if not session.get("logged_in"): return jsonify({"error":"Unauthorized"}),401
    try:
        from research.ai_crash_warning import production_replay_status
        return jsonify(production_replay_status())
    except Exception as ex:
        logger.warning("AI crash production replay status failed: %s",ex)
        return jsonify({"error":str(ex)}),503
