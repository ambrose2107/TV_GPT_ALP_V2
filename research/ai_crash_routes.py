"""AI Crash Early-Warning API."""
from flask import Blueprint, jsonify, request, session
from core.logger import get_logger
from research.ai_crash_warning import build_dashboard
logger=get_logger(__name__)
ai_crash_bp=Blueprint("ai_crash",__name__)
@ai_crash_bp.route("/api/ai-crash/dashboard")
def ai_crash_dashboard():
    if not session.get("logged_in"): return jsonify({"error":"Unauthorized"}),401
    try:return jsonify(build_dashboard(force=request.args.get("refresh")=="1"))
    except Exception as ex:
        logger.warning("AI crash dashboard failed: %s",ex)
        return jsonify({"error":str(ex)}),503
