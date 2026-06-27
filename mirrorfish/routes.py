"""
mirrorfish/routes.py — /api/mirrorfish/* endpoints  v8
Requires dashboard login session.
"""
from flask import Blueprint, request, jsonify, session
from mirrorfish.engine import analyze_symbol, analyze_portfolio, chat, get_provider_status
from core.database import get_closed_positions
from core.logger import get_logger

logger = get_logger(__name__)
mirrorfish_bp = Blueprint("mirrorfish", __name__)

def _auth():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None

@mirrorfish_bp.route("/api/mirrorfish/status", methods=["GET"])
def mf_status():
    e = _auth()
    if e: return e
    return jsonify(get_provider_status())

@mirrorfish_bp.route("/api/mirrorfish/analyze", methods=["POST"])
def mf_analyze():
    e = _auth()
    if e: return e
    data   = request.get_json() or {}
    symbol = data.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    return jsonify(analyze_symbol(symbol, data.get("market_data", {}), data.get("signals", {})))

@mirrorfish_bp.route("/api/mirrorfish/portfolio", methods=["POST"])
def mf_portfolio():
    e = _auth()
    if e: return e
    data      = request.get_json() or {}
    positions = data.get("positions", [])
    closed    = get_closed_positions(limit=30)
    return jsonify(analyze_portfolio(positions, closed))

@mirrorfish_bp.route("/api/mirrorfish/chat", methods=["POST"])
def mf_chat():
    e = _auth()
    if e: return e
    data    = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "message required"}), 400
    return jsonify({"response": chat(message, data.get("context", {}))})
