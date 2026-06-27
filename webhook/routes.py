"""
webhook/routes.py — Flask blueprint for /webhook endpoint
"""
from flask import Blueprint, request, jsonify
from webhook.handler import process_webhook
from core.logger import get_logger

logger = get_logger(__name__)
webhook_bp = Blueprint("webhook", __name__)

@webhook_bp.route("/webhook", methods=["POST"])
def receive_webhook():
    """
    TradingView posts JSON here.
    Paste this URL in TradingView alert: https://YOUR-RAILWAY-URL/webhook
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            logger.warning("Empty or invalid JSON received")
            return jsonify({"success": False, "message": "Invalid JSON"}), 400

        logger.info(f"Webhook received: {payload}")
        result = process_webhook(payload)

        status_code = 200 if result["success"] else 400
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Webhook route error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@webhook_bp.route("/health", methods=["GET"])
def health():
    """Health check for Railway / uptime monitors."""
    return jsonify({"status": "ok", "message": "Trading bot is running"}), 200
