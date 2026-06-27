"""
app.py — Flask application factory  v8
Registers: webhook, dashboard, mirrorfish, analytics blueprints
"""
from flask import Flask
from core.database import init_db
from core.logger import get_logger
from webhook.routes import webhook_bp
from dashboard.routes import dashboard_bp
from mirrorfish.routes import mirrorfish_bp
from core.analytics_routes import analytics_bp
import os

logger = get_logger(__name__)

def create_app():
    app = Flask(__name__,
                template_folder="dashboard/templates",
                static_folder="dashboard/static")
    app.secret_key = os.environ.get("APP_SECRET_KEY", "change-me-in-production")
    # NOTE: Do NOT set SESSION_TYPE="filesystem" without flask-session installed.
    # Flask's default cookie-based sessions work fine and don't need flask-session.
    init_db()
    app.register_blueprint(webhook_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(mirrorfish_bp)
    app.register_blueprint(analytics_bp)
    logger.info("OptiTrade AI v8 app created.")
    return app
