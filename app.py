"""Flask application factory v8.

Registers the core dashboard/webhook routes plus the XAUUSD research
strategy pages, including the combined strategy lab.
"""

import os

from flask import Flask

from core.analytics_routes import analytics_bp
from core.database import init_db
from core.logger import get_logger
from core.scheduler import init_scheduler
from dashboard.routes import dashboard_bp
from mirrorfish.routes import mirrorfish_bp
from research.backtest_routes import backtest_bp
from research.xauusd_confluence_v4_routes import xauusd_confluence_v4_bp
from research.xauusd_fib_mtf_routes import xauusd_fib_mtf_bp
from research.xauusd_pullback_v1_routes import xauusd_pullback_v1_bp
from research.xauusd_strategy_lab_routes import xauusd_strategy_lab_bp
from research.xauusd_ema_retest_v1_routes import xauusd_ema_retest_v1_bp
from research.xauusd_v3_routes import xauusd_v3_bp
from webhook.routes import webhook_bp


logger = get_logger(__name__)


def create_app():
    app = Flask(
        __name__,
        template_folder="dashboard/templates",
        static_folder="dashboard/static",
    )
    app.secret_key = os.environ.get(
        "APP_SECRET_KEY",
        "change-me-in-production",
    )

    init_db()

    app.register_blueprint(webhook_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(mirrorfish_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(backtest_bp)
    app.register_blueprint(xauusd_v3_bp)
    app.register_blueprint(xauusd_fib_mtf_bp)
    app.register_blueprint(xauusd_confluence_v4_bp)
    app.register_blueprint(xauusd_pullback_v1_bp)
    app.register_blueprint(xauusd_strategy_lab_bp)
    app.register_blueprint(xauusd_ema_retest_v1_bp)

    init_scheduler()
    logger.info("OptiTrade AI v8 app created.")
    return app
