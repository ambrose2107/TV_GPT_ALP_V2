"""
core/config.py — All settings from environment variables
"""
import os

class Config:
    SECRET_KEY         = os.environ.get("APP_SECRET_KEY", "change-me-in-production")

    # Alpaca
    ALPACA_API_KEY     = os.environ.get("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY  = os.environ.get("ALPACA_SECRET_KEY", "")
    ALPACA_MODE        = os.environ.get("ALPACA_MODE", "paper")
    ALPACA_BASE_URL    = (
        "https://paper-api.alpaca.markets"
        if os.environ.get("ALPACA_MODE", "paper") == "paper"
        else "https://api.alpaca.markets"
    )

    # Webhook security
    WEBHOOK_SECRET     = os.environ.get("WEBHOOK_SECRET", "my_secret_123")

    # Dashboard
    DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "admin")

    # Telegram
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Risk
    MAX_POSITION_SIZE  = int(os.environ.get("MAX_POSITION_SIZE", 10))
    DAILY_LOSS_LIMIT   = float(os.environ.get("DAILY_LOSS_LIMIT", 500))
    MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", 5))
    KILL_SWITCH        = os.environ.get("KILL_SWITCH", "false").lower() == "true"
