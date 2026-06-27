"""
core/telegram.py — Telegram alert notifications
Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in env vars.

How to set up:
1. Message @BotFather on Telegram -> /newbot -> copy token
2. Message @userinfobot -> copy your chat_id
3. Set both as env vars in Railway / Replit
"""
import requests
from core.config import Config
from core.logger import get_logger

logger = get_logger(__name__)

def send_telegram(message: str) -> bool:
    """Send a message via Telegram bot. Returns True if successful."""
    token   = Config.TELEGRAM_BOT_TOKEN
    chat_id = Config.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        logger.debug("Telegram not configured — skipping notification.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "HTML"
        }, timeout=8)
        resp.raise_for_status()
        logger.info(f"Telegram sent: {message[:60]}...")
        return True
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")
        return False


def alert_order_placed(symbol: str, action: str, qty: float, price: str = "MARKET") -> None:
    emoji = "🟢" if action.lower() == "buy" else "🔴"
    send_telegram(
        f"{emoji} <b>ORDER PLACED</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📌 Symbol:  <b>{symbol}</b>\n"
        f"📋 Action:  <b>{action.upper()}</b>\n"
        f"📦 Qty:     <b>{qty}</b>\n"
        f"💲 Price:   <b>{price}</b>\n"
        f"🤖 Bot:     OptiTrade → Alpaca"
    )

def alert_order_failed(symbol: str, action: str, error: str) -> None:
    send_telegram(
        f"❌ <b>ORDER FAILED</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📌 Symbol:  <b>{symbol}</b>\n"
        f"📋 Action:  <b>{action.upper()}</b>\n"
        f"⚠️ Error:   {error[:200]}"
    )

def alert_position_closed(symbol: str, pnl: float, pnl_pct: float) -> None:
    emoji = "✅" if pnl >= 0 else "❌"
    send_telegram(
        f"{emoji} <b>POSITION CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📌 Symbol:  <b>{symbol}</b>\n"
        f"💰 P&L:     <b>${pnl:+.2f} ({pnl_pct:+.2f}%)</b>"
    )

def alert_kill_switch(enabled: bool) -> None:
    emoji = "🔴" if enabled else "🟢"
    status = "ENABLED — trading HALTED" if enabled else "DISABLED — trading RESUMED"
    send_telegram(f"{emoji} <b>KILL SWITCH {status}</b>")

def alert_daily_loss_limit(current_loss: float, limit: float) -> None:
    send_telegram(
        f"🚨 <b>DAILY LOSS LIMIT REACHED</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📉 Current loss: <b>${current_loss:.2f}</b>\n"
        f"🛑 Limit:        <b>${limit:.2f}</b>\n"
        f"⛔ All trading halted for today."
    )
