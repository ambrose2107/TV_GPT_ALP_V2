"""
webhook/handler.py — Core signal processor with Telegram alerts
"""
from core.config import Config
from core.logger import get_logger
from core.database import log_trade, log_webhook
from core.telegram import alert_order_placed, alert_order_failed
from brokers.alpaca_adapter import AlpacaAdapter

logger = get_logger(__name__)
alpaca = AlpacaAdapter()

VALID_ACTIONS = {"buy", "sell", "close", "close_all"}

def process_webhook(payload: dict) -> dict:
    log_webhook(payload, "received")

    if Config.KILL_SWITCH:
        msg = "Kill switch is ON — all trading halted."
        logger.warning(msg)
        log_webhook(payload, "rejected", msg)
        return {"success": False, "message": msg, "order": None}

    if payload.get("secret") != Config.WEBHOOK_SECRET:
        msg = "Invalid webhook secret."
        logger.warning(msg)
        log_webhook(payload, "rejected", msg)
        return {"success": False, "message": msg, "order": None}

    action     = str(payload.get("action", "")).lower().strip()
    symbol     = str(payload.get("symbol", "")).upper().strip()
    quantity   = float(payload.get("quantity", 1))
    order_type = str(payload.get("order_type", "market")).lower()
    price      = float(payload.get("price", 0))

    if action not in VALID_ACTIONS:
        msg = f"Unknown action: '{action}'. Must be one of {VALID_ACTIONS}"
        log_webhook(payload, "error", msg)
        return {"success": False, "message": msg, "order": None}

    if not symbol and action not in {"close_all"}:
        msg = "Symbol is required."
        log_webhook(payload, "error", msg)
        return {"success": False, "message": msg, "order": None}

    if quantity <= 0:
        msg = f"Invalid quantity: {quantity}"
        log_webhook(payload, "error", msg)
        return {"success": False, "message": msg, "order": None}

    if quantity > Config.MAX_POSITION_SIZE:
        msg = f"Quantity {quantity} exceeds MAX_POSITION_SIZE {Config.MAX_POSITION_SIZE}"
        log_webhook(payload, "rejected", msg)
        return {"success": False, "message": msg, "order": None}

    # Symbol validation
    if action not in {"close_all"} and symbol:
        validation = alpaca.validate_symbol(symbol)
        if not validation["valid"]:
            hint = f" Try '{validation['suggestion']}' instead." if validation.get("suggestion") else ""
            msg  = f"Invalid symbol '{symbol}': {validation['message']}{hint}"
            log_trade(symbol, action, quantity, order_type, "rejected", message=msg)
            log_webhook(payload, "rejected", msg)
            return {"success": False, "message": msg, "order": None}

    try:
        order_result = None

        if action == "buy":
            _try_close_opposite(symbol, "sell")
            if order_type == "limit" and price > 0:
                order_result = alpaca.place_limit_order(symbol, "buy", quantity, price)
            else:
                order_result = alpaca.place_market_order(symbol, "buy", quantity)

        elif action == "sell":
            _try_close_opposite(symbol, "buy")
            if order_type == "limit" and price > 0:
                order_result = alpaca.place_limit_order(symbol, "sell", quantity, price)
            else:
                order_result = alpaca.place_market_order(symbol, "sell", quantity)

        elif action == "close":
            order_result = alpaca.close_position(symbol)

        elif action == "close_all":
            order_result = alpaca.close_all_positions()

        alpaca_id = order_result.get("id") if order_result else None
        log_trade(symbol, action, quantity, order_type, "placed", alpaca_id)
        log_webhook(payload, "success")

        price_str = f"${price}" if order_type == "limit" and price > 0 else "MARKET"
        alert_order_placed(symbol, action, quantity, price_str)

        msg = f"Order placed: {action.upper()} {quantity} {symbol}"
        return {"success": True, "message": msg, "order": order_result}

    except Exception as e:
        msg = f"Order failed: {str(e)}"
        logger.error(msg)
        log_trade(symbol, action, quantity, order_type, "failed", message=msg)
        log_webhook(payload, "error", msg)
        alert_order_failed(symbol, action, str(e)[:200])
        return {"success": False, "message": msg, "order": None}


def _try_close_opposite(symbol: str, existing_side: str):
    try:
        position = alpaca.get_position(symbol)
        if position:
            qty = float(position.get("qty", 0))
            if existing_side == "sell" and qty < 0:
                alpaca.close_position(symbol)
            elif existing_side == "buy" and qty > 0:
                alpaca.close_position(symbol)
    except Exception as e:
        logger.warning(f"Could not check/close opposite for {symbol}: {e}")
