"""
core/order_sync.py
Syncs Alpaca filled orders into our local database so History tab shows real trades.
Runs on every dashboard load and on schedule.
"""
import requests
from datetime import datetime, timezone, timedelta
from core.config import Config
from core.database import log_trade, log_closed_position, get_conn
from core.logger import get_logger

logger = get_logger(__name__)

ALPACA_BASE = Config.ALPACA_BASE_URL

def _headers():
    return {
        "APCA-API-KEY-ID":     Config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": Config.ALPACA_SECRET_KEY,
    }

def _already_synced(alpaca_id: str) -> bool:
    """Check if this Alpaca order ID is already in our trades table."""
    conn = get_conn()
    row  = conn.execute(
        "SELECT id FROM trades WHERE alpaca_id=? LIMIT 1", (alpaca_id,)
    ).fetchone()
    from core.database import _close
    _close(conn)
    return row is not None

def sync_alpaca_orders(days: int = 7) -> int:
    """
    Pull recent filled orders from Alpaca and add any missing ones to our DB.
    Returns number of new records added.
    """
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url   = f"{ALPACA_BASE}/v2/orders"
        params = {"status":"all","limit":200,"after":since,"direction":"desc"}
        r = requests.get(url, headers=_headers(), params=params, timeout=15)
        r.raise_for_status()
        orders = r.json()
        added  = 0
        for o in orders:
            oid    = o.get("id","")
            sym    = o.get("symbol","")
            side   = o.get("side","")
            qty    = float(o.get("filled_qty") or o.get("qty") or 0)
            status = o.get("status","")
            otype  = o.get("type","market")
            filled_at = o.get("filled_at") or o.get("submitted_at","")
            fill_price= float(o.get("filled_avg_price") or 0)

            if not sym or not oid:
                continue

            # Map to our action
            action = side.lower()  # "buy" or "sell"

            # Only add if not already tracked
            if not _already_synced(oid):
                our_status = "placed" if status in ("filled","partially_filled","accepted") else status
                log_trade(sym, action, qty, otype, our_status, oid,
                         f"Synced from Alpaca | fill_price=${fill_price} | {status}")
                added += 1

                # If this is a sell/close, try to match with a prior buy for P&L
                if action == "sell" and fill_price > 0:
                    _try_log_closed_pnl(sym, qty, fill_price, oid)

        if added:
            logger.info(f"Synced {added} orders from Alpaca")
        return added
    except Exception as e:
        logger.warning(f"Alpaca order sync failed: {e}")
        return 0

def _try_log_closed_pnl(symbol: str, qty: float, exit_price: float, alpaca_id: str):
    """
    Try to find the matching buy order to calculate P&L for a completed round trip.
    """
    try:
        conn = get_conn()
        # Find most recent buy for this symbol not yet matched
        row = conn.execute(
            """SELECT id, message FROM trades
               WHERE symbol=? AND action='buy' AND status='placed'
               ORDER BY id DESC LIMIT 1""", (symbol,)
        ).fetchone()
        from core.database import _close
        _close(conn)

        if not row:
            return

        # Try to extract entry price from the message
        msg = row["message"] or ""
        entry_price = 0.0
        if "fill_price=$" in msg:
            try:
                entry_price = float(msg.split("fill_price=$")[1].split(" ")[0].split("|")[0].strip())
            except:
                pass

        if entry_price > 0:
            log_closed_position(symbol, qty, entry_price, exit_price, "long", alpaca_id=alpaca_id)
            logger.info(f"Auto-logged closed P&L: {symbol} entry={entry_price} exit={exit_price}")
    except Exception as e:
        logger.warning(f"Auto P&L calc failed for {symbol}: {e}")


def sync_positions_pnl() -> list:
    """
    Get all closed positions from Alpaca's activity endpoint for full P&L history.
    """
    try:
        url    = f"{ALPACA_BASE}/v2/account/activities/FILL"
        params = {"page_size":100}
        r      = requests.get(url, headers=_headers(), params=params, timeout=15)
        r.raise_for_status()
        activities = r.json()
        return activities
    except Exception as e:
        logger.warning(f"Activities fetch failed: {e}")
        return []
