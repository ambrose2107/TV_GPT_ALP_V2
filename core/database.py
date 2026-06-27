"""
core/database.py — SQLite: trades, webhook_log, closed_positions  v8
v8 changes:
  - get_recent_webhooks() adds timestamp_display field (dual US+JST timezone)
  - get_closed_positions() adds closed_at_display field
  - get_db_connection() alias for analytics_routes compatibility
"""
import sqlite3, os, threading
from core.logger import get_logger

logger = get_logger(__name__)
DB_PATH = os.environ.get("DB_PATH", "trades.db")
_local  = threading.local()

def get_conn():
    global DB_PATH
    DB_PATH = os.environ.get("DB_PATH", "trades.db")
    if DB_PATH == ":memory:":
        if not hasattr(_local, "conn") or _local.conn is None:
            _local.conn = sqlite3.connect(":memory:", check_same_thread=False)
            _local.conn.row_factory = sqlite3.Row
        return _local.conn
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# alias used by analytics_routes
def get_db_connection():
    return get_conn()

def _close(conn):
    if DB_PATH != ":memory:":
        conn.close()

def reset_memory_db():
    if hasattr(_local, "conn") and _local.conn:
        _local.conn.close()
    _local.conn = None

def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    DEFAULT (datetime('now')),
            symbol      TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            quantity    REAL    NOT NULL,
            order_type  TEXT    NOT NULL,
            status      TEXT    NOT NULL,
            alpaca_id   TEXT,
            message     TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS webhook_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    DEFAULT (datetime('now')),
            raw_payload TEXT,
            status      TEXT,
            error       TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS closed_positions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            closed_at    TEXT    DEFAULT (datetime('now')),
            symbol       TEXT    NOT NULL,
            qty          REAL    NOT NULL,
            entry_price  REAL,
            exit_price   REAL,
            pnl          REAL,
            pnl_pct      REAL,
            side         TEXT,
            hold_time    TEXT,
            alpaca_id    TEXT
        )
    """)
    conn.commit()
    _close(conn)

# ── Dual-timezone formatter (inline, no circular import) ─────────────────────
def _dual_tz(ts_str: str) -> str:
    """
    Convert a UTC timestamp string from DB to:
    '2026-05-23 09:35 EDT  |  22:35 JST'
    Falls back to original string on any error.
    """
    if not ts_str:
        return ""
    try:
        import pytz
        from datetime import datetime
        ts = str(ts_str).replace("T", " ").split(".")[0].rstrip("Z")
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        utc = pytz.utc.localize(dt)
        ny  = utc.astimezone(pytz.timezone("US/Eastern"))
        jst = utc.astimezone(pytz.timezone("Asia/Tokyo"))
        return f"{ny.strftime('%Y-%m-%d %H:%M %Z')}  |  {jst.strftime('%H:%M JST')}"
    except Exception:
        return str(ts_str)[:16]

# ── Trades ────────────────────────────────────────────────────────────────────
def log_trade(symbol, action, quantity, order_type, status, alpaca_id=None, message=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO trades (symbol,action,quantity,order_type,status,alpaca_id,message) VALUES (?,?,?,?,?,?,?)",
        (symbol, action, quantity, order_type, status, alpaca_id, message)
    )
    conn.commit()
    _close(conn)

def get_recent_trades(limit=50):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    _close(conn)
    return [dict(r) for r in rows]

def get_all_trades():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM trades ORDER BY id DESC").fetchall()
    _close(conn)
    return [dict(r) for r in rows]

# ── Webhooks ──────────────────────────────────────────────────────────────────
def log_webhook(raw_payload, status, error=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO webhook_log (raw_payload,status,error) VALUES (?,?,?)",
        (str(raw_payload), status, error)
    )
    conn.commit()
    _close(conn)

def get_recent_webhooks(limit=20):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM webhook_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    _close(conn)
    result = []
    for r in rows:
        d = dict(r)
        # v8: add dual-timezone display for Recent Signals
        d["timestamp_display"] = _dual_tz(d.get("timestamp", ""))
        result.append(d)
    return result

# ── Closed Positions ──────────────────────────────────────────────────────────
def log_closed_position(symbol, qty, entry_price, exit_price, side="long",
                        hold_time=None, alpaca_id=None):
    if entry_price and exit_price:
        if side == "long":
            pnl     = (exit_price - entry_price) * qty
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl     = (entry_price - exit_price) * qty
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
    else:
        pnl = pnl_pct = None

    conn = get_conn()
    conn.execute(
        """INSERT INTO closed_positions
           (symbol,qty,entry_price,exit_price,pnl,pnl_pct,side,hold_time,alpaca_id)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (symbol, qty, entry_price, exit_price, pnl, pnl_pct, side, hold_time, alpaca_id)
    )
    conn.commit()
    _close(conn)

def get_closed_positions(limit=100):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM closed_positions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    _close(conn)
    result = []
    for r in rows:
        d = dict(r)
        # v8: add dual-timezone display
        d["closed_at_display"] = _dual_tz(d.get("closed_at", ""))
        result.append(d)
    return result

def get_all_closed_positions():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM closed_positions ORDER BY id DESC").fetchall()
    _close(conn)
    result = []
    for r in rows:
        d = dict(r)
        d["closed_at_display"] = _dual_tz(d.get("closed_at", ""))
        result.append(d)
    return result

def get_closed_summary():
    conn = get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*)          as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers,
            SUM(pnl)          as total_pnl,
            AVG(pnl)          as avg_pnl,
            MAX(pnl)          as best_trade,
            MIN(pnl)          as worst_trade
        FROM closed_positions
        WHERE pnl IS NOT NULL
    """).fetchone()
    _close(conn)
    return dict(row) if row else {}
