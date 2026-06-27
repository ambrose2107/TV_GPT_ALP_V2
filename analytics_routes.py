"""
core/analytics_routes.py — v9
Institutional-grade analytics from Alpaca closed orders.
Features:
- Pulls LIVE from Alpaca /v2/orders (all filled orders, any date range)
- Fallback to local DB closed_positions
- Date range filter, per-symbol breakdown, equity curve
- Portfolio allocation pie data, drawdown, Sharpe ratio, holding time
"""
from flask import Blueprint, jsonify, session, request
from core.database import get_conn, _close, get_closed_positions
from core.logger import get_logger
from core.config import Config
from collections import defaultdict
import requests
from datetime import datetime, timezone, timedelta
import math

logger = get_logger(__name__)
analytics_bp = Blueprint("analytics", __name__)

def _auth():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None

def _alpaca_headers():
    import os
    key = (Config.ALPACA_API_KEY
           or os.environ.get("APCA_API_KEY_ID", "")
           or os.environ.get("ALPACA_KEY", ""))
    sec = (Config.ALPACA_SECRET_KEY
           or os.environ.get("APCA_API_SECRET_KEY", "")
           or os.environ.get("ALPACA_SECRET", ""))
    return {
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": sec,
    }

def _fetch_alpaca_orders(date_from: str = None, date_to: str = None) -> list:
    """
    Pull ALL filled orders from Alpaca, optionally filtered by date range.
    Uses Alpaca's next_page_token for proper pagination (bypasses 500/page limit).
    Default after=2015-01-01 ensures full account history is returned.
    """
    try:
        base = Config.ALPACA_BASE_URL
        url  = f"{base}/v2/orders"

        after_ts = f"{date_from}T00:00:00Z" if date_from else "2015-01-01T00:00:00Z"
        until_ts = f"{date_to}T23:59:59Z"   if date_to   else None

        all_orders  = []
        page_token  = None

        while True:
            params = {
                "status":    "all",
                "limit":     500,
                "direction": "asc",
                "after":     after_ts,
            }
            if until_ts:
                params["until"] = until_ts
            if page_token:
                params["page_token"] = page_token

            r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=15)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_orders.extend(batch)

            # Alpaca returns next_page_token in headers when more pages exist
            page_token = r.headers.get("X-Next-Page-Token") or r.headers.get("next_page_token")
            if not page_token or len(batch) < 500:
                break

        trades = []
        for o in all_orders:
            if o.get("status") not in ("filled", "partially_filled"):
                continue
            sym   = o.get("symbol", "")
            side  = o.get("side", "")
            qty   = float(o.get("filled_qty") or o.get("qty") or 0)
            price = float(o.get("filled_avg_price") or 0)
            ts    = o.get("filled_at") or o.get("submitted_at") or ""
            if not sym or qty == 0:
                continue
            trades.append({
                "symbol":           sym,
                "side":             side,
                "qty":              qty,
                "price":            price,
                "timestamp":        ts[:19].replace("T", " "),
                "date":             ts[:10],
                "alpaca_id":        o.get("id", ""),
                "position_intent":  o.get("position_intent", ""),
                "source":           "alpaca",
            })

        logger.info(f"Fetched {len(trades)} filled orders from Alpaca (scanned {len(all_orders)} total)")
        return trades
    except Exception as e:
        logger.warning(f"Alpaca orders fetch failed: {e}")
        return []

def _build_round_trips(trades: list) -> list:
    """
    Match buy→sell pairs per symbol into closed round-trip P&L records.
    Uses FIFO matching.

    Orphan sells (sell with no matching buy in this dataset — common when the
    opening buy predates the query window) are included as best-effort records
    using the sell price as both entry and exit so they still appear in the
    analytics table (P&L shown as 0 since entry is unknown).

    This ensures a sell-only order history still renders charts correctly.
    """
    buys  = defaultdict(list)
    sells = defaultdict(list)
    for t in trades:
        sym = t["symbol"]
        if t["side"] == "buy":
            buys[sym].append(dict(t))   # copy so we can mutate qty
        else:
            sells[sym].append(t)

    round_trips = []
    all_syms = set(list(buys.keys()) + list(sells.keys()))

    for sym in all_syms:
        bq = list(buys[sym])   # FIFO queue (mutable copies)
        sq = list(sells[sym])

        for sell in sq:
            remaining_sell_qty = sell["qty"]

            # Match against available buys (FIFO)
            while remaining_sell_qty > 0 and bq:
                buy         = bq[0]
                matched_qty = min(buy["qty"], remaining_sell_qty)
                pnl         = (sell["price"] - buy["price"]) * matched_qty
                pnl_pct     = ((sell["price"] - buy["price"]) / buy["price"] * 100) if buy["price"] > 0 else 0
                round_trips.append({
                    "symbol":      sym,
                    "side":        "long",
                    "qty":         matched_qty,
                    "entry_price": buy["price"],
                    "exit_price":  sell["price"],
                    "pnl":         round(pnl, 2),
                    "pnl_pct":     round(pnl_pct, 2),
                    "entry_date":  buy["date"],
                    "exit_date":   sell["date"],
                    "closed_at":   sell["timestamp"],
                    "source":      sell.get("source", "alpaca"),
                })
                buy["qty"]         -= matched_qty
                remaining_sell_qty -= matched_qty
                if buy["qty"] <= 0:
                    bq.pop(0)

            # Orphan sell — no matching buy found (position opened outside date window)
            # Record with entry_price = exit_price so pnl = 0 but trade still shows up
            if remaining_sell_qty > 0:
                round_trips.append({
                    "symbol":      sym,
                    "side":        "long",
                    "qty":         remaining_sell_qty,
                    "entry_price": sell["price"],   # unknown — use sell price
                    "exit_price":  sell["price"],
                    "pnl":         0.0,             # unknown
                    "pnl_pct":     0.0,
                    "entry_date":  sell["date"],    # unknown
                    "exit_date":   sell["date"],
                    "closed_at":   sell["timestamp"],
                    "source":      sell.get("source", "alpaca"),
                    "orphan":      True,            # flag so UI can note it
                })

    return sorted(round_trips, key=lambda x: x["closed_at"])

def _compute_analytics(rows: list) -> dict:
    """Full institutional-grade analytics from a list of closed trade rows."""
    if not rows:
        return {
            "symbols": [], "totals": {}, "equity_curve": [],
            "allocation_pie": [], "monthly_pnl": [], "drawdown": [],
            "raw_count": 0
        }

    by_symbol = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "win_pnl": 0.0, "loss_pnl": 0.0,
        "max_win": 0.0, "max_loss": 0.0,
        "total_investment": 0.0,
    })
    timeline = []

    for r in rows:
        sym  = r.get("symbol", "UNKNOWN")
        pnl  = float(r.get("pnl") or 0)
        qty  = float(r.get("qty") or 0)
        ep   = float(r.get("entry_price") or 0)
        s    = by_symbol[sym]
        s["trades"]           += 1
        s["total_pnl"]        += pnl
        s["total_investment"] += qty * ep
        if pnl > 0:
            s["wins"]    += 1;  s["win_pnl"] += pnl
            s["max_win"]  = max(s["max_win"], pnl)
        else:
            s["losses"]  += 1;  s["loss_pnl"] += pnl
            s["max_loss"] = min(s["max_loss"], pnl)
        date_str = str(r.get("closed_at", r.get("exit_date", "")))[:10]
        timeline.append({"date": date_str, "pnl": pnl, "symbol": sym,
                          "qty": qty, "entry_price": ep})

    # Per-symbol output
    symbols_out = []
    for sym, s in sorted(by_symbol.items(), key=lambda x: -x[1]["trades"]):
        wr    = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0
        aw    = round(s["win_pnl"]  / s["wins"],   2) if s["wins"]   > 0 else 0
        al    = round(s["loss_pnl"] / s["losses"], 2) if s["losses"] > 0 else 0
        pf    = round(abs(s["win_pnl"] / s["loss_pnl"]), 2) if s["loss_pnl"] != 0 else 99.0
        roi   = round(s["total_pnl"] / s["total_investment"] * 100, 2) if s["total_investment"] > 0 else 0
        symbols_out.append({
            "symbol":        sym,
            "trades":        s["trades"],
            "wins":          s["wins"],
            "losses":        s["losses"],
            "win_rate":      wr,
            "total_pnl":     round(s["total_pnl"], 2),
            "avg_win":       aw,
            "avg_loss":      al,
            "profit_factor": pf,
            "max_win":       round(s["max_win"],  2),
            "max_loss":      round(s["max_loss"], 2),
            "total_investment": round(s["total_investment"], 2),
            "roi_pct":       roi,
        })

    # Overall totals
    n         = len(rows)
    wins      = sum(1 for r in rows if float(r.get("pnl") or 0) > 0)
    all_pnl   = sum(float(r.get("pnl") or 0) for r in rows)
    win_pnl   = sum(float(r.get("pnl") or 0) for r in rows if float(r.get("pnl") or 0) > 0)
    loss_pnl  = sum(float(r.get("pnl") or 0) for r in rows if float(r.get("pnl") or 0) <= 0)
    total_inv = sum(float(r.get("qty") or 0) * float(r.get("entry_price") or 0) for r in rows)
    pnl_list  = [float(r.get("pnl") or 0) for r in rows]
    avg_pnl   = all_pnl / n if n > 0 else 0
    var       = sum((x - avg_pnl)**2 for x in pnl_list) / n if n > 0 else 0
    sharpe    = round(avg_pnl / math.sqrt(var) if var > 0 else 0, 2)

    totals = {
        "total_trades":    n,
        "total_wins":      wins,
        "total_losses":    n - wins,
        "win_rate":        round(wins / n * 100, 1) if n else 0,
        "total_pnl":       round(all_pnl, 2),
        "profit_factor":   round(abs(win_pnl / loss_pnl), 2) if loss_pnl != 0 else 99.0,
        "avg_trade_pnl":   round(avg_pnl, 2),
        "best_trade":      round(max(pnl_list, default=0), 2),
        "worst_trade":     round(min(pnl_list, default=0), 2),
        "total_investment":round(total_inv, 2),
        "overall_roi_pct": round(all_pnl / total_inv * 100, 2) if total_inv > 0 else 0,
        "sharpe_approx":   sharpe,
        "expectancy":      round(avg_pnl, 2),
    }

    # Equity curve + drawdown
    timeline_sorted = sorted(timeline, key=lambda x: x["date"])
    cumulative = 0
    peak       = 0
    equity_curve = []
    drawdown     = []
    for t in timeline_sorted:
        cumulative += t["pnl"]
        peak = max(peak, cumulative)
        dd   = round(cumulative - peak, 2)
        equity_curve.append({
            "date":           t["date"],
            "cumulative_pnl": round(cumulative, 2),
            "trade_pnl":      round(t["pnl"], 2),
            "symbol":         t["symbol"],
        })
        drawdown.append({"date": t["date"], "drawdown": dd})

    max_dd = round(min((x["drawdown"] for x in drawdown), default=0), 2)
    totals["max_drawdown"] = max_dd

    # Monthly P&L
    monthly = defaultdict(float)
    for t in timeline_sorted:
        mo = t["date"][:7]  # YYYY-MM
        monthly[mo] += t["pnl"]
    monthly_pnl = [{"month": mo, "pnl": round(pnl, 2)}
                   for mo, pnl in sorted(monthly.items())]

    # Portfolio allocation pie (total $ invested per symbol as % of total)
    total_all_inv = sum(s["total_investment"] for s in symbols_out) or 1
    allocation_pie = [
        {
            "symbol":   s["symbol"],
            "invested": round(s["total_investment"], 2),
            "pct":      round(s["total_investment"] / total_all_inv * 100, 1),
            "pnl":      s["total_pnl"],
            "roi_pct":  s["roi_pct"],
        }
        for s in symbols_out
    ]

    return {
        "symbols":       symbols_out,
        "totals":        totals,
        "equity_curve":  equity_curve,
        "drawdown":      drawdown,
        "allocation_pie":allocation_pie,
        "monthly_pnl":   monthly_pnl,
        "raw_count":     n,
    }


@analytics_bp.route("/api/analytics/summary", methods=["GET", "POST"])
def analytics_summary():
    e = _auth()
    if e: return e

    body      = request.get_json(silent=True) or {}
    date_from = body.get("date_from") or request.args.get("date_from")
    date_to   = body.get("date_to")   or request.args.get("date_to")
    source    = body.get("source", "auto")  # "alpaca", "db", "auto"

    rows = []
    alpaca_available = bool(Config.ALPACA_API_KEY and Config.ALPACA_SECRET_KEY)

    # Try Alpaca if keys are configured
    if source in ("alpaca", "auto") and alpaca_available:
        alpaca_trades = _fetch_alpaca_orders(date_from, date_to)
        if alpaca_trades:
            rows = _build_round_trips(alpaca_trades)

    # Always fall back to local DB (primary source when Alpaca not configured or returns nothing)
    if not rows:
        db_rows = get_closed_positions(limit=2000)
        if date_from or date_to:
            df = date_from or "2000-01-01"
            dt = date_to   or "2099-12-31"
            db_rows = [r for r in db_rows if df <= str(r.get("closed_at",""))[:10] <= dt]
        rows = db_rows

    if not rows:
        return jsonify({
            "symbols": [], "totals": {}, "equity_curve": [],
            "allocation_pie": [], "monthly_pnl": [], "drawdown": [],
            "raw_count": 0, "source": "empty"
        })

    result = _compute_analytics(rows)
    result["source"] = rows[0].get("source", "db") if rows else "empty"
    result["date_from"] = date_from
    result["date_to"]   = date_to
    return jsonify(result)


@analytics_bp.route("/api/analytics/symbol/<symbol>", methods=["GET"])
def analytics_symbol_detail(symbol):
    e = _auth()
    if e: return e
    date_from = request.args.get("date_from")
    date_to   = request.args.get("date_to")
    alpaca_trades = _fetch_alpaca_orders(date_from, date_to)
    if alpaca_trades:
        trips = _build_round_trips(alpaca_trades)
        sym_rows = [r for r in trips if r.get("symbol","").upper() == symbol.upper()]
    else:
        db_rows  = get_closed_positions(limit=2000)
        sym_rows = [r for r in db_rows if r.get("symbol","").upper() == symbol.upper()]
    return jsonify({"symbol": symbol.upper(), "trades": sym_rows})


@analytics_bp.route("/api/analytics/debug", methods=["GET"])
def analytics_debug():
    """Debug endpoint — shows raw Alpaca order count + sample. Remove in production."""
    e = _auth()
    if e: return e
    try:
        base = Config.ALPACA_BASE_URL
        headers = _alpaca_headers()
        # Quick account check
        acct = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
        # Raw orders (last 30 days)
        orders = requests.get(f"{base}/v2/orders",
                              headers=headers,
                              params={"status": "all", "limit": 10, "direction": "desc"},
                              timeout=10)
        return jsonify({
            "alpaca_base_url": base,
            "has_api_key":     bool(Config.ALPACA_API_KEY),
            "account_status":  acct.status_code,
            "account_ok":      acct.ok,
            "orders_status":   orders.status_code,
            "orders_sample":   orders.json()[:3] if orders.ok else orders.text[:300],
        })
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@analytics_bp.route("/api/calc/price/<symbol>", methods=["GET"])
def calc_price(symbol):
    """Return latest trade price for a symbol from Alpaca."""
    e = _auth()
    if e: return e
    symbol = symbol.upper().strip()
    try:
        base = Config.ALPACA_BASE_URL
        # Try latest trade first
        r = requests.get(
            f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
            headers=_alpaca_headers(),
            timeout=8,
        )
        if r.ok:
            price = r.json().get("trade", {}).get("p")
            if price:
                return jsonify({"symbol": symbol, "price": round(float(price), 4)})

        # Fallback: latest bar
        r2 = requests.get(
            f"https://data.alpaca.markets/v2/stocks/{symbol}/bars/latest",
            headers=_alpaca_headers(),
            timeout=8,
        )
        if r2.ok:
            price = r2.json().get("bar", {}).get("c")
            if price:
                return jsonify({"symbol": symbol, "price": round(float(price), 4)})

        # Fallback: last filled order for that symbol
        orders = requests.get(
            f"{base}/v2/orders",
            headers=_alpaca_headers(),
            params={"status": "filled", "symbols": symbol, "limit": 1, "direction": "desc"},
            timeout=8,
        )
        if orders.ok and orders.json():
            o = orders.json()[0]
            price = float(o.get("filled_avg_price") or 0)
            if price:
                return jsonify({"symbol": symbol, "price": round(price, 4), "source": "last_order"})

        return jsonify({"error": f"No price found for {symbol}"}), 404
    except Exception as ex:
        logger.warning(f"calc_price error for {symbol}: {ex}")
        return jsonify({"error": str(ex)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PORTFOLIO ANALYTICS — open positions breakdown
# Used by the Analytics tab (replaces closed-orders view)
# ─────────────────────────────────────────────────────────────────────────────

# Sector mapping for common symbols (extended list)
_SECTOR_MAP = {
    # Technology
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMD":"Technology",
    "INTC":"Technology","QCOM":"Technology","AVGO":"Technology","TXN":"Technology",
    "MU":"Technology","AMAT":"Technology","KLAC":"Technology","LRCX":"Technology",
    "MRVL":"Technology","SMCI":"Technology","SNOW":"Technology","CRM":"Technology",
    "ORCL":"Technology","IBM":"Technology","HPQ":"Technology","DELL":"Technology",
    "PANW":"Technology","CRWD":"Technology","ZS":"Technology","OKTA":"Technology",
    "PLTR":"Technology","DDOG":"Technology","NET":"Technology","MDB":"Technology",
    # Consumer / Internet
    "AMZN":"Consumer Disc.","TSLA":"Consumer Disc.","HD":"Consumer Disc.",
    "MCD":"Consumer Disc.","NKE":"Consumer Disc.","SBUX":"Consumer Disc.",
    "BKNG":"Consumer Disc.","ABNB":"Consumer Disc.","LYFT":"Consumer Disc.",
    "UBER":"Consumer Disc.","DASH":"Consumer Disc.",
    # Communication
    "META":"Communication","GOOGL":"Communication","GOOG":"Communication",
    "NFLX":"Communication","DIS":"Communication","PARA":"Communication",
    "T":"Communication","VZ":"Communication","TMUS":"Communication",
    # Financials
    "JPM":"Financials","BAC":"Financials","GS":"Financials","MS":"Financials",
    "C":"Financials","WFC":"Financials","BX":"Financials","BLK":"Financials",
    "V":"Financials","MA":"Financials","PYPL":"Financials","SQ":"Financials",
    "COIN":"Financials","HOOD":"Financials","MARA":"Financials","RIOT":"Financials",
    # Healthcare
    "JNJ":"Healthcare","UNH":"Healthcare","PFE":"Healthcare","MRNA":"Healthcare",
    "LLY":"Healthcare","ABT":"Healthcare","TMO":"Healthcare","ISRG":"Healthcare",
    "GILD":"Healthcare","BIIB":"Healthcare","REGN":"Healthcare","VRTX":"Healthcare",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
    "OXY":"Energy","MPC":"Energy","PSX":"Energy",
    # ETFs
    "SPY":"ETF","QQQ":"ETF","IWM":"ETF","DIA":"ETF","GLD":"ETF",
    "SLV":"ETF","TLT":"ETF","HYG":"ETF","ARKK":"ETF","SQQQ":"ETF","TQQQ":"ETF",
    # Industrials
    "BA":"Industrials","CAT":"Industrials","DE":"Industrials","GE":"Industrials",
    "RTX":"Industrials","LMT":"Industrials","NOC":"Industrials",
    # Real Estate
    "AMT":"Real Estate","PLD":"Real Estate","EQIX":"Real Estate",
}

def _get_sector(symbol: str) -> str:
    return _SECTOR_MAP.get(symbol.upper(), "Other")


@analytics_bp.route("/api/analytics/portfolio")
def api_portfolio_analytics():
    """
    Live open-position analytics.
    Returns per-symbol P&L, sector breakdown, allocation, best/worst performers.
    """
    e = _auth()
    if e: return e

    base = Config.ALPACA_BASE_URL
    hdrs = _alpaca_headers()
    if not hdrs.get("APCA-API-KEY-ID"):
        return jsonify({"error": "Alpaca API key not set. Add ALPACA_API_KEY to env vars."}), 503

    try:
        # ── 1. Account ────────────────────────────────────────────────────────
        acc_r = requests.get(f"{base}/v2/account", headers=hdrs, timeout=10)
        acc_r.raise_for_status()
        acc = acc_r.json()
        equity      = float(acc.get("equity", 0) or 0)
        cash        = float(acc.get("cash", 0) or 0)
        buying_power= float(acc.get("buying_power", 0) or 0)
        portfolio_value = float(acc.get("portfolio_value", equity) or equity)

        # ── 2. Positions ──────────────────────────────────────────────────────
        pos_r = requests.get(f"{base}/v2/positions", headers=hdrs, timeout=10)
        pos_r.raise_for_status()
        raw_positions = pos_r.json()

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            return jsonify({"error": "Alpaca 401 — check ALPACA_API_KEY / ALPACA_SECRET_KEY"}), 401
        return jsonify({"error": str(e)}), 500
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    if not raw_positions:
        return jsonify({
            "positions": [], "sector_breakdown": {}, "summary": {},
            "account": {"equity": equity, "cash": cash, "buying_power": buying_power,
                        "portfolio_value": portfolio_value},
            "message": "No open positions in your Alpaca account.",
        })

    # ── 3. Build per-symbol stats ─────────────────────────────────────────────
    positions = []
    sector_totals: dict[str, dict] = {}
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    total_cost_basis = 0.0

    for p in raw_positions:
        sym          = p.get("symbol", "")
        qty          = float(p.get("qty", 0) or 0)
        side         = p.get("side", "long")
        avg_entry    = float(p.get("avg_entry_price", 0) or 0)
        current_price= float(p.get("current_price", 0) or 0)
        market_value = float(p.get("market_value", 0) or 0)
        cost_basis   = float(p.get("cost_basis", 0) or 0)
        unreal_pl    = float(p.get("unrealized_pl", 0) or 0)
        unreal_pct   = float(p.get("unrealized_plpc", 0) or 0) * 100
        sector       = _get_sector(sym)

        pos_data = {
            "symbol":          sym,
            "qty":             qty,
            "side":            side,
            "avg_entry":       round(avg_entry, 4),
            "current_price":   round(current_price, 4),
            "market_value":    round(market_value, 2),
            "cost_basis":      round(cost_basis, 2),
            "unrealized_pl":   round(unreal_pl, 2),
            "unrealized_pct":  round(unreal_pct, 3),
            "sector":          sector,
            "allocation_pct":  round(market_value / portfolio_value * 100, 2) if portfolio_value else 0,
        }
        positions.append(pos_data)

        total_market_value  += market_value
        total_unrealized_pl += unreal_pl
        total_cost_basis    += cost_basis

        # Sector rollup
        if sector not in sector_totals:
            sector_totals[sector] = {"market_value": 0.0, "unrealized_pl": 0.0,
                                      "count": 0, "symbols": []}
        sector_totals[sector]["market_value"]   += market_value
        sector_totals[sector]["unrealized_pl"]  += unreal_pl
        sector_totals[sector]["count"]          += 1
        sector_totals[sector]["symbols"].append(sym)

    # Sort by unrealized P&L desc
    positions.sort(key=lambda x: x["unrealized_pl"], reverse=True)

    # ── 4. Sector % of portfolio ──────────────────────────────────────────────
    sector_breakdown = {}
    for sec, data in sorted(sector_totals.items(),
                             key=lambda x: x[1]["market_value"], reverse=True):
        pct = data["market_value"] / portfolio_value * 100 if portfolio_value else 0
        sector_breakdown[sec] = {
            "market_value":  round(data["market_value"], 2),
            "unrealized_pl": round(data["unrealized_pl"], 2),
            "allocation_pct": round(pct, 2),
            "count":         data["count"],
            "symbols":       data["symbols"],
        }

    # ── 5. Best / Worst performers ────────────────────────────────────────────
    best  = max(positions, key=lambda x: x["unrealized_pct"]) if positions else {}
    worst = min(positions, key=lambda x: x["unrealized_pct"]) if positions else {}

    total_invested_pct = total_market_value / portfolio_value * 100 if portfolio_value else 0
    roi_pct = total_unrealized_pl / total_cost_basis * 100 if total_cost_basis else 0

    return jsonify({
        "positions":        positions,
        "sector_breakdown": sector_breakdown,
        "account": {
            "equity":          round(equity, 2),
            "cash":            round(cash, 2),
            "buying_power":    round(buying_power, 2),
            "portfolio_value": round(portfolio_value, 2),
        },
        "summary": {
            "open_count":           len(positions),
            "total_market_value":   round(total_market_value, 2),
            "total_unrealized_pl":  round(total_unrealized_pl, 2),
            "total_cost_basis":     round(total_cost_basis, 2),
            "roi_pct":              round(roi_pct, 3),
            "invested_pct":         round(total_invested_pct, 2),
            "cash_pct":             round(cash / portfolio_value * 100, 2) if portfolio_value else 0,
            "best_symbol":          best.get("symbol", "—"),
            "best_pct":             best.get("unrealized_pct", 0),
            "best_pl":              best.get("unrealized_pl", 0),
            "worst_symbol":         worst.get("symbol", "—"),
            "worst_pct":            worst.get("unrealized_pct", 0),
            "worst_pl":             worst.get("unrealized_pl", 0),
        },
    })
