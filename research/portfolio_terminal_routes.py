"""Portfolio terminal APIs: broker equity history and bounded market breadth."""
from datetime import datetime, timedelta, timezone
import math
import os
import time

import requests
from flask import Blueprint, jsonify, session

from core.config import Config
from core.logger import get_logger
from core.account_history import build_history_summary

terminal_bp = Blueprint("portfolio_terminal", __name__)
logger = get_logger(__name__)
_TERMINAL_CACHE = {"at": 0, "payload": None}
_MARKET_CACHE = {"at": 0, "payload": None}
CACHE_SECONDS = 300

# A disclosed, liquid US large-cap / technology watch universe. This is not
# an exchange-wide scan; the UI labels it accordingly.
STOCK_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","AVGO","TSLA","AMD",
    "MU","INTC","QCOM","TXN","AMAT","LRCX","KLAC","MRVL","TSM","ASML",
    "ARM","ON","ADI","MCHP","NXPI","SNPS","CDNS","ANET","DELL","SMCI",
    "ORCL","CRM","NOW","SNOW","PLTR","PANW","CRWD","FTNT","DDOG","NET",
    "MDB","SHOP","APP","UBER","NFLX","ADBE","INTU","IBM","CSCO","ACN",
    "AMZN","WMT","COST","LLY","JPM","XOM","V","MA","CAT","GE",
    "COIN","HOOD","RBLX","ROKU","SOFI","RIVN","F","GM","DIS","ABNB",
    "LITE","CIEN","COHR","SNDK","WDC","STX","CRWV","NBIS","VRT","ETN",
]
STOCK_UNIVERSE = list(dict.fromkeys(STOCK_UNIVERSE))
SECTOR_PROXIES = {
    "Broad technology": ["XLK", "IYW"],
    "Semiconductors": ["SOXX", "SMH", "XSD"],
    "Software": ["IGV"],
    "Cloud infrastructure": ["SKYY", "WCLD"],
    "Cybersecurity": ["CIBR", "HACK"],
    "Internet platforms": ["FDN"],
    "Communication services": ["XLC"],
    "Technology growth": ["IGM"],
}
ALL_SYMBOLS = list(dict.fromkeys(STOCK_UNIVERSE + [s for group in SECTOR_PROXIES.values() for s in group]))


def _headers():
    key = Config.ALPACA_API_KEY or os.environ.get("APCA_API_KEY_ID", "") or os.environ.get("ALPACA_KEY", "")
    secret = Config.ALPACA_SECRET_KEY or os.environ.get("APCA_API_SECRET_KEY", "") or os.environ.get("ALPACA_SECRET", "")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials are not configured.")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _auth_error():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _price_metrics(symbol, items):
    rows = []
    for item in items or []:
        close = _num(item.get("c"))
        if close is None or close <= 0:
            continue
        rows.append({
            "date": str(item.get("t", ""))[:10],
            "close": close,
            "volume": _num(item.get("v")),
        })
    rows.sort(key=lambda row: row["date"])
    if len(rows) < 2:
        return None
    last, previous = rows[-1], rows[-2]
    day_return = (last["close"] / previous["close"] - 1) * 100
    five_day = (last["close"] / rows[-6]["close"] - 1) * 100 if len(rows) >= 6 else None
    return {
        "symbol": symbol, "price": round(last["close"], 4),
        "day_pct": round(day_return, 2),
        "five_day_pct": round(five_day, 2) if five_day is not None else None,
        "volume": last["volume"], "session": last["date"],
    }


def _load_activities(headers, base):
    rows, token, has_more = [], None, False
    for page in range(10):
        params = {
            "activity_types": "FILL,TRANS,MISC,DIV,DIVCGL,DIVCGS,DIVNRA,DIVROC,DIVTXEX,DIVFT,DIVTW",
            "page_size": 100, "direction": "asc",
        }
        if token:
            params["page_token"] = token
        response = requests.get(base + "/v2/account/activities", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError("Alpaca activities response was not a list.")
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 100:
            break
        next_token = batch[-1].get("id")
        if not next_token or next_token == token:
            break
        token = next_token
        if page == 9:
            has_more = True
    return rows, has_more


@terminal_bp.route("/api/portfolio-manager/terminal")
def portfolio_terminal_data():
    error = _auth_error()
    if error:
        return error
    force = False
    from flask import request
    force = request.args.get("refresh") == "1"
    now = time.time()
    if not force and _TERMINAL_CACHE["payload"] and now - _TERMINAL_CACHE["at"] < CACHE_SECONDS:
        return jsonify(_TERMINAL_CACHE["payload"])
    base = Config.ALPACA_BASE_URL.rstrip("/")
    warnings = []
    payload = {
        "source": "Alpaca Broker API",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "equity_history": [],
        "account": {},
        "capital": {},
        "warnings": warnings,
    }
    try:
        headers = _headers()
        account_response = requests.get(base + "/v2/account", headers=headers, timeout=10)
        account_response.raise_for_status()
        account = account_response.json()
        positions_response = requests.get(base + "/v2/positions", headers=headers, timeout=10)
        positions_response.raise_for_status()
        positions = positions_response.json()
        equity = _num(account.get("equity") or account.get("portfolio_value")) or 0.0
        market_value = sum(_num(p.get("market_value")) or 0.0 for p in positions)
        open_cost = sum(_num(p.get("cost_basis")) or 0.0 for p in positions)
        unrealized = sum(_num(p.get("unrealized_pl")) or 0.0 for p in positions)
        cash = _num(account.get("cash")) or 0.0
        payload["account"] = {
            "equity": equity, "cash": cash, "market_value": market_value,
            "unrealized_pl": unrealized, "buying_power": _num(account.get("buying_power")),
        }
        payload["capital"] = {
            "open_positions_cost_basis": round(open_cost, 2),
            "current_positions_market_value": round(market_value, 2),
            "unrealized_pl": round(unrealized, 2),
            "net_cash_contributions": None,
            "gross_lifetime_buys": None,
            "gross_lifetime_sells": None,
            "activity_count": 0,
            "activity_history_complete": False,
        }
        try:
            activities, has_more = _load_activities(headers, base)
            summary, _ = build_history_summary(activities, positions, account)
            payload["capital"].update({
                "net_cash_contributions": summary.get("net_cash_flows"),
                "gross_lifetime_buys": summary.get("gross_lifetime_buys"),
                "gross_lifetime_sells": summary.get("gross_lifetime_sells"),
                "gross_dividends": summary.get("gross_dividends"),
                "dividend_withheld": summary.get("dividend_withheld"),
                "fees": summary.get("fees"),
                "activity_count": summary.get("activity_count"),
                "activity_history_complete": not has_more,
            })
            if has_more:
                warnings.append("Broker activity history reached the 1,000-record safety limit; lifetime contribution totals may be incomplete.")
        except Exception as exc:
            logger.warning("Portfolio terminal cash-flow history unavailable: %s", exc)
            warnings.append("Cash contribution history is unavailable; live equity and open-position cost basis are still shown.")
        try:
            history_url = base + "/v2/account/portfolio/history"
            history_response = requests.get(
                history_url, headers=headers,
                params={"period": "all", "timeframe": "1D", "extended_hours": "true"},
                timeout=12,
            )
            history_response.raise_for_status()
            history = history_response.json()
            timestamps = history.get("timestamp") or []
            equities = history.get("equity") or []
            curve = []
            for stamp, value in zip(timestamps, equities):
                number = _num(value)
                if number is None or number <= 0:
                    continue
                try:
                    dt = datetime.fromtimestamp(float(stamp), tz=timezone.utc)
                except (TypeError, ValueError, OSError):
                    continue
                curve.append({"timestamp": int(stamp), "date": dt.date().isoformat(), "equity": round(number, 2)})
            payload["equity_history"] = curve
            payload["history_source"] = "Alpaca account portfolio history"
            if not curve:
                warnings.append("Alpaca returned no daily equity history for this account.")
        except Exception as exc:
            logger.warning("Portfolio equity history unavailable: %s", exc)
            warnings.append("Broker equity-history endpoint is unavailable. No synthetic equity curve was substituted.")
        _TERMINAL_CACHE.update({"at": now, "payload": payload})
        return jsonify(payload)
    except requests.RequestException as exc:
        logger.warning("Portfolio terminal account request failed: %s", exc)
        return jsonify({"error": "Could not load the live Alpaca account.", "detail": str(exc)[:220]}), 502
    except Exception as exc:
        logger.exception("Portfolio terminal failed")
        return jsonify({"error": "Portfolio terminal calculation failed.", "detail": str(exc)[:220]}), 500


@terminal_bp.route("/api/portfolio-manager/market-pulse")
def portfolio_market_pulse():
    error = _auth_error()
    if error:
        return error
    from flask import request
    force = request.args.get("refresh") == "1"
    now = time.time()
    if not force and _MARKET_CACHE["payload"] and now - _MARKET_CACHE["at"] < CACHE_SECONDS:
        return jsonify(_MARKET_CACHE["payload"])
    try:
        headers = _headers()
        start = (datetime.now(timezone.utc) - timedelta(days=18)).date().isoformat()
        response = requests.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            headers=headers,
            params={
                "symbols": ",".join(ALL_SYMBOLS), "timeframe": "1Day",
                "start": start, "limit": 10000, "adjustment": "all",
                "feed": "iex", "sort": "asc",
            },
            timeout=18,
        )
        response.raise_for_status()
        body = response.json()
        bars = body.get("bars") if isinstance(body, dict) else None
        if not isinstance(bars, dict):
            raise ValueError("Alpaca market-data response did not contain a bars map.")
        metrics = {symbol: result for symbol, items in bars.items() if (result := _price_metrics(symbol, items)) is not None}
        stocks = [metrics[s] for s in STOCK_UNIVERSE if s in metrics]
        day_sorted = sorted(stocks, key=lambda row: row["day_pct"])
        five_sorted = sorted([x for x in stocks if x["five_day_pct"] is not None], key=lambda row: row["five_day_pct"])
        sectors = []
        for name, members in SECTOR_PROXIES.items():
            available = [metrics[s] for s in members if s in metrics]
            if not available:
                continue
            day_values = [x["day_pct"] for x in available]
            five_values = [x["five_day_pct"] for x in available if x["five_day_pct"] is not None]
            sectors.append({
                "sector": name, "proxies": members,
                "day_pct": round(sum(day_values) / len(day_values), 2),
                "five_day_pct": round(sum(five_values) / len(five_values), 2) if five_values else None,
                "proxy_count": len(available),
            })
        sectors.sort(key=lambda row: row["five_day_pct"] if row["five_day_pct"] is not None else row["day_pct"], reverse=True)
        payload = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "latest_session": max((x["session"] for x in stocks), default=None),
            "source": "Alpaca Market Data API · IEX feed · adjusted daily bars",
            "universe_count": len(stocks),
            "universe_note": "Tracked liquid US large-cap and technology names; not a scan of every US-listed stock.",
            "gainers_today": sorted(stocks, key=lambda row: row["day_pct"], reverse=True)[:8],
            "losers_today": day_sorted[:8],
            "gainers_5d": sorted([x for x in stocks if x["five_day_pct"] is not None], key=lambda row: row["five_day_pct"], reverse=True)[:8],
            "losers_5d": five_sorted[:8],
            "sectors": sectors,
            "warnings": [],
        }
        if len(stocks) < 20:
            payload["warnings"].append("Market-data coverage is thin for the tracked universe; rankings may be incomplete.")
        _MARKET_CACHE.update({"at": now, "payload": payload})
        return jsonify(payload)
    except requests.RequestException as exc:
        logger.warning("Portfolio market pulse request failed: %s", exc)
        return jsonify({"error": "Market movers data is unavailable from Alpaca.", "detail": str(exc)[:220]}), 502
    except Exception as exc:
        logger.exception("Portfolio market pulse failed")
        return jsonify({"error": "Market pulse calculation failed.", "detail": str(exc)[:220]}), 500
