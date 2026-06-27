"""
core/market_data.py — Market data engine
Priority: Alpaca Data API → yfinance → Demo mode
All three work on Railway. Sandbox shows demo data.
"""
import os, math, time, random, requests
from datetime import datetime, timedelta, timezone
from core.config import Config
from core.logger import get_logger

logger = get_logger(__name__)

ALPACA_DATA_URL = "https://data.alpaca.markets"

def _alpaca_headers():
    # Support both naming conventions used across different deployments
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

def _yahoo_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "application/json,text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

# ── Check if data sources are reachable ─────────────────────────────────────
def _alpaca_reachable():
    try:
        r = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/AAPL/bars",
            params={"timeframe":"1Day","limit":1,"feed":"iex"},
            headers=_alpaca_headers(), timeout=8)
        return r.status_code == 200
    except:
        return False

def _yahoo_reachable():
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
            params={"interval":"1d","range":"5d"},
            headers=_yahoo_headers(), timeout=8)
        return r.status_code == 200
    except:
        return False

# ── ALPACA DATA API ──────────────────────────────────────────────────────────
def alpaca_get_bars(symbol: str, timeframe: str = "1Day",
                    limit: int = 200, feed: str = "iex",
                    start=None):
    """
    Get OHLCV bars from Alpaca Data API.
    Always passes a `start` date so the full period is covered.
    Tries iex feed first (free/delayed), then sip, then no-feed param.
    Returns list of dicts: {t, o, h, l, c, v}
    """
    hdrs = _alpaca_headers()
    if not hdrs.get("APCA-API-KEY-ID"):
        logger.warning("⚠️ Alpaca API key not set — skipping Alpaca, falling back to Yahoo")
        return None

    # Calculate start date from limit + timeframe if not provided
    if not start:
        now = datetime.now(timezone.utc)
        if timeframe in ("1Day", "1Day"):
            # daily bars: go back limit trading days × 1.5 to cover weekends/holidays
            start = (now - timedelta(days=int(limit * 1.5))).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif timeframe in ("1Week",):
            start = (now - timedelta(weeks=int(limit * 1.5))).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif timeframe in ("1Hour", "4Hour"):
            start = (now - timedelta(hours=int(limit * 1.8))).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif timeframe in ("5Min", "15Min"):
            start = (now - timedelta(minutes=int(limit * 2))).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start = (now - timedelta(days=int(limit * 1.5))).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol.upper()}/bars"
    base_params = {"timeframe": timeframe, "limit": limit, "start": start,
                   "sort": "asc", "adjustment": "raw"}

    # Try feeds in order: iex (free delayed), sip (subscription), then default
    for attempt_feed in ["iex", "sip", None]:
        try:
            params = dict(base_params)
            if attempt_feed:
                params["feed"] = attempt_feed
            r = requests.get(url, params=params, headers=hdrs, timeout=12)
            if r.status_code == 403:
                logger.debug(f"Alpaca 403 {symbol} feed={attempt_feed}, trying next")
                continue
            if r.status_code == 401:
                logger.warning("Alpaca 401 Unauthorized — check ALPACA_API_KEY / ALPACA_SECRET_KEY env vars")
                return None
            r.raise_for_status()
            bars = r.json().get("bars", [])
            if bars:
                logger.info(f"✅ Alpaca: {symbol} {timeframe} feed={attempt_feed or 'default'} ({len(bars)} bars) from {start}")
                return bars
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                continue
            logger.warning(f"Alpaca bars {symbol} {timeframe} feed={attempt_feed}: {e}")
        except Exception as e:
            logger.warning(f"Alpaca bars {symbol} {timeframe}: {e}")
            break
    logger.warning(f"⚠️ Alpaca bars failed for {symbol} — falling back to Yahoo")
    return None


def alpaca_get_multi_bars(symbols, timeframe: str = "1Day",
                          limit: int = 100):
    """Get bars for multiple symbols in one call."""
    try:
        url = f"{ALPACA_DATA_URL}/v2/stocks/bars"
        params = {"symbols": ",".join(s.upper() for s in symbols),
                  "timeframe": timeframe, "limit": limit,
                  "feed": "iex", "sort": "asc", "adjustment": "raw"}
        r = requests.get(url, params=params, headers=_alpaca_headers(), timeout=15)
        r.raise_for_status()
        return r.json().get("bars", {})
    except Exception as e:
        logger.warning(f"Alpaca multi-bars: {e}")
        return None


def alpaca_get_snapshot(symbol: str):
    """Get latest quote + trade + daily bar for a symbol."""
    try:
        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol.upper()}/snapshot"
        r   = requests.get(url, params={"feed":"iex"}, headers=_alpaca_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Alpaca snapshot {symbol}: {e}")
        return None


def alpaca_get_multi_snapshots(symbols):
    """Snapshots for multiple symbols."""
    try:
        url = f"{ALPACA_DATA_URL}/v2/stocks/snapshots"
        r   = requests.get(url, params={"symbols":",".join(s.upper() for s in symbols),
                                         "feed":"iex"},
                           headers=_alpaca_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Alpaca multi-snapshots: {e}")
        return None


# ── YAHOO FINANCE FALLBACK ────────────────────────────────────────────────────
def yahoo_get_chart(symbol: str, interval: str = "1d",
                    period: str = "6mo"):
    """Yahoo Finance v8 chart data."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
        r   = requests.get(url, params={"interval":interval,"range":period},
                           headers=_yahoo_headers(), timeout=12)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        ts  = res["timestamp"]
        q   = res["indicators"]["quote"][0]
        return {"timestamps":ts, "open":q.get("open",[]),
                "high":q.get("high",[]), "low":q.get("low",[]),
                "close":q.get("close",[]), "volume":q.get("volume",[])}
    except Exception as e:
        logger.warning(f"Yahoo chart {symbol}: {e}")
        return None


def yahoo_get_options(symbol: str):
    """Yahoo Finance options chain."""
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/options/{symbol.upper()}"
        r   = requests.get(url, headers=_yahoo_headers(), timeout=12)
        r.raise_for_status()
        res = r.json()["optionChain"]["result"][0]
        return {
            "symbol":  symbol.upper(),
            "spot":    res.get("quote",{}).get("regularMarketPrice",0),
            "expiries":res.get("expirationDates",[]),
            "calls":   res.get("options",[{}])[0].get("calls",[]),
            "puts":    res.get("options",[{}])[0].get("puts",[]),
        }
    except Exception as e:
        logger.warning(f"Yahoo options {symbol}: {e}")
        return None


def yahoo_earnings_date(symbol: str):
    """Get next earnings date from Yahoo."""
    try:
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol.upper()}"
        r   = requests.get(url, params={"modules":"calendarEvents"},
                           headers=_yahoo_headers(), timeout=10)
        r.raise_for_status()
        dates = (r.json()["quoteSummary"]["result"][0]
                  ["calendarEvents"]["earnings"].get("earningsDate",[]))
        if dates:
            return datetime.fromtimestamp(dates[0]["raw"]).strftime("%Y-%m-%d")
    except:
        pass
    return None


# ── DEMO DATA (sandbox / offline fallback) ────────────────────────────────────
def _gen_price_series(seed_price: float, n: int, vol: float = 0.015):
    """Generate realistic random-walk price series."""
    random.seed(hash(seed_price) % 1000)
    prices, p = [], seed_price
    for _ in range(n):
        p = p * (1 + random.gauss(0.0002, vol))
        prices.append(round(p, 2))
    return prices


def demo_bars(symbol: str, n: int = 200, weekly: bool = False,
              intraday_minutes: int = 0):
    """
    Generate demo OHLCV bars with CORRECT timestamps ending at TODAY.
    Works backwards from now to find the true start date, then fills forward.
    intraday_minutes: 0=daily/weekly, 5=5m, 15=15m, 60=1h, 240=4h
    """
    # ── Updated seed prices reflecting ~May 2026 levels ──────────────────────
    SEEDS = {
        "AAPL":200, "MSFT":420, "NVDA":130, "TSLA":255, "AMD":115,
        "META":585, "GOOGL":170, "JPM":240, "SPY":575, "QQQ":490,
        "XLK":230, "XLV":145, "XLF":48,  "XLY":195, "XLI":135,
        "XLC":100, "XLP":82,  "XLE":88,  "XLB":90,  "XLU":72, "XLRE":40,
        "MU":100,  "AMZN":210,"NFLX":1080,"CRM":320, "SOXX":230,
        "GLD":325, "SLV":33,  "USO":72,  "IWM":205, "DIA":425,
        "PINS":35, "SNAP":12, "UBER":75, "PLTR":115, "COIN":240,
        "RBLX":55, "DKNG":22, "BITO":25, "NFLX":1080,
    }
    seed = SEEDS.get(symbol.upper(), 100)
    vol  = 0.003 if intraday_minutes > 0 else 0.015

    closes = _gen_price_series(seed, n * 2, vol=vol)  # extra buffer

    now = datetime.now(timezone.utc)

    if intraday_minutes > 0:
        # ── Intraday bars: count backwards from NOW through NYSE market hours ─
        # NYSE hours UTC: 13:30-20:00 (9:30-16:00 ET)
        step = timedelta(minutes=intraday_minutes)

        # Start from current bar boundary, walk backwards collecting valid slots
        ts = now.replace(second=0, microsecond=0)
        ts_min = ts.hour * 60 + ts.minute
        ts_min = (ts_min // intraday_minutes) * intraday_minutes
        ts = ts.replace(hour=ts_min // 60, minute=ts_min % 60)

        valid_slots = []
        safety = 0
        while len(valid_slots) < n and safety < n * 20:
            safety += 1
            ts -= step
            if ts.weekday() >= 5:
                continue
            mins_utc = ts.hour * 60 + ts.minute
            if mins_utc < 810 or mins_utc >= 1200:
                continue
            valid_slots.append(ts)

        valid_slots.reverse()  # oldest first

        bars = []
        for i, slot in enumerate(valid_slots[-n:]):
            c = closes[i % len(closes)]
            o = closes[(i - 1) % len(closes)] if i > 0 else c
            hi = max(o, c) * (1 + random.uniform(0, 0.002))
            lo = min(o, c) * (1 - random.uniform(0, 0.002))
            bars.append({
                "t":      slot.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "o":      round(o, 2), "h": round(hi, 2),
                "l":      round(lo, 2), "c": round(c, 2),
                "v":      int(random.uniform(50_000, 2_000_000)),
                "source": "demo"
            })
        return bars

    # ── Daily / weekly bars ───────────────────────────────────────────────────
    # Count backwards from TODAY to find the exact start date
    step = timedelta(weeks=1) if weekly else timedelta(days=1)
    start = now
    count = 0
    while count < n:
        start -= step
        if not weekly and start.weekday() >= 5:
            continue
        count += 1

    # Now walk forward from start, generating exactly n bars ending at/near today
    bars  = []
    cidx  = 0
    d     = start
    for _ in range(n * 3):  # safety limit
        if len(bars) >= n:
            break
        d += step
        if not weekly and d.weekday() >= 5:
            continue
        if d > now + timedelta(days=3):  # never go more than 3 days past today
            break
        c = closes[cidx % len(closes)]
        cidx += 1
        o = closes[(cidx - 2) % len(closes)] if cidx > 1 else c
        hi = max(o, c) * (1 + random.uniform(0, 0.008))
        lo = min(o, c) * (1 - random.uniform(0, 0.008))
        bars.append({
            "t":      d.strftime("%Y-%m-%dT00:00:00+00:00"),
            "o":      round(o, 2),  "h": round(hi, 2),
            "l":      round(lo, 2), "c": round(c, 2),
            "v":      int(random.uniform(5_000_000, 80_000_000)),
            "source": "demo"
        })
    return bars


# ── PERIOD CONFIG ─────────────────────────────────────────────────────────────
# Maps UI period key → (alpaca_timeframe, bar_count, yahoo_interval, yahoo_period)
PERIOD_CONFIG = {
    # Intraday
    "5m":   ("5Min",  100,  "5m",  "5d"),
    "15m":  ("15Min", 200,  "15m", "5d"),
    "1h":   ("1Hour", 300,  "1h",  "30d"),
    "4h":   ("1Hour", 500,  "1h",  "60d"),
    # Daily
    "1mo":  ("1Day",  22,   "1d",  "1mo"),
    "3mo":  ("1Day",  66,   "1d",  "3mo"),
    "6mo":  ("1Day",  132,  "1d",  "6mo"),
    "1y":   ("1Day",  252,  "1d",  "1y"),
    # Short-view periods for chart selector
    "1D":   ("15Min", 390,  "15m", "5d"),   # today in 15m bars
    "1W":   ("1Hour", 200,  "1h",  "5d"),   # 1 week hourly bars
    # Weekly
    "3y":   ("1Day",  756,  "1d",  "3y"),
    "5y":   ("1Day",  1260, "1d",  "5y"),
    "10y":  ("1Week", 520,  "1wk", "10y"),
    # Aliases
    "3Y":   ("1Day",  756,  "1d",  "3y"),
    "5Y":   ("1Day",  1260, "1d",  "5y"),
    "10Y":  ("1Week", 520,  "1wk", "10y"),
}
# Backward-compat aliases
ALPACA_TF_MAP = {k: (v[0], v[1]) for k, v in PERIOD_CONFIG.items()}
YAHOO_MAP     = {k: (v[2], v[3]) for k, v in PERIOD_CONFIG.items()}


def get_bars(symbol: str, period: str = "1y"):
    """
    Unified bar fetcher. period = any key in PERIOD_CONFIG.
    e.g: "1mo","3mo","6mo","1y","3y","5y","10y","5m","15m","1h","1D","1W"
    Returns list of dicts: {t, o, h, l, c, v, source}
    """
    symbol = symbol.upper()
    cfg    = PERIOD_CONFIG.get(period, PERIOD_CONFIG["1y"])
    alp_tf, alp_limit, yh_interval, yh_period = cfg
    weekly = period in ("1W","10Y","10y")

    # 1. Alpaca Data API (primary)
    bars = alpaca_get_bars(symbol, alp_tf, alp_limit)
    if bars:
        for b in bars:
            b["source"] = "alpaca"
        logger.info(f"✅ Alpaca: {symbol} {period} ({len(bars)} bars)")
        return bars

    # 2. Yahoo Finance (fallback)
    chart = yahoo_get_chart(symbol, yh_interval, yh_period)
    if chart:
        closes    = chart.get("close",[])
        opens     = chart.get("open", [])
        highs     = chart.get("high", [])
        lows      = chart.get("low",  [])
        volumes   = chart.get("volume",[])
        timestamps= chart.get("timestamps",[])
        bars = []
        for i in range(len(closes)):
            if closes[i] is None: continue
            ts = datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat() \
                 if i < len(timestamps) else ""
            bars.append({
                "t": ts,
                "o": opens[i]   if i<len(opens)  and opens[i]   is not None else closes[i],
                "h": highs[i]   if i<len(highs)  and highs[i]   is not None else closes[i],
                "l": lows[i]    if i<len(lows)   and lows[i]    is not None else closes[i],
                "c": closes[i],
                "v": volumes[i] if i<len(volumes) and volumes[i] is not None else 0,
                "source": "yahoo"
            })
        logger.info(f"✅ Yahoo: {symbol} {period} ({len(bars)} bars)")
        return bars

    # 3. Demo fallback (sandbox — all sources blocked)
    logger.warning(f"⚠️ Demo: {symbol} {period} ({alp_limit} bars)")
    # Pass intraday_minutes so demo generates correct timestamps
    intraday_min = 0
    if period == "5m":   intraday_min = 5
    elif period == "15m": intraday_min = 15
    elif period == "1h":  intraday_min = 60
    elif period == "4h":  intraday_min = 240
    return demo_bars(symbol, alp_limit, weekly=weekly, intraday_minutes=intraday_min)


def get_quote_live(symbol: str):
    """Get current price. Alpaca → Yahoo → Demo."""
    symbol = symbol.upper()
    snap   = alpaca_get_snapshot(symbol)
    if snap:
        db = snap.get("dailyBar", {})
        lt = snap.get("latestTrade", {})
        return {
            "symbol": symbol,
            "price":  lt.get("p", db.get("c", 0)),
            "open":   db.get("o", 0),
            "high":   db.get("h", 0),
            "low":    db.get("l", 0),
            "close":  db.get("c", 0),
            "volume": db.get("v", 0),
            "source": "alpaca"
        }

    # Yahoo fallback
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"interval":"1d","range":"1d"},
            headers=_yahoo_headers(), timeout=8)
        if r.ok:
            meta = r.json()["chart"]["result"][0]["meta"]
            return {
                "symbol": symbol,
                "price":  meta.get("regularMarketPrice", 0),
                "open":   meta.get("regularMarketOpen",  0),
                "high":   meta.get("regularMarketDayHigh",0),
                "low":    meta.get("regularMarketDayLow", 0),
                "close":  meta.get("chartPreviousClose",  0),
                "volume": meta.get("regularMarketVolume", 0),
                "source": "yahoo"
            }
    except:
        pass

    # Demo
    seed = {"AAPL":182,"MSFT":415,"NVDA":875,"TSLA":175}.get(symbol, 100)
    return {"symbol":symbol, "price":seed, "open":seed,"high":seed*1.01,
            "low":seed*0.99,"close":seed,"volume":10000000,"source":"demo"}


def get_multi_quotes(symbols):
    """Batch quotes. Returns {symbol: quote_dict}"""
    results = {}
    # Try Alpaca batch first
    snaps = alpaca_get_multi_snapshots(symbols)
    if snaps:
        for sym, snap in snaps.items():
            db = snap.get("dailyBar", {})
            lt = snap.get("latestTrade", {})
            results[sym] = {
                "symbol": sym, "price": lt.get("p", db.get("c",0)),
                "change_pct": snap.get("dailyBar",{}).get("vw",0),
                "volume": db.get("v",0), "source":"alpaca"
            }
        missing = [s for s in symbols if s.upper() not in results]
    else:
        missing = symbols

    for sym in missing:
        q = get_quote_live(sym)
        results[sym.upper()] = q
        time.sleep(0.05)

    return results


def get_options_data(symbol: str):
    """Get options chain. Yahoo Finance → None."""
    chain = yahoo_get_options(symbol)
    if chain:
        return chain
    logger.warning(f"Options data unavailable for {symbol}")
    return None


def get_earnings_date(symbol: str):
    """Get next earnings date from Yahoo."""
    return yahoo_earnings_date(symbol)


# ── Historical volatility ─────────────────────────────────────────────────────
def calc_hist_vol(closes, window: int = 20):
    if len(closes) < window + 1:
        return None
    rets = [math.log(closes[i]/closes[i-1])
            for i in range(1, len(closes))
            if closes[i] and closes[i-1] and closes[i-1] > 0]
    if len(rets) < window:
        return None
    recent = rets[-window:]
    mean   = sum(recent) / len(recent)
    var    = sum((r-mean)**2 for r in recent) / (len(recent)-1)
    return round(math.sqrt(var) * math.sqrt(252) * 100, 1)


def data_source_status():
    """Check which data sources are live. Used by /health endpoint."""
    import os
    from core.config import Config
    has_key = bool(
        Config.ALPACA_API_KEY
        or os.environ.get("APCA_API_KEY_ID", "")
        or os.environ.get("ALPACA_KEY", "")
    )
    alp  = _alpaca_reachable() if has_key else False
    yh   = _yahoo_reachable()
    return {
        "alpaca_data":  "live" if alp else "unavailable",
        "yahoo":        "live" if yh  else "unavailable",
        "fallback":     "demo" if (not alp and not yh) else "not-needed",
        "alpaca_key_set": has_key,
        "note": (
            "Alpaca key not set — set ALPACA_API_KEY + ALPACA_SECRET_KEY env vars for live data"
            if not has_key
            else ("On Railway all sources are live." if (not alp and not yh) else "Live data active")
        )
    }
