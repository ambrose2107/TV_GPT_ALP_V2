"""
core/data_engine.py
Free data engine using yfinance + SEC EDGAR + Stooq.
All sources work on Railway. Sandbox has network restrictions (expected).
"""
import requests, json, time, os
from datetime import datetime, timedelta
from core.logger import get_logger

logger = get_logger(__name__)

# ── Session with proper headers (bypasses Yahoo rate limits) ─────────────────
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept": "application/json,text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com",
    })
    return s

_session = make_session()

# ── Yahoo Finance v8 (chart data, free, no auth) ────────────────────────────
def get_chart(symbol: str, interval: str = "1d", period: str = "6mo") -> dict | None:
    """Fetch OHLCV from Yahoo Finance v8 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
    params = {"interval": interval, "range": period, "includePrePost": "false"}
    try:
        r = _session.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        result = data["chart"]["result"][0]
        ts     = result["timestamp"]
        q      = result["indicators"]["quote"][0]
        return {
            "timestamps": ts,
            "open":   q.get("open",   []),
            "high":   q.get("high",   []),
            "low":    q.get("low",    []),
            "close":  q.get("close",  []),
            "volume": q.get("volume", []),
        }
    except Exception as e:
        logger.warning(f"get_chart {symbol} {interval}: {e}")
        return None


def get_quote(symbol: str) -> dict | None:
    """Get current quote."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
    try:
        r = _session.get(url, params={"interval":"1d","range":"1d"}, timeout=10)
        r.raise_for_status()
        meta = r.json()["chart"]["result"][0]["meta"]
        return {
            "symbol":  symbol.upper(),
            "price":   meta.get("regularMarketPrice", 0),
            "prev":    meta.get("chartPreviousClose",  0),
            "change":  meta.get("regularMarketPrice",0) - meta.get("chartPreviousClose",0),
            "pct":     (meta.get("regularMarketPrice",0)/max(meta.get("chartPreviousClose",1),0.01)-1)*100,
            "volume":  meta.get("regularMarketVolume", 0),
            "currency":meta.get("currency","USD"),
        }
    except Exception as e:
        logger.warning(f"get_quote {symbol}: {e}")
        return None


def get_quotes_batch(symbols: list) -> dict:
    """Get quotes for multiple symbols."""
    results = {}
    for sym in symbols:
        q = get_quote(sym)
        if q:
            results[sym] = q
        time.sleep(0.1)
    return results


# ── Options data ─────────────────────────────────────────────────────────────
def get_options_chain(symbol: str) -> dict | None:
    """Get near-term options chain from Yahoo."""
    url = f"https://query1.finance.yahoo.com/v7/finance/options/{symbol.upper()}"
    try:
        r = _session.get(url, timeout=12)
        r.raise_for_status()
        data   = r.json()
        result = data["optionChain"]["result"][0]
        calls  = result.get("options", [{}])[0].get("calls", [])
        puts   = result.get("options", [{}])[0].get("puts",  [])
        spot   = result.get("quote", {}).get("regularMarketPrice", 0)
        exp    = result.get("expirationDates", [])
        return {
            "symbol": symbol.upper(),
            "spot":   spot,
            "expiries": exp,
            "calls":  calls,
            "puts":   puts,
        }
    except Exception as e:
        logger.warning(f"get_options {symbol}: {e}")
        return None


def get_implied_vol(symbol: str) -> float | None:
    """Get ATM implied volatility from nearest expiry."""
    chain = get_options_chain(symbol)
    if not chain or not chain["calls"]:
        return None
    spot  = chain["spot"]
    calls = chain["calls"]
    # Find ATM call
    atm = min(calls, key=lambda c: abs(c.get("strike",0) - spot))
    iv  = atm.get("impliedVolatility", None)
    return round(iv * 100, 1) if iv else None


# ── Sector ETF data ──────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology":       "XLK",
    "Healthcare":       "XLV",
    "Financials":       "XLF",
    "Consumer Discr.":  "XLY",
    "Industrials":      "XLI",
    "Communication":    "XLC",
    "Consumer Staples": "XLP",
    "Energy":           "XLE",
    "Materials":        "XLB",
    "Utilities":        "XLU",
    "Real Estate":      "XLRE",
}

TOP_ETFS_PER_SECTOR = {
    "Technology":       ["QQQ","SOXX","IGV","VGT","ARKK"],
    "Healthcare":       ["IBB","XBI","IHI","ARKG","IHF"],
    "Financials":       ["KBE","KRE","IAI","KBWB","IYF"],
    "Consumer Discr.":  ["XRT","RTH","FDIS","IBUY","VCR"],
    "Industrials":      ["ITA","XAR","JETS","PAVE","VIS"],
    "Communication":    ["FCOM","VOX","IYZ","SOCL","ESPO"],
    "Consumer Staples": ["VDC","KXI","FSTA","IYK","PBJ"],
    "Energy":           ["OIH","XOP","FCG","AMLP","IEZ"],
    "Materials":        ["GDX","GDXJ","MOO","MXI","URNM"],
    "Utilities":        ["VPU","FXU","IDU","FUTY","RYU"],
    "Real Estate":      ["VNQ","IYR","SCHH","RWR","REZ"],
}


def get_sector_returns(days: int = 30) -> dict:
    """Get N-day returns for all sector ETFs."""
    period = "3mo" if days <= 30 else "6mo"
    results = {}
    for name, etf in SECTOR_ETFS.items():
        chart = get_chart(etf, "1d", period)
        if not chart:
            results[name] = {"etf": etf, "return": None, "error": "no data"}
            continue
        closes = [c for c in chart["close"] if c is not None]
        if len(closes) < days:
            results[name] = {"etf": etf, "return": None, "error": "insufficient data"}
            continue
        ret = (closes[-1] / closes[-min(days, len(closes))] - 1) * 100
        results[name] = {"etf": etf, "return": round(ret, 2), "closes": closes[-5:]}
    return results


def get_etf_money_flow(ticker: str, days_short: int = 10, days_long: int = 30) -> dict:
    """Calculate money flow (price * volume) for an ETF."""
    chart = get_chart(ticker, "1d", "3mo")
    if not chart:
        return {"ticker": ticker, "error": "no data"}
    closes  = [c for c in chart["close"]  if c is not None]
    volumes = [v for v in chart["volume"] if v is not None]
    n = min(len(closes), len(volumes))
    if n < 5:
        return {"ticker": ticker, "error": "insufficient data"}
    ret_s = (closes[-1]/closes[-min(days_short,n)] - 1)*100 if n >= days_short else None
    ret_l = (closes[-1]/closes[-min(days_long, n)] - 1)*100 if n >= days_long  else None
    avg_vol = sum(volumes[-min(days_short,n):]) / min(days_short, n)
    price   = closes[-1]
    return {
        "ticker":    ticker,
        "price":     round(price, 2),
        "return_10": round(ret_s, 2) if ret_s is not None else None,
        "return_30": round(ret_l, 2) if ret_l is not None else None,
        "avg_volume":int(avg_vol),
        "money_flow":round(price * avg_vol / 1e6, 1),  # $M
    }


# ── SEC EDGAR (free, no auth) ─────────────────────────────────────────────────
SEC_HEADERS = {"User-Agent": "TradingBot research@tradingbot.app"}

FUNDS = {
    "Berkshire Hathaway":    "0001067983",
    "Bridgewater Associates":"0001350694",
    "Renaissance Technologies":"0001037389",
    "Citadel":               "0001423689",
    "Two Sigma":             "0001448942",
}

def get_13f_filing(cik: str, fund_name: str) -> dict:
    """Fetch latest 13F-HR filing metadata."""
    try:
        url  = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        r    = requests.get(url, headers=SEC_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        accNums = filings.get("accessionNumber", [])
        dates   = filings.get("filingDate", [])
        for i, form in enumerate(forms):
            if form in ("13F-HR", "13F-HR/A"):
                return {
                    "fund":      fund_name,
                    "cik":       cik,
                    "accession": accNums[i],
                    "date":      dates[i],
                    "found":     True,
                }
        return {"fund": fund_name, "found": False, "error": "No 13F-HR found"}
    except Exception as e:
        return {"fund": fund_name, "found": False, "error": str(e)}


def get_13f_holdings(cik: str, accession: str) -> list:
    """Parse holdings from 13F filing XML."""
    import xml.etree.ElementTree as ET
    try:
        acc_clean = accession.replace("-", "")
        idx_url   = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.json"
        r         = requests.get(idx_url, headers=SEC_HEADERS, timeout=15)
        r.raise_for_status()
        idx       = r.json()
        items     = idx.get("directory", {}).get("item", [])
        xml_file  = None
        for f in items:
            name = f.get("name", "").lower()
            if "infotable" in name and name.endswith(".xml"):
                xml_file = f["name"]
                break
        if not xml_file:
            for f in items:
                if f.get("name","").endswith(".xml") and "primary" not in f.get("name","").lower():
                    xml_file = f["name"]
                    break
        if not xml_file:
            return []
        xml_url  = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{xml_file}"
        xr       = requests.get(xml_url, headers=SEC_HEADERS, timeout=20)
        xr.raise_for_status()
        root     = ET.fromstring(xr.text)
        ns       = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}
        holdings = []
        for info in (root.findall(".//ns:infoTable", ns) or root.findall(".//infoTable")):
            def _t(tag):
                el = info.find(f"ns:{tag}", ns) or info.find(tag)
                return el.text.strip() if el is not None and el.text else ""
            holdings.append({
                "name":   _t("nameOfIssuer"),
                "cusip":  _t("cusip"),
                "value":  int(_t("value") or 0),
                "shares": int(_t("sshPrnamt") or 0),
                "type":   _t("putCall") or "Shares",
            })
        return sorted(holdings, key=lambda x: x["value"], reverse=True)[:50]
    except Exception as e:
        logger.warning(f"Holdings parse error {cik}: {e}")
        return []


# ── Earnings calendar ─────────────────────────────────────────────────────────
def get_earnings_calendar(symbol: str) -> str | None:
    """Get next earnings date from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    try:
        r = _session.get(url, params={"modules": "calendarEvents"}, timeout=10)
        r.raise_for_status()
        events = r.json()["quoteSummary"]["result"][0]["calendarEvents"]
        dates  = events.get("earnings", {}).get("earningsDate", [])
        if dates:
            ts = dates[0].get("raw", 0)
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except:
        pass
    return None


# ── Historical volatility ─────────────────────────────────────────────────────
def calc_hist_vol(closes: list, window: int = 20) -> float | None:
    """Annualised historical volatility from close prices."""
    import math
    if len(closes) < window + 1:
        return None
    rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes)) if closes[i] and closes[i-1]]
    if len(rets) < window:
        return None
    recent = rets[-window:]
    mean   = sum(recent) / len(recent)
    var    = sum((r - mean)**2 for r in recent) / (len(recent) - 1)
    return round(math.sqrt(var) * math.sqrt(252) * 100, 1)
