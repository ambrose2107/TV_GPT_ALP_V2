"""Portfolio research enrichment: SEC company filings and sector-relative strength.

SEC documents are downloaded in a bounded background job and persisted in the
existing scheduler_settings table. No broker orders are ever submitted here.
"""
from datetime import datetime, timedelta, timezone
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

from core.config import Config
from core.database import get_setting, set_setting
from core.logger import get_logger

logger = get_logger(__name__)
SEC_CACHE_KEY = "portfolio_sec_research_v1"
SEC_CACHE_TTL_HOURS = 12
SEC_HEADERS = {
    "User-Agent": os.environ.get(
        "SEC_USER_AGENT", "OptiTradeBot research contact research@optitrade.app"
    ),
    "Accept-Encoding": "gzip, deflate",
}
_SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Industrials": "XLI",
    "Communication Services": "XLC", "Consumer Staples": "XLP",
    "Energy": "XLE", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Semiconductors": "SOXX",
    "Software": "IGV", "Biotechnology": "XBI",
}
_TICKER_MAP = {"at": 0.0, "data": {}}


def _alpaca_credentials():
    key = (getattr(Config, "ALPACA_API_KEY", "") or os.environ.get("APCA_API_KEY_ID", "")
           or os.environ.get("ALPACA_KEY", ""))
    secret = (getattr(Config, "ALPACA_SECRET_KEY", "") or os.environ.get("APCA_API_SECRET_KEY", "")
              or os.environ.get("ALPACA_SECRET", ""))
    return key, secret


def _load_ticker_map():
    if _TICKER_MAP["data"] and time.time() - _TICKER_MAP["at"] < 86400:
        return _TICKER_MAP["data"]
    response = requests.get("https://www.sec.gov/files/company_tickers.json",
                            headers=SEC_HEADERS, timeout=15)
    response.raise_for_status()
    raw = response.json()
    mapping = {}
    for row in raw.values():
        ticker = str(row.get("ticker") or "").upper().strip()
        cik = str(row.get("cik_str") or "").strip()
        if ticker and cik:
            mapping[ticker] = str(int(cik)).zfill(10)
    _TICKER_MAP.update({"at": time.time(), "data": mapping})
    return mapping


def _clean_filing_text(html, limit=1800):
    soup = BeautifulSoup(html or "", "lxml")
    for node in soup(["script", "style", "ix:header", "header", "footer"]):
        node.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:limit]


def _filings_for_ticker(ticker, cik, now):
    url = "https://data.sec.gov/submissions/CIK%s.json" % cik
    response = requests.get(url, headers=SEC_HEADERS, timeout=15)
    response.raise_for_status()
    recent = (response.json().get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []
    cutoff_periodic = (now - timedelta(days=210)).date().isoformat()
    cutoff_current = (now - timedelta(days=60)).date().isoformat()
    candidates = []
    for i, form in enumerate(forms):
        if i >= len(dates) or i >= len(accessions) or i >= len(docs):
            continue
        filing_date = dates[i]
        if form in ("10-K", "10-Q") and filing_date >= cutoff_periodic:
            candidates.append((filing_date, 0 if form == "10-Q" else 1, form, accessions[i], docs[i]))
        elif form == "8-K" and filing_date >= cutoff_current:
            candidates.append((filing_date, 2, form, accessions[i], docs[i]))
    # At most one recent periodic report and one recent 8-K per company.
    selected = []
    periodic = [x for x in candidates if x[2] in ("10-K", "10-Q")]
    current = [x for x in candidates if x[2] == "8-K"]
    if periodic:
        selected.append(max(periodic, key=lambda x: x[0]))
    if current:
        selected.append(max(current, key=lambda x: x[0]))
    output = []
    for filing_date, _, form, accession, primary_doc in selected:
        accession_path = accession.replace("-", "")
        doc_url = "https://www.sec.gov/Archives/edgar/data/%s/%s/%s" % (
            str(int(cik)), accession_path, primary_doc
        )
        try:
            doc_response = requests.get(doc_url, headers=SEC_HEADERS, timeout=20)
            doc_response.raise_for_status()
            output.append({
                "symbol": ticker, "form": form, "filing_date": filing_date,
                "accession": accession, "url": doc_url,
                "text_excerpt": _clean_filing_text(doc_response.text),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.warning("SEC document download failed for %s %s: %s", ticker, form, exc)
        time.sleep(0.12)  # SEC fair-access pacing
    return output


def load_sec_research():
    raw = get_setting(SEC_CACHE_KEY)
    if not raw:
        return {"status": "not_started", "filings": [], "analysis": None,
                "message": "SEC filing download is scheduled; no completed run is stored yet."}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"status": "unavailable", "filings": [], "analysis": None,
                "message": "Stored SEC research could not be decoded."}


def refresh_sec_research():
    """Download recent company 10-K/10-Q/8-Ks for live holdings and analyse them once."""
    key, secret = _alpaca_credentials()
    if not key or not secret:
        result = {"status": "unavailable", "filings": [], "analysis": None,
                  "message": "Alpaca credentials are unavailable; SEC holdings list could not be built.",
                  "as_of": datetime.now(timezone.utc).isoformat()}
        set_setting(SEC_CACHE_KEY, json.dumps(result))
        return result
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    base = (Config.ALPACA_BASE_URL or "https://paper-api.alpaca.markets").rstrip("/")
    positions_response = requests.get(base + "/v2/positions", headers=headers, timeout=12)
    positions_response.raise_for_status()
    symbols = list(dict.fromkeys(
        str(p.get("symbol") or "").upper() for p in positions_response.json()
        if p.get("symbol")
    ))[:24]
    ticker_map = _load_ticker_map()
    now = datetime.now(timezone.utc)
    filings, errors = [], []
    for symbol in symbols:
        cik = ticker_map.get(symbol)
        if not cik:
            errors.append(symbol + ": no SEC ticker/CIK mapping")
            continue
        try:
            filings.extend(_filings_for_ticker(symbol, cik, now))
        except Exception as exc:
            logger.warning("SEC submissions failed for %s: %s", symbol, exc)
            errors.append(symbol + ": SEC submissions unavailable")
        time.sleep(0.12)

    news = {"configured": False, "message": "MarketAux news was not fetched."}
    try:
        from research.early_signal_routes import _marketaux_news_batch
        news = _marketaux_news_batch(symbols)
    except Exception as exc:
        logger.warning("Automatic portfolio news refresh unavailable: %s", exc)
        news = {"configured": False, "error": "News refresh unavailable."}
    try:
        sector_rotation = sector_rotation_snapshot()
    except Exception as exc:
        logger.warning("Automatic sector rotation refresh unavailable: %s", exc)
        sector_rotation = {"available": False, "message": "Sector rotation data unavailable."}

    analysis = None
    if filings:
        try:
            from mirrorfish.engine import chat, get_provider_status
            providers = get_provider_status()
            if any(v.get("configured") for v in providers.values()):
                evidence = [{
                    "ticker": f["symbol"], "form": f["form"],
                    "filing_date": f["filing_date"], "url": f["url"],
                    "excerpt": f["text_excerpt"][:1300],
                } for f in filings[:30]]
                prompt = (
                    "You are a forensic public-equity analyst. Analyse only the SEC filing "
                    "evidence below. For each ticker identify material positives, negatives, "
                    "guidance/cash-flow/debt/dilution/customer or litigation risks where visible, "
                    "and what to verify next. Separate facts stated in filings from inference. "
                    "Do not invent figures or claim an excerpt covers the whole filing. "
                    "Flag missing/ambiguous evidence. Finish with a concise cross-holding risk "
                    "summary. Include filing dates and ticker labels.\nEVIDENCE:\n" +
                    json.dumps(evidence, separators=(",", ":"))[:19000]
                )
                preferred = "huggingface" if providers.get("huggingface", {}).get("configured") else None
                analysis = str(chat(prompt, {"module": "Automatic SEC Filing Review"}, provider_name=preferred, max_tokens=1500))[:10000]
        except Exception as exc:
            logger.warning("Automatic SEC AI analysis unavailable: %s", exc)
            errors.append("SEC documents downloaded; AI filing analysis unavailable")
    result = {
        "status": "complete" if filings else "partial",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "symbols_checked": symbols, "filing_count": len(filings),
        "filings": filings, "analysis": analysis, "news": news,
        "sector_rotation": sector_rotation, "errors": errors,
        "source": "SEC EDGAR company submissions and filing primary documents",
        "next_refresh_hours": SEC_CACHE_TTL_HOURS,
    }
    set_setting(SEC_CACHE_KEY, json.dumps(result))
    return result


def sector_rotation_snapshot():
    """One batched Alpaca request for 5D/1M/3M sector and theme relative strength."""
    key, secret = _alpaca_credentials()
    if not key or not secret:
        return {"available": False, "message": "Alpaca market-data credentials unavailable."}
    symbols = list(_SECTOR_ETFS.values()) + ["SPY"]
    start = (datetime.now(timezone.utc) - timedelta(days=125)).date().isoformat()
    response = requests.get(
        "https://data.alpaca.markets/v2/stocks/bars",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        params={"symbols": ",".join(symbols), "timeframe": "1Day", "start": start,
                "limit": 10000, "adjustment": "all", "feed": "iex", "sort": "asc"},
        timeout=18,
    )
    response.raise_for_status()
    bars = response.json().get("bars") or {}
    def ret(rows, n):
        closes = [float(x.get("c")) for x in rows if x.get("c") and float(x["c"]) > 0]
        return round((closes[-1] / closes[-n-1] - 1) * 100, 2) if len(closes) > n else None
    spy = bars.get("SPY") or []
    rows = []
    for sector, etf in _SECTOR_ETFS.items():
        series = bars.get(etf) or []
        if not series:
            continue
        metrics = {"sector": sector, "etf": etf,
                   "return_5d_pct": ret(series, 5),
                   "return_1m_pct": ret(series, 21),
                   "return_3m_pct": ret(series, 63)}
        for label, n in (("5d", 5), ("1m", 21), ("3m", 63)):
            own, bench = metrics["return_" + label + "_pct"], ret(spy, n)
            metrics["relative_" + label + "_pct"] = round(own - bench, 2) if own is not None and bench is not None else None
        rows.append(metrics)
    rows.sort(key=lambda x: x["relative_1m_pct"] if x["relative_1m_pct"] is not None else -999, reverse=True)
    return {"available": bool(rows), "as_of": datetime.now(timezone.utc).isoformat(),
            "benchmark": "SPY", "sectors": rows,
            "methodology": "Sector ETF total-return proxy vs SPY; 5/21/63 trading-session relative strength. Not fund flows."}
