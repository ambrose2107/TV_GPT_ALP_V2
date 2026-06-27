"""
research/insider_flow.py — SEC Form 4 insider buys + unusual options flow
"""
import requests, time
from core.market_data import get_options_data, get_quote_live
from core.logger import get_logger

logger = get_logger(__name__)
SEC_H = {"User-Agent":"OptiTradeBot research@optitrade.app"}

DEFAULT_SYMS = [
    "AAPL","MSFT","NVDA","TSLA","AMD","META","GOOGL","JPM","BAC",
    "AMZN","NFLX","CRM","ORCL","INTC","MU","PLTR","SOFI","RBLX",
    "UBER","SNAP","PINS","DKNG","COIN","HOOD","SQ",
]

def get_sec_form4_buys(days=30) -> list:
    from datetime import datetime, timedelta
    end   = datetime.now()
    start = end - timedelta(days=days)
    url   = (f"https://efts.sec.gov/LATEST/search-index?forms=4"
             f"&dateRange=custom&startdt={start.strftime('%Y-%m-%d')}"
             f"&enddt={end.strftime('%Y-%m-%d')}")
    try:
        r    = requests.get(url, headers=SEC_H, timeout=15)
        r.raise_for_status()
        hits = r.json().get("hits",{}).get("hits",[])
        out  = []
        for h in hits[:30]:
            s = h.get("_source",{})
            names = s.get("display_names") or []
            out.append({
                "filer":   names[0] if names else "Unknown",
                "company": s.get("entity_name",""),
                "date":    s.get("period_of_report") or s.get("file_date",""),
            })
        return out
    except Exception as e:
        logger.warning(f"SEC Form4: {e}")
        return []


def get_unusual_options(symbol: str) -> dict:
    base = {"symbol":symbol,"unusual":False}
    chain = get_options_data(symbol)
    if not chain:
        return base
    spot    = chain.get("spot",0)
    u_calls, u_puts = [], []
    for opt in chain.get("calls",[]):
        oi  = opt.get("openInterest",0) or 0
        vol = opt.get("volume",0) or 0
        if oi > 0 and vol > 300:
            ratio = vol/oi
            if ratio > 2:
                u_calls.append({"strike":opt.get("strike"),"expiry":opt.get("expiration",""),
                                 "volume":int(vol),"oi":int(oi),"ratio":round(ratio,1),
                                 "iv":round(opt.get("impliedVolatility",0)*100,1)})
    for opt in chain.get("puts",[]):
        oi  = opt.get("openInterest",0) or 0
        vol = opt.get("volume",0) or 0
        if oi > 0 and vol > 300:
            ratio = vol/oi
            if ratio > 2:
                u_puts.append({"strike":opt.get("strike"),"expiry":opt.get("expiration",""),
                                "volume":int(vol),"oi":int(oi),"ratio":round(ratio,1),
                                "iv":round(opt.get("impliedVolatility",0)*100,1)})
    u_calls.sort(key=lambda x: x["ratio"], reverse=True)
    u_puts.sort(key=lambda x: x["ratio"], reverse=True)
    return {
        "symbol":       symbol,"spot":spot,
        "unusual":      bool(u_calls or u_puts),
        "call_dominant":len(u_calls) >= len(u_puts),
        "unusual_calls":u_calls[:5],
        "unusual_puts": u_puts[:5],
    }


def get_confluence_stocks(symbols=None) -> dict:
    if not symbols:
        symbols = DEFAULT_SYMS
    confluence, all_flows = [], {}
    for sym in symbols:
        flow = get_unusual_options(sym)
        all_flows[sym] = flow
        if flow.get("unusual"):
            q     = get_quote_live(sym)
            confluence.append({
                "symbol":       sym,
                "price":        q.get("price") if q else None,
                "source":       q.get("source","demo") if q else "demo",
                "call_dominant":flow.get("call_dominant"),
                "unusual_calls":flow.get("unusual_calls",[])[:3],
                "unusual_puts": flow.get("unusual_puts",[])[:3],
            })
        time.sleep(0.1)

    insiders = get_sec_form4_buys(30)
    from datetime import datetime
    return {
        "generated":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        "confluence_stocks": confluence,
        "insider_buys":      insiders[:15],
        "scanned":           len(symbols),
    }
