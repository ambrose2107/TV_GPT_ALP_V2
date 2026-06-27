"""
research/sec_filings.py — Institutional 13F tracker via SEC EDGAR
"""
import requests, time
from core.market_data import get_quote_live
from core.logger import get_logger

logger = get_logger(__name__)
SEC_H = {"User-Agent":"OptiTradeBot research@optitrade.app"}

FUNDS = {
    "Berkshire Hathaway":     "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies":"0001037389",
    "Citadel":                "0001423689",
    "Two Sigma":              "0001448942",
}
CUSIP_MAP = {
    "037833100":"AAPL","594918104":"MSFT","023135106":"AMZN","30303M102":"META",
    "02079K305":"GOOGL","88160R101":"TSLA","67066G104":"NVDA","46625H100":"JPM",
    "70450Y103":"PYPL","025816109":"AMD","92826C839":"V","713448108":"PFE",
    "532457108":"LLY","084670702":"BRK-B","478160104":"JNJ","881624209":"TGT",
    "882184100":"TMO","023608102":"AMGN","037735108":"APD","125523100":"CI",
}

def _get_filing(cik, fund_name):
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json",
                         headers=SEC_H, timeout=15)
        r.raise_for_status()
        d = r.json()
        f = d.get("filings",{}).get("recent",{})
        forms   = f.get("form",[])
        accNums = f.get("accessionNumber",[])
        dates   = f.get("filingDate",[])
        for i, form in enumerate(forms):
            if form in ("13F-HR","13F-HR/A"):
                return {"fund":fund_name,"cik":cik,"accession":accNums[i],"date":dates[i],"found":True}
        return {"fund":fund_name,"found":False,"error":"No 13F-HR found"}
    except Exception as e:
        return {"fund":fund_name,"found":False,"error":str(e)}

def _get_holdings(cik, accession):
    import xml.etree.ElementTree as ET
    try:
        ac = accession.replace("-","")
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{ac}/{accession}-index.json"
        r = requests.get(idx_url, headers=SEC_H, timeout=15)
        r.raise_for_status()
        items = r.json().get("directory",{}).get("item",[])
        xml_f = next((f["name"] for f in items
                      if "infotable" in f.get("name","").lower() and f.get("name","").endswith(".xml")), None)
        if not xml_f:
            xml_f = next((f["name"] for f in items if f.get("name","").endswith(".xml")), None)
        if not xml_f: return []
        xr = requests.get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{ac}/{xml_f}",
                          headers=SEC_H, timeout=20)
        xr.raise_for_status()
        root = ET.fromstring(xr.text)
        ns   = {"ns":"http://www.sec.gov/edgar/document/thirteenf/informationtable"}
        holdings = []
        for info in (root.findall(".//ns:infoTable",ns) or root.findall(".//infoTable")):
            def g(tag):
                el = info.find(f"ns:{tag}",ns) or info.find(tag)
                return el.text.strip() if el is not None and el.text else ""
            holdings.append({"name":g("nameOfIssuer"),"cusip":g("cusip"),
                             "value":int(g("value") or 0),"shares":int(g("sshPrnamt") or 0)})
        return sorted(holdings, key=lambda x: x["value"], reverse=True)[:50]
    except Exception as e:
        logger.warning(f"Holdings parse {cik}: {e}")
        return []

def get_institutional_tracker() -> dict:
    results = {}
    for fund, cik in FUNDS.items():
        logger.info(f"Fetching 13F: {fund}")
        filing = _get_filing(cik, fund)
        if filing.get("found"):
            time.sleep(0.5)
            holdings = _get_holdings(cik, filing["accession"])
            # enrich with tickers
            for h in holdings[:20]:
                h["ticker"] = CUSIP_MAP.get(h.get("cusip",""))
            filing["top_holdings"]   = holdings[:15]
            filing["total_holdings"] = len(holdings)
            filing["total_value_bn"] = round(sum(h["value"] for h in holdings)/1e6, 1)
        else:
            filing["top_holdings"] = []; filing["total_holdings"] = 0; filing["total_value_bn"] = 0
        results[fund] = filing
        time.sleep(0.3)
    return results

def analyze_institutional_momentum(all_filings: dict) -> list:
    stock_data = {}
    for fund_name, filing in all_filings.items():
        for h in filing.get("top_holdings",[])[:20]:
            name = h.get("name","")
            if not name: continue
            if name not in stock_data:
                stock_data[name] = {"name":name,"ticker":h.get("ticker"),
                                    "funds":[],"total_value":0,"total_shares":0}
            stock_data[name]["funds"].append(fund_name)
            stock_data[name]["total_value"]  += h.get("value",0)
            stock_data[name]["total_shares"] += h.get("shares",0)
    multi = [v for v in stock_data.values() if len(v["funds"]) >= 2]
    multi.sort(key=lambda x: x["total_value"], reverse=True)
    return multi[:10]
