"""
research/earnings.py — Earnings whiplash: HV vs IV, asymmetric setups
"""
import time
from datetime import datetime, timedelta
from core.market_data import get_bars, get_options_data, get_earnings_date, calc_hist_vol
from core.logger import get_logger

logger = get_logger(__name__)

SP500 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","UNH",
    "XOM","LLY","JNJ","AVGO","PG","MA","HD","CVX","MRK","ABBV",
    "COST","PEP","KO","WMT","BAC","MCD","CRM","NFLX","TMO","CSCO",
    "ABT","ACN","TXN","NKE","ORCL","IBM","QCOM","AMGN","HON","CAT",
    "MU","LRCX","AMD","REGN","SLB","MMC","TJX","EOG","VRTX","KLAC",
    "F","GM","PANW","PYPL","UBER","SNAP","PINS","RBLX","ZM","DKNG",
]

def get_earnings_whiplash(max_stocks=50) -> dict:
    today    = datetime.now().date()
    deadline = today + timedelta(days=14)
    results  = []
    source   = "demo"

    logger.info(f"Scanning {max_stocks} stocks for earnings whiplash...")

    for sym in SP500[:max_stocks]:
        try:
            ed = get_earnings_date(sym)
            if not ed:
                time.sleep(0.05)
                continue
            edate = datetime.strptime(ed[:10], "%Y-%m-%d").date()
            if not (today <= edate <= deadline):
                continue

            bars = get_bars(sym, "1D")
            if not bars or len(bars) < 30:
                continue
            if bars and bars[0].get("source") != "demo":
                source = "live"

            closes = [b["c"] for b in bars if b.get("c")]
            if len(closes) < 30:
                continue

            hv    = calc_hist_vol(closes)
            if not hv or hv < 8:
                continue

            price = closes[-1]
            iv    = None
            try:
                chain = get_options_data(sym)
                if chain and chain.get("calls"):
                    spot  = chain.get("spot") or price
                    calls = chain["calls"]
                    if calls:
                        atm  = min(calls, key=lambda c: abs(c.get("strike",0)-spot))
                        raw  = atm.get("impliedVolatility", 0)
                        iv   = round(raw*100, 1) if raw else None
            except:
                pass

            # Historical moves (big gap-up/down days as proxy)
            hist_moves = []
            for i in range(1, len(closes)):
                if closes[i] and closes[i-1]:
                    ret = abs(closes[i]/closes[i-1]-1)*100
                    if ret > 4:
                        hist_moves.append(round(ret,1))
            avg_move   = round(sum(hist_moves[-4:])/len(hist_moves[-4:]),1) if hist_moves else None
            asymmetric = (iv is not None and hv > 0 and iv < hv*0.78)

            results.append({
                "symbol":       sym,
                "earnings_date":ed[:10],
                "days_to_earn": (edate-today).days,
                "price":        round(price,2),
                "hist_vol":     hv,
                "impl_vol":     iv,
                "iv_hv_ratio":  round(iv/hv,2) if iv else None,
                "avg_move":     avg_move,
                "asymmetric":   asymmetric,
            })
            logger.info(f"  {sym}: HV={hv}% IV={iv}% earns={ed}")
            time.sleep(0.15)

        except Exception as e:
            logger.debug(f"Skip {sym}: {e}")

    results.sort(key=lambda x: x["hist_vol"], reverse=True)
    return {
        "scan_date":         str(today),
        "stocks_scanned":    max_stocks,
        "with_earnings":     len(results),
        "data_source":       source,
        "top10":             results[:10],
        "asymmetric_setups": [r for r in results[:10] if r["asymmetric"]][:3],
    }
