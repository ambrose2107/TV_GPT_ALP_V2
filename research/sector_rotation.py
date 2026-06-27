"""
research/sector_rotation.py — Real sector rotation with volume/money flow analysis
"""
import time
from datetime import datetime
from core.market_data import get_bars
from core.logger import get_logger

logger = get_logger(__name__)

SECTOR_ETFS = {
    "Technology":       "XLK",  "Healthcare":       "XLV",
    "Financials":       "XLF",  "Consumer Discr.":  "XLY",
    "Industrials":      "XLI",  "Communication":    "XLC",
    "Consumer Staples": "XLP",  "Energy":           "XLE",
    "Materials":        "XLB",  "Utilities":        "XLU",
    "Real Estate":      "XLRE",
}

TOP_ETFS = {
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

def _pct(bars, n):
    """% return over last n bars."""
    closes = [b["c"] for b in bars if b.get("c")]
    if len(closes) < n+1:
        return None
    return round((closes[-1]/closes[-n]-1)*100, 2)

def _avg_volume(bars, n=10):
    vols = [b.get("v",0) for b in bars[-n:] if b.get("v")]
    return int(sum(vols)/len(vols)) if vols else 0

def _money_flow(bars, n=10):
    """Average daily money flow = price × volume ($M)"""
    flows = []
    for b in bars[-n:]:
        c = b.get("c"); v = b.get("v")
        if c and v:
            flows.append(c * v)
    return round(sum(flows)/len(flows)/1e6, 1) if flows else 0

def _volume_trend(bars):
    """Compare recent 10-day avg volume vs prior 10-day avg."""
    vols = [b.get("v",0) for b in bars if b.get("v")]
    if len(vols) < 20:
        return None, None
    recent = sum(vols[-10:])/10
    prior  = sum(vols[-20:-10])/10
    if prior <= 0:
        return None, None
    chg = round((recent/prior - 1)*100, 1)
    trend = "Rising ↑" if chg > 5 else "Falling ↓" if chg < -5 else "Stable →"
    return trend, chg

def get_sector_rotation() -> dict:
    logger.info("Running sector rotation analysis...")
    sectors = {}
    source  = "demo"

    # ── Fetch all sector ETF bars (1 year of daily data) ─────────────────────
    for name, etf in SECTOR_ETFS.items():
        bars = get_bars(etf, "1y")          # ← FIXED: was "1D"
        if not bars:
            sectors[name] = {"etf": etf, "return_30d": None, "return_1y_ago": None, "error": "no data"}
            continue

        if bars[0].get("source") != "demo":
            source = "live"

        ret_5d  = _pct(bars, 5)
        ret_22d = _pct(bars, 22)   # ~1 month
        ret_66d = _pct(bars, 66)   # ~3 months

        # 1-year-ago comparison: use bars from ~252 days back if available
        ret_prior = None
        if len(bars) >= 282:
            prior_slice = [b for b in bars[-282:-252] if b.get("c")]
            if len(prior_slice) >= 5:
                closes = [b["c"] for b in prior_slice]
                ret_prior = round((closes[-1]/closes[0]-1)*100, 2)

        # Volume metrics
        vol_trend, vol_chg = _volume_trend(bars)
        avg_vol   = _avg_volume(bars, 10)
        mf        = _money_flow(bars, 10)
        current_price = bars[-1].get("c") if bars else None

        # Rotation detection
        rotating     = False
        flip         = False
        if ret_22d is not None and ret_prior is not None:
            rotating = (ret_prior < 0 < ret_22d) or (ret_prior > 0 > ret_22d) or (ret_22d > ret_prior + 3)
            flip     = (ret_prior < 0 < ret_22d)

        # Signal: combine price + volume
        signal = "Neutral"
        if ret_22d is not None and vol_trend:
            if ret_22d > 1 and "Rising" in vol_trend:
                signal = "Strong Buy ↑↑"
            elif ret_22d > 0.5:
                signal = "Buy ↑"
            elif ret_22d < -1 and "Rising" in vol_trend:
                signal = "Distribution ↓↓"
            elif ret_22d < -0.5:
                signal = "Sell ↓"

        sectors[name] = {
            "etf":           etf,
            "price":         round(current_price,2) if current_price else None,
            "return_5d":     ret_5d,
            "return_30d":    ret_22d,
            "return_90d":    ret_66d,
            "return_1y_ago": ret_prior,
            "avg_volume":    avg_vol,
            "money_flow_m":  mf,
            "vol_trend":     vol_trend,
            "vol_chg_pct":   vol_chg,
            "rotating":      rotating,
            "flip":          flip,
            "signal":        signal,
        }
        time.sleep(0.05)

    # ── ETF money flows for rotating/strong sectors ───────────────────────────
    rotating  = {k:v for k,v in sectors.items() if v.get("rotating")}
    top_secs  = sorted(
        [(k,v) for k,v in sectors.items() if v.get("return_30d") is not None],
        key=lambda x: x[1]["return_30d"], reverse=True
    )[:3]
    focus_secs = list(rotating.keys()) + [k for k,v in top_secs if k not in rotating]

    etf_flows = {}
    for sec_name in focus_secs[:4]:
        etfs  = TOP_ETFS.get(sec_name, [])
        flows = []
        for etf in etfs:
            bars = get_bars(etf, "3mo")      # ← FIXED: was "1D"
            if not bars:
                continue
            closes = [b["c"] for b in bars if b.get("c")]
            r10    = _pct(bars, 10)
            r22    = _pct(bars, 22)
            mf     = _money_flow(bars, 10)
            vt, vc = _volume_trend(bars)
            flows.append({
                "ticker":     etf,
                "price":      round(closes[-1],2) if closes else 0,
                "return_10":  r10,
                "return_30":  r22,
                "money_flow": mf,
                "vol_trend":  vt,
                "vol_chg":    vc,
            })
            time.sleep(0.05)
        flows.sort(key=lambda x: x.get("money_flow",0) or 0, reverse=True)
        etf_flows[sec_name] = flows

    ranked = sorted(
        [(k, v["return_30d"]) for k,v in sectors.items() if v.get("return_30d") is not None],
        key=lambda x: x[1], reverse=True
    )

    return {
        "generated":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_source":  source,
        "sectors":      sectors,
        "rotating":     rotating,
        "etf_flows":    etf_flows,
        "ranked":       ranked,
    }
