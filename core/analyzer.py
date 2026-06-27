"""
core/analyzer.py — Analyzer Pro: RSI, MACD, ADX, Bollinger, EMA50, VWAP
Uses unified market_data engine (Alpaca → Yahoo → Demo)
"""
import math
from core.logger import get_logger
from core.market_data import get_bars

logger = get_logger(__name__)

def _clean(lst): return [x for x in (lst or []) if x is not None]

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(closes, period=14):
    c = _clean(closes)
    if len(c) < period+1: return None
    gains = [max(c[i]-c[i-1],0) for i in range(1,len(c))]
    losses= [max(c[i-1]-c[i],0) for i in range(1,len(c))]
    ag = sum(gains[:period])/period
    al = sum(losses[:period])/period
    for i in range(period, len(gains)):
        ag = (ag*(period-1)+gains[i])/period
        al = (al*(period-1)+losses[i])/period
    return round(100-(100/(1+ag/al)) if al else 100, 2)

def rsi_label(v):
    if v is None: return "N/A", 6
    if v >= 80:   return "Extreme OB",   2
    if v >= 70:   return "Overbought",   3
    if v >= 60:   return "Bull Momentum",4
    if v >= 55:   return "Slightly Bull",5
    if v >= 45:   return "Neutral",      6
    if v >= 40:   return "Slightly Bear",7
    if v >= 30:   return "Oversold",     9
    return "Extreme OS", 10

def _ema_series(data, span):
    if not data: return []
    k, s = 2/(span+1), data[0]
    result = [s]
    for p in data[1:]:
        s = p*k + s*(1-k)
        result.append(s)
    return result

def calc_ema(closes, span):
    c = _clean(closes)
    if len(c) < span: return None
    series = _ema_series(c, span)
    return round(series[-1], 4) if series else None

def calc_macd(closes):
    c = _clean(closes)
    if len(c) < 26: return None, None, None
    e12 = _ema_series(c, 12)
    e26 = _ema_series(c, 26)
    macd = [a-b for a,b in zip(e12, e26)]
    sig  = _ema_series(macd, 9)
    return round(macd[-1],4), round(sig[-1],4), round(macd[-1]-sig[-1],4)

def macd_label(m, s, h):
    if m is None: return "N/A", 6
    if m > 0 and h > 0 and m > s: return "Strong Bull", 1
    if m > 0 and h > 0:           return "Bull",        3
    if m > 0 and h < 0:           return "Weakening",   5
    if m < 0 and h > 0:           return "Recovering",  7
    if m < 0 and h < 0 and m < s: return "Strong Bear", 10
    if m < 0 and h < 0:           return "Bear",        8
    return "Neutral", 6

def calc_bb(closes, period=20, mult=2):
    c = _clean(closes)
    if len(c) < period: return None,None,None,None
    w   = c[-period:]
    mid = sum(w)/period
    sig = math.sqrt(sum((x-mid)**2 for x in w)/period)
    upper = mid + mult*sig
    lower = mid - mult*sig
    pb    = (c[-1]-lower)/(upper-lower) if upper != lower else 0.5
    return round(upper,4), round(mid,4), round(lower,4), round(pb,4)

def bb_label(pb):
    if pb is None: return "N/A",6
    if pb >= 1.1:  return "Extreme Upper",2
    if pb >= 0.85: return "Upper Break",  3
    if pb >= 0.6:  return "Above Mid",    4
    if pb >= 0.4:  return "In Bands",     6
    if pb >= 0.2:  return "Below Mid",    7
    if pb >= 0.0:  return "Lower Break",  9
    return "Extreme Lower", 10

def calc_adx(highs, lows, closes, period=14):
    h = _clean(highs); l = _clean(lows); c = _clean(closes)
    n = min(len(h),len(l),len(c))
    if n < period+2: return None, None, None
    h,l,c = h[-n:],l[-n:],c[-n:]
    trs,dmp,dmn = [],[],[]
    for i in range(1,n):
        tr  = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        dp  = max(h[i]-h[i-1],0) if (h[i]-h[i-1])>(l[i-1]-l[i]) else 0
        dn  = max(l[i-1]-l[i],0) if (l[i-1]-l[i])>(h[i]-h[i-1]) else 0
        trs.append(tr); dmp.append(dp); dmn.append(dn)
    def smooth(d,p):
        s=[sum(d[:p])]
        for x in d[p:]: s.append(s[-1]-s[-1]/p+x)
        return s
    atr=smooth(trs,period); dp_s=smooth(dmp,period); dn_s=smooth(dmn,period)
    dx_vals=[]
    for i in range(len(atr)):
        if not atr[i]: continue
        dip=100*dp_s[i]/atr[i]; din=100*dn_s[i]/atr[i]
        if dip+din: dx_vals.append(100*abs(dip-din)/(dip+din))
    if not dx_vals: return None,None,None
    adx = sum(dx_vals[-period:])/min(period,len(dx_vals))
    last_atr = atr[-1]
    dip_last = 100*dp_s[-1]/last_atr if last_atr else 0
    din_last = 100*dn_s[-1]/last_atr if last_atr else 0
    return round(adx,1), round(dip_last,1), round(din_last,1)

def adx_label(adx, dip, din):
    if adx is None: return "N/A",6
    if adx < 20:                    return "No Trend",    6
    up = (dip or 0) > (din or 0)
    if adx >= 40 and up:            return "Very Strong↑",2
    if adx >= 30 and up:            return "Strong Up",   3
    if adx >= 20 and up:            return "Trending↑",   4
    if adx >= 40 and not up:        return "Very Strong↓",9
    if adx >= 30 and not up:        return "Strong Down",  8
    return "Trending↓", 7

def calc_vwap(highs, lows, closes, volumes):
    h=_clean(highs);l=_clean(lows);c=_clean(closes);v=_clean(volumes)
    n=min(len(h),len(l),len(c),len(v))
    if n < 2: return None,None
    ctv=cv=0
    for i in range(n):
        typ=(h[i]+l[i]+c[i])/3
        ctv+=typ*v[i]; cv+=v[i]
    vwap = ctv/cv if cv else c[-1]
    pct  = (c[-1]-vwap)/vwap*100 if vwap else 0
    return round(vwap,4), round(pct,2)

def vwap_label(pct):
    if pct is None: return "N/A",6
    if pct >= 3:   return "Far Over",   2
    if pct >= 1:   return "Over VWAP",  4
    if pct >= 0.2: return "Slightly↑",  5
    if pct >= -0.2:return "Near VWAP",  6
    if pct >= -1:  return "Slightly↓",  7
    if pct >= -3:  return "Under VWAP", 8
    return "Far Under", 9

def ema50_label(pct):
    if pct is None: return "N/A",6
    if pct >= 8:   return "Super Trend↑",1
    if pct >= 4:   return "Strong↑",     2
    if pct >= 1:   return "Uptrend",     4
    if pct >= -1:  return "Consolidate", 6
    if pct >= -4:  return "Downtrend",   7
    if pct >= -8:  return "Strong↓",     9
    return "Super Trend↓", 10

def score_to_signal(avg):
    if avg <= 2:   return "Strong Bull 🟢","bull2"
    if avg <= 4:   return "Bull 🟢",       "bull1"
    if avg <= 5.5: return "Slightly Bull 🟡","bln"
    if avg <= 6.5: return "Neutral ⚪",    "neut"
    if avg <= 8:   return "Slightly Bear 🟠","brn"
    if avg <= 9:   return "Bear 🔴",       "bear1"
    return "Strong Bear 🔴","bear2"

# ── Single timeframe analysis ─────────────────────────────────────────────────
def analyze_one_tf(bars: list) -> dict:
    if not bars or len(bars) < 20:
        return {"error": f"Insufficient data ({len(bars) if bars else 0} bars)"}

    closes  = [b["c"] for b in bars if b.get("c")]
    highs   = [b["h"] for b in bars if b.get("h")]
    lows    = [b["l"] for b in bars if b.get("l")]
    volumes = [b.get("v",0) for b in bars]

    if len(closes) < 20:
        return {"error": "Too few valid bars"}

    row    = {}
    scores = []

    # RSI
    rv = calc_rsi(closes)
    rl, rs = rsi_label(rv)
    row["rsi"] = {"v": rv, "l": rl}; scores.append(rs)

    # MACD
    m, s, hst = calc_macd(closes)
    ml, ms = macd_label(m, s, hst)
    row["macd"] = {"v": round(m,4) if m else None, "l": ml}; scores.append(ms)

    # ADX
    av, dip, din = calc_adx(highs, lows, closes)
    al, as_ = adx_label(av, dip, din)
    row["adx"] = {"v": av, "l": al}; scores.append(as_)

    # Bollinger
    bu, bm, bl, pb = calc_bb(closes)
    bl2, bs = bb_label(pb)
    row["bb"] = {"v": round(pb,3) if pb is not None else None, "l": bl2}; scores.append(bs)

    # EMA50
    e50 = calc_ema(closes, 50)
    if e50 and closes[-1]:
        ep = (closes[-1]-e50)/e50*100
        el, es = ema50_label(ep)
        row["ema50"] = {"v": f"{ep:+.2f}%", "l": el}; scores.append(es)
    else:
        row["ema50"] = {"v": None, "l": "N/A"}; scores.append(6)

    # VWAP
    vv, vp = calc_vwap(highs, lows, closes, volumes)
    vl, vs = vwap_label(vp)
    row["vwap"] = {"v": f"{vp:+.2f}%" if vp is not None else None, "l": vl}; scores.append(vs)

    avg = sum(scores)/len(scores)
    sl, sc = score_to_signal(avg)
    row["result"] = {"l": sl, "css": sc, "score": round(avg,2)}
    return row


# ── Multi-symbol multi-timeframe ──────────────────────────────────────────────
# Supported timeframes for analyzer
TF_LIST = ["5m","15m","1h","4h","1D","1W"]

def analyze_symbol(symbol: str, timeframes: list = None) -> dict:
    if timeframes is None:
        timeframes = ["15m","1h","1D"]

    quote = None
    try:
        from core.market_data import get_quote_live
        quote = get_quote_live(symbol)
    except:
        pass

    tf_results = {}
    all_scores = []

    for tf in timeframes:
        # Map UI TF names to PERIOD_CONFIG keys
        period_key = tf  # they match directly now
        bars = get_bars(symbol, period_key)
        if not bars:
            tf_results[tf] = {"error": "no data"}
            continue
        result = analyze_one_tf(bars)
        tf_results[tf] = result
        if "result" in result:
            all_scores.append(result["result"]["score"])

    oa = sum(all_scores)/len(all_scores) if all_scores else 6.0
    ol, oc = score_to_signal(oa)

    return {
        "symbol":        symbol.upper(),
        "price":         quote.get("price") if quote else None,
        "source":        quote.get("source","demo") if quote else "demo",
        "timeframes":    tf_results,
        "overall_score": round(oa, 2),
        "overall_label": ol,
        "overall_css":   oc,
    }


def analyze_multiple(symbols: list, timeframes: list = None) -> list:
    results = []
    for sym in symbols:
        try:
            r = analyze_symbol(sym.upper().strip(), timeframes)
            results.append(r)
        except Exception as e:
            results.append({"symbol": sym.upper(), "error": str(e), "timeframes": {}})
    return results
