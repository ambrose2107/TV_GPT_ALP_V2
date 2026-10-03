"""Lightweight AI crash early-warning model for the Backtest tab."""
from __future__ import annotations
import io, threading, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from core.logger import get_logger
from core.market_data import alpaca_get_multi_bars, alpaca_get_bars, yahoo_get_chart

logger = get_logger(__name__)

_CACHE={"payload":None,"ts":0.0}
_FUND_CACHE={"payload":None,"ts":0.0}
_LOCK=threading.Lock()
CACHE_TTL=6*60*60
FUND_CACHE_TTL=24*60*60
_HIST_CACHE={"payload":None,"ts":0.0}
HIST_CACHE_TTL=24*60*60
FRED={"hy_oas":"BAMLH0A0HYM2","nfci":"NFCI","dfii10":"DFII10","unrate":"UNRATE"}
MARKET=["SPY","QQQ","RSP","IWM","SOXX","^VIX"]
HYPERSCALERS=["MSFT","GOOGL","AMZN","META","ORCL"]

def _clip(x,lo=0.0,hi=100.0):
    try:return float(max(lo,min(hi,x)))
    except:return 0.0

def _fred(sid):
    u=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1990-01-01"
    r=requests.get(u,timeout=12); r.raise_for_status()
    d=pd.read_csv(io.StringIO(r.text)); d["observation_date"]=pd.to_datetime(d["observation_date"],errors="coerce")
    d[sid]=pd.to_numeric(d[sid],errors="coerce")
    return d.dropna(subset=["observation_date",sid]).set_index("observation_date")[sid]

def _market():
    stocks=[s for s in MARKET if s != "^VIX"]
    raw=alpaca_get_multi_bars(stocks,timeframe="1Day",limit=800)
    frames={}
    if raw:
        for sym,bars in raw.items():
            if not bars: continue
            z=pd.DataFrame(bars)
            if z.empty or "t" not in z.columns or "c" not in z.columns: continue
            z.index=pd.to_datetime(z["t"],utc=True,errors="coerce")
            frames[sym]=pd.to_numeric(z["c"],errors="coerce")
    # Batch Alpaca can fail independently of per-symbol requests. Retry missing
    # symbols individually, then use the lightweight Yahoo chart API (no yfinance
    # cookie/session dependency), which is more reliable on small Render workers.
    for sym in stocks:
        if sym in frames: continue
        try:
            bars=alpaca_get_bars(sym,timeframe="1Day",limit=800)
            if bars:
                z=pd.DataFrame(bars)
                if not z.empty and "t" in z.columns and "c" in z.columns:
                    z.index=pd.to_datetime(z["t"],utc=True,errors="coerce")
                    q=pd.to_numeric(z["c"],errors="coerce").dropna()
                    if not q.empty: frames[sym]=q
        except Exception: pass
    for sym in stocks:
        if sym in frames: continue
        try:
            chart=yahoo_get_chart(sym,interval="1d",period="3y")
            if chart and chart.get("timestamps") and chart.get("close"):
                q=pd.Series(pd.to_numeric(chart["close"],errors="coerce"),
                    index=pd.to_datetime(chart["timestamps"],unit="s",utc=True,errors="coerce")).dropna()
                if not q.empty: frames[sym]=q
        except Exception: pass

    # VIX is an index, not an Alpaca-tradable symbol. Fetch it from Yahoo,
    # then use CBOE's public historical CSV as an independent fallback.
    vix_loaded = False
    try:
        chart = yahoo_get_chart("^VIX", interval="1d", period="3y")
        if chart:
            ts = chart.get("timestamps") or []
            closes = chart.get("close") or []
            if ts and closes and len(ts) == len(closes):
                q = pd.Series(
                    pd.to_numeric(closes, errors="coerce"),
                    index=pd.to_datetime(ts, unit="s", utc=True, errors="coerce"),
                    name="^VIX",
                ).dropna()
                q = q[~q.index.isna()]
                if not q.empty:
                    frames["^VIX"] = q
                    vix_loaded = True
    except Exception as exc:
        logger.warning("Yahoo VIX fetch failed: %s", exc)

    if not vix_loaded:
        try:
            url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
            response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            vix_df = pd.read_csv(io.StringIO(response.text))
            date_col = next((x for x in vix_df.columns if x.strip().lower() == "date"), None)
            close_col = next((x for x in vix_df.columns if x.strip().lower() in ("close", "vix close")), None)
            if date_col and close_col:
                q = pd.Series(
                    pd.to_numeric(vix_df[close_col], errors="coerce").to_numpy(),
                    index=pd.to_datetime(vix_df[date_col], errors="coerce", utc=True),
                    name="^VIX",
                ).dropna()
                q = q[~q.index.isna()]
                q = q[q.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1100)]
                if not q.empty:
                    frames["^VIX"] = q
                    vix_loaded = True
        except Exception as exc:
            logger.warning("CBOE VIX history fetch failed: %s", exc)

    # Alpaca does not provide the VIX index through its ordinary stock-bars API.
    # If both official-index sources fail, use VIXY from Alpaca as a clearly
    # labelled ETF proxy for live risk scoring; never treat its price as VIX.
    volatility_source = "Official VIX index (Yahoo/CBOE)"
    if not vix_loaded:
        try:
            bars = alpaca_get_bars("VIXY", timeframe="1Day", limit=800)
            if bars:
                z = pd.DataFrame(bars)
                if not z.empty and "t" in z.columns and "c" in z.columns:
                    z.index = pd.to_datetime(z["t"], utc=True, errors="coerce")
                    q = pd.to_numeric(z["c"], errors="coerce").dropna()
                    if not q.empty:
                        frames["VIXY"] = q
                        vix_loaded = True
                        volatility_source = "Alpaca VIXY ETF proxy (20D return)"
        except Exception as exc:
            logger.warning("Alpaca VIXY proxy fetch failed: %s", exc)

    if not frames or "SPY" not in frames:
        raise RuntimeError("Live equity market data unavailable from Alpaca and Yahoo.")
    if not vix_loaded or ("^VIX" not in frames and "VIXY" not in frames):
        raise RuntimeError("Volatility data unavailable from Yahoo, CBOE and Alpaca VIXY. No placeholder volatility value used.")
    if "^VIX" in frames:
        volatility_source = "Official VIX index (Yahoo/CBOE)"
    # Align the independently fetched series on their common date index.
    px = pd.concat(frames, axis=1).dropna(how="all")
    px.attrs["volatility_source"] = volatility_source
    return px

def _fundamental_proxy_uncached():
    rows=[]
    for ticker in HYPERSCALERS:
        try:
            t=yf.Ticker(ticker); inc=t.quarterly_income_stmt; cf=t.quarterly_cashflow; bs=t.quarterly_balance_sheet
            if inc is None or inc.empty: continue
            rk=next((k for k in ["Total Revenue","Operating Revenue"] if k in inc.index),None)
            ck=next((k for k in ["Capital Expenditure","Capital Ex Expenditures"] if k in cf.index),None)
            dk=next((k for k in ["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt"] if k in bs.index),None)
            if not rk: continue
            rev=pd.to_numeric(inc.loc[rk],errors="coerce").dropna().sort_index()
            cap=pd.to_numeric(cf.loc[ck],errors="coerce").dropna().sort_index() if ck else pd.Series(dtype=float)
            debt=pd.to_numeric(bs.loc[dk],errors="coerce").dropna().sort_index() if dk else pd.Series(dtype=float)
            if len(rev)<4: continue
            rt=float(rev.tail(4).sum()); ct=float(abs(cap.tail(4).sum())) if len(cap) else None
            dn=float(debt.iloc[-1]) if len(debt) else None; dp=float(debt.iloc[-5]) if len(debt)>=5 else None
            rp=float(rev.iloc[-8:-4].sum()) if len(rev)>=8 else None
            cp=float(abs(cap.iloc[-8:-4].sum())) if len(cap)>=8 else None
            rows.append({"ticker":ticker,"revenue_ttm":rt,"capex_ttm":ct,"debt":dn,
              "revenue_growth":((rt/rp)-1)*100 if rp and rp>0 else None,
              "capex_growth":((ct/cp)-1)*100 if ct is not None and cp and cp>0 else None,
              "debt_growth":((dn/dp)-1)*100 if dn is not None and dp and dp>0 else None})
        except Exception: continue
    if not rows:return {"available":False,"rows":[],"capex_revenue":None,"capex_growth_gap":None,"debt_growth_gap":None}
    d=pd.DataFrame(rows); cr=(d.capex_ttm/d.revenue_ttm*100).replace([np.inf,-np.inf],np.nan).dropna()
    cg=(d.capex_growth-d.revenue_growth).dropna(); dg=(d.debt_growth-d.revenue_growth).dropna()
    return {"available":True,"rows":rows,"capex_revenue":float(cr.mean()) if len(cr) else None,
            "capex_growth_gap":float(cg.mean()) if len(cg) else None,"debt_growth_gap":float(dg.mean()) if len(dg) else None}

def _fundamental_proxy():
    """Cache slow quarterly financial statements separately from market data."""
    now=time.time()
    if _FUND_CACHE["payload"] is not None and now-_FUND_CACHE["ts"] < FUND_CACHE_TTL:
        return _FUND_CACHE["payload"]
    payload=_fundamental_proxy_uncached()
    _FUND_CACHE["payload"]=payload
    _FUND_CACHE["ts"]=now
    return payload


def _latest_change(s,n):
    s=s.dropna()
    if s.empty:return None,None
    return float(s.iloc[-1]),float(s.iloc[-1]-s.iloc[-min(n+1,len(s))])

def _score(px,f,fund):
    hy=f["hy_oas"]; hn,hc=_latest_change(hy,20); hb=float(hy.rolling(252,min_periods=60).median().iloc[-1]) if len(hy)>=60 else float(hy.median())
    credit=_clip(.65*_clip((hn/max(hb,.5)-.85)*100)+.35*_clip((hc or 0)*35))
    nf=f["nfci"]; nn=float(nf.iloc[-1]); nm=nf.rolling(104,min_periods=26).mean().iloc[-1]; ns=nf.rolling(104,min_periods=26).std().iloc[-1]
    liq=_clip(((nn-float(nm))/max(float(ns),.15)+.5)*35) if len(nf)>=26 else 50
    real=f["dfii10"]; rn,rc=_latest_change(real,60); real_score=_clip(((rn or 0)-1)*22+max(rc or 0,0)*10)
    volatility_source = px.attrs.get("volatility_source", "Official VIX index (Yahoo/CBOE)")
    if "^VIX" in px.columns and px["^VIX"].notna().any():
        v = px["^VIX"].dropna()
        vn = float(v.iloc[-1])
        volatility_proxy = None
        vol = _clip((vn-16)*4)
        volatility_source = "Official VIX index (Yahoo/CBOE)"
    elif "VIXY" in px.columns and px["VIXY"].notna().any():
        v = px["VIXY"].dropna()
        vn = None
        vixy_return = ((float(v.iloc[-1])/float(v.iloc[-min(21, len(v))])-1)*100) if len(v)>1 else 0.0
        volatility_proxy = round(vixy_return, 2)
        vol = _clip(50 + vixy_return*2.0)
        volatility_source = "Alpaca VIXY ETF proxy (20D return)"
    else:
        raise RuntimeError("No valid volatility index or ETF proxy available.")
    b=[]
    bd={}
    for a,label in [("RSP","equal_weight"),("IWM","small_caps"),("SOXX","semis")]:
        r=(px[a]/px["SPY"]).dropna()
        ret=((r.iloc[-1]/r.iloc[-64])-1)*100 if len(r)>64 else 0
        b.append(_clip(50-ret*2.5)); bd[label]=round(ret,2)
    breadth=float(np.mean(b))
    un=f["unrate"].dropna(); sahm=None
    if len(un)>=13:
        m=un.rolling(3).mean(); sahm=float(m.iloc[-1]-m.rolling(12).min().iloc[-1])
    rec=_clip((sahm or 0)*120)
    if len(un)>=6:rec=_clip(rec+(float(un.iloc[-1])-float(un.iloc[-4]))*18)
    cgap=fund.get("capex_growth_gap"); dgap=fund.get("debt_growth_gap"); ratio=fund.get("capex_revenue")
    ai=35
    if cgap is not None: ai+=_clip(cgap*2,-20,35)
    if ratio is not None: ai+=_clip((ratio-15)*1.5,-15,25)
    ai=_clip(ai)
    afin=_clip(35+(_clip(dgap*2.5,-15,45) if dgap is not None else 0))
    comps={"AI Fundamental":round(_clip(.65*ai+.35*afin),1),"AI Financing":round(afin,1),"Credit":round(credit,1),
      "Liquidity":round(liq,1),"Market Breadth":round(breadth,1),"Real Rates":round(real_score,1),"Volatility":round(vol,1),"Recession":round(rec,1)}
    w={"AI Fundamental":.18,"AI Financing":.14,"Credit":.20,"Liquidity":.10,"Market Breadth":.14,"Real Rates":.10,"Volatility":.05,"Recession":.09}
    score=_clip(sum(comps[k]*w[k] for k in comps))
    return {"score":round(score,1),"components":comps,"details":{"hy_oas":hn,"hy_oas_20d_change":hc,"nfci":nn,
      "real10y":rn,"real10y_60d_change":rc,"vix":vn,"volatility_proxy":volatility_proxy,
      "volatility_source":volatility_source,"sahm":sahm,"breadth":bd,
      "capex_revenue_pct":ratio,"capex_growth_gap_pct":cgap,"debt_growth_gap_pct":dgap}}

def _v2_overlay(score, components, history):
    """V2 regime confirmation: persistence + acceleration + independent blocks.
    This is a confirmation layer, not a separately trained probability model.
    """
    vals=[float(x.get("v",0)) for x in (history or []) if x.get("v") is not None]
    tail=vals[-20:]
    persistence=(sum(v>=50 for v in tail)/len(tail)*100) if tail else 0.0
    acceleration=(tail[-1]-tail[0]) if len(tail)>=2 else 0.0
    confirmed=sum(bool(components.get(k,0)>=70) for k in
                  ("Credit","Liquidity","Market Breadth","Recession"))
    confirmation_score=min(100.0, confirmed/4.0*100.0)
    accel_score=_clip(50.0+acceleration*5.0)
    v2_score=_clip(0.70*float(score)+0.12*persistence+0.10*confirmation_score+0.08*accel_score)
    if v2_score>=85 or (v2_score>=75 and confirmed>=3):
        regime="CRISIS"
    elif v2_score>=70 or (v2_score>=62 and confirmed>=2):
        regime="DEFENSIVE"
    elif v2_score>=50 or (v2_score>=45 and persistence>=60):
        regime="PRE-CRISIS"
    elif v2_score>=25 or (v2_score>=20 and persistence>=50):
        regime="WATCH"
    else:
        regime="NORMAL"
    return {
        "score":round(v2_score,1),
        "regime":regime,
        "persistence_20d":round(persistence,1),
        "acceleration_20d":round(acceleration,1),
        "confirmed_blocks":confirmed,
        "confirmation_score":round(confirmation_score,1),
        "signal_strength":"HIGH" if confirmed>=3 and persistence>=60 else
                         "ELEVATED" if confirmed>=2 or persistence>=60 else "LOW",
        "note":"V2 adds persistence, acceleration and multi-block confirmation; it is not a calibrated crash probability."
    }


def _regime(s):
    return "CRISIS" if s>=85 else "DEFENSIVE" if s>=70 else "PRE-CRISIS" if s>=50 else "WATCH" if s>=25 else "NORMAL"

def _history(px,core=None):
    """Recent, time-varying broad-market proxy history (not the live composite score)."""
    idx=px.index
    idx=idx[idx>=idx.max()-pd.Timedelta(days=365)]
    if len(idx)>140: idx=idx[-140:]
    out=[]
    for d in idx:
        try:
            hist=px.loc[:d]
            has_vix = "^VIX" in hist.columns and hist["^VIX"].notna().any()
            has_vixy = "VIXY" in hist.columns and hist["VIXY"].notna().any()
            v = hist["^VIX"].dropna() if has_vix else (hist["VIXY"].dropna() if has_vixy else pd.Series(dtype=float))
            q=hist["QQQ"].dropna()
            spy=hist["SPY"].dropna()
            if v.empty or len(q)<20 or len(spy)<20: continue
            if has_vix:
                vn=float(v.iloc[-1])
                vol=_clip((vn-18)*3.2)
            else:
                vixy_return=((float(v.iloc[-1])/float(v.iloc[-min(21,len(v))])-1)*100) if len(v)>1 else 0.0
                vol=_clip(50+vixy_return*2.0)
            mom=((float(q.iloc[-1])/float(q.iloc[-min(127,len(q))])-1)*100) if len(q)>20 else 0.0
            sh=spy.tail(126)
            dd=(float(sh.max())/float(sh.iloc[-1])-1)*100 if len(sh)>20 and float(sh.max()) else 0.0
            rel=[]
            for sym in ("RSP","IWM","SOXX"):
                if sym in hist.columns:
                    r=(hist[sym]/hist["SPY"]).dropna()
                    if len(r)>20: rel.append((float(r.iloc[-1])/float(r.iloc[-min(64,len(r))])-1)*100)
            breadth=float(np.mean([_clip(50-x*2.5) for x in rel])) if rel else 50.0
            momentum=_clip(50-mom*1.5)
            drawdown=_clip(dd*2.2)
            score=_clip(.30*vol+.35*momentum+.20*drawdown+.15*breadth)
            out.append({"t":d.strftime("%Y-%m-%d"),"v":round(score,1)})
        except Exception:
            continue
    return out

def _historical_crash_replay(f):
    """Replay a time-varying, broad-market warning proxy over prior drawdowns.
    This is a transparent historical diagnostic, not a fitted probability model.
    Cached separately so it adds only a few daily series and does not slow every refresh.
    """
    now=time.time()
    cached=_HIST_CACHE["payload"]
    cache_ttl=HIST_CACHE_TTL if cached and cached.get("available") else 15*60
    if cached is not None and now-_HIST_CACHE["ts"]<cache_ttl:
        return cached
    try:
        from concurrent.futures import ThreadPoolExecutor
        def _load_long(sym):
            chart=yahoo_get_chart(sym,interval="1d",period="max")
            if not chart or not chart.get("timestamps") or not chart.get("close"):
                raise RuntimeError("Long-run Yahoo history unavailable for "+sym)
            ix=pd.to_datetime(chart["timestamps"],unit="s",utc=True,errors="coerce").tz_localize(None).normalize()
            q=pd.Series(pd.to_numeric(chart["close"],errors="coerce"),index=ix).dropna()
            q=q[~q.index.isna()]
            return sym,q[~q.index.duplicated(keep="last")]
        with ThreadPoolExecutor(max_workers=3) as pool:
            series=dict(pool.map(_load_long,("SPY","QQQ","^VIX")))
        p=pd.concat(series,axis=1).sort_index()
        p=p[~p.index.duplicated(keep="last")]
        hy=f.get("hy_oas",pd.Series(dtype=float)).copy()
        nf=f.get("nfci",pd.Series(dtype=float)).copy()
        hy.index=pd.to_datetime(hy.index,errors="coerce").tz_localize(None).normalize()
        nf.index=pd.to_datetime(nf.index,errors="coerce").tz_localize(None).normalize()
        p["hy_oas"]=hy.reindex(p.index).ffill()
        p["nfci"]=nf.reindex(p.index).ffill()
        # Daily values use only data available on that date; no present-day component
        # scores are carried backward into the historical replay.
        hy_med=p["hy_oas"].rolling(252,min_periods=60).median()
        hy_chg=p["hy_oas"].diff(20)
        credit=((p["hy_oas"]/hy_med.clip(lower=.5)-.85)*65 + hy_chg.fillna(0)*25).clip(0,100)
        vol=((p["^VIX"]-18)*3.2).clip(0,100)
        qmom=p["QQQ"].pct_change(126)*100
        momentum=(50-qmom*1.5).clip(0,100)
        spy_high=p["SPY"].rolling(252,min_periods=60).max()
        drawdown=((1-p["SPY"]/spy_high).clip(lower=0)*220).clip(0,100)
        nf_mean=p["nfci"].rolling(104,min_periods=26).mean()
        nf_std=p["nfci"].rolling(104,min_periods=26).std().clip(lower=.15)
        liquidity=(40+(p["nfci"]-nf_mean)/nf_std*25).clip(0,100)
        p["replay_score"]=.30*credit+.20*vol+.25*momentum+.15*drawdown+.10*liquidity
        p=p.dropna(subset=["SPY","QQQ","^VIX","replay_score"])
        episodes=[
          {"name":"Dot-com bust","symbol":"QQQ","peak":"2000-03-10"},
          {"name":"Global financial crisis","symbol":"SPY","peak":"2007-10-09"},
          {"name":"COVID shock","symbol":"SPY","peak":"2020-02-19"},
          {"name":"2022 bear market","symbol":"SPY","peak":"2022-01-03"},
        ]
        results=[]
        for e in episodes:
            s=e["symbol"]; peak_date=pd.Timestamp(e["peak"])
            q=p[s].dropna()
            if q.empty or q.index.min()>peak_date or q.index.max()<peak_date:
                results.append({"episode":e["name"],"peak_date":e["peak"],"status":"Insufficient history"})
                continue
            peak_pos=int(q.index.get_indexer([q.index[q.index.get_indexer([peak_date],method="nearest")[0]]])[0])
            peak_ix=q.index[peak_pos]; peak_price=float(q.iloc[peak_pos])
            future=q.iloc[peak_pos+1:peak_pos+253]
            breach=future[future<=peak_price*.80]
            if breach.empty:
                results.append({"episode":e["name"],"peak_date":peak_ix.strftime("%Y-%m-%d"),"status":"20% threshold not found in 12M"})
                continue
            breach_date=breach.index[0]
            hist=p.loc[:breach_date]
            pre=hist.loc[(hist.index>=breach_date-pd.Timedelta(days=90)) & (hist.index<breach_date)]
            crossed=pre[pre["replay_score"]>=60]
            first=crossed.index[0] if not crossed.empty else None
            peak_row=p.loc[:peak_ix].iloc[-1]
            results.append({
              "episode":e["name"],"peak_date":peak_ix.strftime("%Y-%m-%d"),
              "20pct_date":breach_date.strftime("%Y-%m-%d"),
              "days_to_20pct":int((breach_date-peak_ix).days),
              "score_at_peak":round(float(peak_row["replay_score"]),1),
              "max_score_pre_breach":round(float(pre["replay_score"].max()),1) if not pre.empty else None,
              "first_signal_date":first.strftime("%Y-%m-%d") if first is not None else None,
              "lead_days":int((breach_date-first).days) if first is not None else None,
              "status":"Signal before -20% threshold" if first is not None else "No 60+ signal in 90D pre-breach window"
            })
        payload={"available":True,"episodes":results,"method":"Time-varying broad-market proxy: HY OAS level/change, VIX, QQQ 6M momentum, SPY 1Y drawdown and NFCI. 60/100 is an exploratory threshold, not a calibrated probability.",
          "coverage_start":p.index.min().strftime("%Y-%m-%d") if not p.empty else None,
          "coverage_end":p.index.max().strftime("%Y-%m-%d") if not p.empty else None,
          "threshold":60,"warning":"This is a retrospective episode replay, not a fully out-of-sample validation. It tests broad-market crash stress, not AI-specific crashes."}
    except Exception as exc:
        logger.warning("Historical crash replay unavailable: %s",exc)
        payload={"available":False,"episodes":[],"warning":"Historical replay unavailable: "+str(exc)[:220],
          "method":"Requires long-run SPY, QQQ, VIX and historical macro series."}
    _HIST_CACHE["payload"]=payload; _HIST_CACHE["ts"]=now
    return payload

def build_dashboard(force=False):
    now=time.time()
    if not force and _CACHE["payload"] is not None and now-_CACHE["ts"]<CACHE_TTL:return _CACHE["payload"]
    with _LOCK:
        now=time.time()
        if not force and _CACHE["payload"] is not None and now-_CACHE["ts"]<CACHE_TTL:return _CACHE["payload"]
        px=_market(); f={k:_fred(v) for k,v in FRED.items()}; fund=_fundamental_proxy(); core=_score(px,f,fund); s=core["score"]; comps=core["components"]
        drivers=[k for k,v in sorted(comps.items(),key=lambda x:x[1],reverse=True)[:3] if v>=40]
        labels={"AI Fundamental":"Hyperscaler capex/revenue proxy","AI Financing":"Debt/revenue proxy","Credit":"HY credit stress",
          "Liquidity":"Financial conditions","Market Breadth":"RSP/IWM/SOXX vs SPY","Real Rates":"10Y real-rate pressure",
          "Volatility":"VIX pressure","Recession":"Sahm/unemployment pressure"}
        factors=[{"name":k,"value":v,"label":labels[k],"status":"HIGH" if v>=70 else "ELEVATED" if v>=50 else "LOW"} for k,v in comps.items()]
        hist=_history(px,core)
        market_date=px.index.max()
        market_age_days=max(0,int((pd.Timestamp.now(tz="UTC")-market_date).total_seconds()/86400)) if getattr(market_date,"tzinfo",None) else max(0,int((datetime.now()-market_date).total_seconds()/86400))
        fundamental_count=len(fund.get("rows",[]) or [])
        quality_warnings=[]
        if market_age_days>5: quality_warnings.append("Market prices may be stale.")
        if fundamental_count<3: quality_warnings.append("Hyperscaler fundamental coverage is limited; AI capex/debt scores rely partly on neutral defaults.")
        missing_macro=[k for k,v in f.items() if v is None or len(v.dropna())==0]
        if missing_macro: quality_warnings.append("Missing macro series: "+", ".join(missing_macro))
        volatility_source = core["details"].get("volatility_source", "unknown")
        if "VIXY ETF proxy" in volatility_source:
            quality_warnings.append("Official VIX index unavailable; using Alpaca VIXY ETF proxy. VIXY is not the VIX index.")
        data_quality={"market_latest_date":market_date.strftime("%Y-%m-%d"),"market_age_days":market_age_days,
          "volatility_source":volatility_source,
          "fundamental_companies_covered":fundamental_count,"fundamental_companies_expected":len(HYPERSCALERS),
          "macro_series_covered":len(f)-len(missing_macro),"macro_series_expected":len(FRED),
          "warnings":quality_warnings}
        payload={"as_of":datetime.now(timezone.utc).isoformat(),"score":s,"regime":_regime(s),"drivers":drivers,
          "components":factors,"confirmations":{"credit":comps["Credit"]>=70,"recession":comps["Recession"]>=70,
          "breadth":comps["Market Breadth"]>=70,"liquidity":comps["Liquidity"]>=70},
          "details":core["details"],"fundamentals":fund,"data_quality":data_quality,"history":hist,
          "v2":_v2_overlay(s,comps,hist),
          "historical_replay":_historical_crash_replay(f),
          "methodology":{"weights":{"AI Fundamental":18,"AI Financing":14,"Credit":20,"Liquidity":10,"Market Breadth":14,"Real Rates":10,"Volatility":5,"Recession":9},
          "note":"Early-warning monitor, not a crash-date predictor. Hyperscaler capex is a proxy, not AI-only capex."}}
        _CACHE["payload"]=payload; _CACHE["ts"]=now; return payload
