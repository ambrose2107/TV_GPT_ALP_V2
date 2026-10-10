"""Lightweight AI crash early-warning model for the Backtest tab."""
from __future__ import annotations
import io, os, threading, time
from datetime import datetime, timedelta, timezone
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
# Historical validation data is deliberately separate from live market inputs.
# Render Free has ephemeral disk; this cache avoids repeated downloads within a running instance.
_VALIDATION_CACHE_PATH=os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance", "ai_crash_validation_history.csv")
# Production replay is deliberately isolated from the live dashboard path.
_PROD_CACHE={"payload":None,"ts":0.0}
_PROD_JOB={"status":"idle","result":None,"error":None,"started_at":None,"finished_at":None}
_PROD_LOCK=threading.Lock()
_PROD_THREAD=None
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
    """Build a company-level quarterly capex/debt proxy with explicit period coverage.

    yfinance statement column dates are fiscal period ends, not SEC filing dates.
    They are labelled as period ends so the dashboard does not imply point-in-time
    availability that this live snapshot source cannot guarantee.
    """
    rows=[]
    for ticker in HYPERSCALERS:
        row={"ticker":ticker,"source":"Yahoo Finance quarterly statements",
             "revenue_ttm":None,"capex_ttm":None,"debt":None,
             "revenue_growth":None,"capex_growth":None,"debt_growth":None,
             "revenue_period_end":None,"capex_period_end":None,"debt_period_end":None,
             "coverage_status":"MISSING"}
        try:
            t=yf.Ticker(ticker); inc=t.quarterly_income_stmt; cf=t.quarterly_cashflow; bs=t.quarterly_balance_sheet
            rk=next((k for k in ["Total Revenue","Operating Revenue"] if inc is not None and not inc.empty and k in inc.index),None)
            ck=next((k for k in ["Capital Expenditure","Capital Ex Expenditures"] if cf is not None and not cf.empty and k in cf.index),None)
            dk=next((k for k in ["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt"] if bs is not None and not bs.empty and k in bs.index),None)
            rev=pd.to_numeric(inc.loc[rk],errors="coerce").dropna().sort_index() if rk else pd.Series(dtype=float)
            cap=pd.to_numeric(cf.loc[ck],errors="coerce").dropna().sort_index() if ck else pd.Series(dtype=float)
            debt=pd.to_numeric(bs.loc[dk],errors="coerce").dropna().sort_index() if dk else pd.Series(dtype=float)
            if len(rev)>=4:
                rt=float(rev.tail(4).sum()); row["revenue_ttm"]=rt
                row["revenue_period_end"]=pd.Timestamp(rev.index[-1]).strftime("%Y-%m-%d")
                rp=float(rev.iloc[-8:-4].sum()) if len(rev)>=8 else None
                row["revenue_growth"]=((rt/rp)-1)*100 if rp and rp>0 else None
            if len(cap)>=1:
                ct=float(abs(cap.tail(4).sum())); row["capex_ttm"]=ct
                row["capex_period_end"]=pd.Timestamp(cap.index[-1]).strftime("%Y-%m-%d")
                cp=float(abs(cap.iloc[-8:-4].sum())) if len(cap)>=8 else None
                row["capex_growth"]=((ct/cp)-1)*100 if cp and cp>0 else None
            if len(debt)>=1:
                dn=float(debt.iloc[-1]); row["debt"]=dn
                row["debt_period_end"]=pd.Timestamp(debt.index[-1]).strftime("%Y-%m-%d")
                dp=float(debt.iloc[-5]) if len(debt)>=5 else None
                row["debt_growth"]=((dn/dp)-1)*100 if dp and dp>0 else None
            required=[row["revenue_ttm"] is not None,row["capex_ttm"] is not None,row["debt"] is not None]
            row["coverage_status"]="COMPLETE" if all(required) else "PARTIAL" if any(required) else "MISSING"
        except Exception as exc:
            row["error"]=str(exc)[:180]
        rows.append(row)
    d=pd.DataFrame(rows)
    valid=d[d["revenue_ttm"].notna()]
    cr=(valid.capex_ttm/valid.revenue_ttm*100).replace([np.inf,-np.inf],np.nan).dropna() if len(valid) else pd.Series(dtype=float)
    cg=(valid.capex_growth-valid.revenue_growth).dropna() if len(valid) else pd.Series(dtype=float)
    dg=(valid.debt_growth-valid.revenue_growth).dropna() if len(valid) else pd.Series(dtype=float)
    metric_counts={
        "revenue":int(d.revenue_ttm.notna().sum()),
        "capex":int(d.capex_ttm.notna().sum()),
        "debt":int(d.debt.notna().sum()),
        "capex_growth_gap":int(cg.size),
        "debt_growth_gap":int(dg.size),
    }
    return {"available":bool(len(valid)),"rows":rows,
        "companies_expected":len(HYPERSCALERS),
        "companies_with_revenue":metric_counts["revenue"],
        "companies_with_complete_statements":int((d.coverage_status=="COMPLETE").sum()),
        "metric_coverage":metric_counts,
        "capex_revenue":float(cr.mean()) if len(cr) else None,
        "capex_growth_gap":float(cg.mean()) if len(cg) else None,
        "debt_growth_gap":float(dg.mean()) if len(dg) else None,
        "source":"Yahoo Finance quarterly statements",
        "date_semantics":"Fiscal period end; not filing/availability date"}

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
    # Neutral midpoint for every unavailable AI metric. Missing observations are
    # separately surfaced in coverage metadata and must never masquerade as evidence.
    ai=50
    if cgap is not None: ai+=_clip(cgap*2,-20,35)
    if ratio is not None: ai+=_clip((ratio-15)*1.5,-15,25)
    ai=_clip(ai)
    afin=_clip(50+(_clip(dgap*2.5,-15,45) if dgap is not None else 0))
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
    # Never reuse a superficially "available" replay if its history is only a
    # recent provider window. This was masking the repository baseline fallback.
    cached_start=pd.to_datetime((cached or {}).get("coverage_start"),errors="coerce")
    cached_end=pd.to_datetime((cached or {}).get("coverage_end"),errors="coerce")
    required_start=pd.Timestamp("2000-06-01")
    required_end=pd.Timestamp.now().normalize()-pd.Timedelta(days=10)
    cached_history_valid=bool(
        cached and cached.get("available") is True
        and pd.notna(cached_start) and cached_start<=required_start
        and pd.notna(cached_end) and cached_end>=required_end
    )
    cache_ttl=HIST_CACHE_TTL if cached_history_valid else 0
    if cached is not None and cached_history_valid and now-_HIST_CACHE["ts"]<cache_ttl:
        return cached
    if cached is not None and not cached_history_valid:
        logger.warning("Ignoring cached historical replay with incomplete/stale coverage: %s to %s",
            (cached or {}).get("coverage_start"),(cached or {}).get("coverage_end"))
    try:
        from concurrent.futures import ThreadPoolExecutor
        source_map={}
        # Prefer previously verified validation history so every page refresh does not
        # re-download decades of prices. Cache is validation-only; live scoring still
        # uses the current market-data pipeline. Require all crash eras before reuse.
        cache_path=_VALIDATION_CACHE_PATH
        required_start=pd.Timestamp("2000-01-01")
        if os.path.isfile(cache_path):
            try:
                cached_history=pd.read_csv(cache_path,parse_dates=["date"]).set_index("date").sort_index()
                cached_history.index=pd.to_datetime(cached_history.index,errors="coerce").tz_localize(None).normalize()
                needed={"SPY","QQQ","^VIX"}
                if needed.issubset(cached_history.columns) and not cached_history.empty and cached_history.index.min()<=pd.Timestamp("2000-06-01") and cached_history.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                    source_map.update({sym:"Local cached validation history" for sym in needed})
                    p=cached_history[list(needed)].apply(pd.to_numeric,errors="coerce")
                    p=p[~p.index.isna()].sort_index()
                    p=p[~p.index.duplicated(keep="last")]
                    # Jump to shared scoring block below with verified cached series.
                    series=None
                else:
                    p=None
                    logger.warning("Ignoring incomplete local validation cache; it must cover 2000 through recent sessions")
            except Exception as exc:
                p=None
                logger.warning("Could not read local validation cache: %s",exc)
        else:
            p=None
        def _load_long(sym):
            # Repository baseline is the durable historical source for validation.
            # Refresh the recent tail separately; never confuse a short live window
            # with the complete historical baseline.
            baseline_path=os.path.join(os.path.dirname(__file__),"data","ai_crash_validation_history.csv")
            try:
                col="VIX_close" if sym=="^VIX" else "QQQ_adj_close" if sym=="QQQ" else "SPY_close"
                base=pd.read_csv(baseline_path,usecols=["date",col])
                ix=pd.to_datetime(base["date"],errors="coerce").dt.normalize()
                q=pd.Series(pd.to_numeric(base[col],errors="coerce").to_numpy(),index=ix).dropna()
                q=q[~q.index.isna()]
                q=q[~q.index.duplicated(keep="last")].sort_index()
                expected_start=pd.Timestamp("2000-01-01") if sym=="SPY" else pd.Timestamp("1999-03-01") if sym=="QQQ" else pd.Timestamp("1990-01-01")
                if len(q)>=250 and q.index.min()<=expected_start+pd.Timedelta(days=400):
                    # Fetch only a recent tail, merge by date, and preserve baseline
                    # values where no recent provider observation is available.
                    # Refresh the recent tail using independent providers. Yahoo is
                    # frequently rate-limited on Render; do not let a Yahoo 429 alone
                    # leave the replay silently stuck on an old baseline.
                    refreshed=False
                    try:
                        chart=yahoo_get_chart(sym,interval="1d",period="2y")
                        if chart and chart.get("timestamps") and chart.get("close"):
                            rx=pd.to_datetime(chart["timestamps"],unit="s",utc=True,errors="coerce").tz_localize(None).normalize()
                            rq=pd.Series(pd.to_numeric(chart["close"],errors="coerce"),index=rx).dropna()
                            rq=rq[~rq.index.isna()]
                            rq=rq[~rq.index.duplicated(keep="last")]
                            if not rq.empty and rq.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                                q=pd.concat([q.loc[q.index<rq.index.min()],rq]).sort_index()
                                q=q[~q.index.duplicated(keep="last")]
                                refreshed=True
                    except Exception as exc:
                        logger.warning("Yahoo recent-tail refresh failed for %s: %s",sym,exc)
                    if not refreshed and sym in ("SPY","QQQ"):
                        try:
                            bars=alpaca_get_bars(sym,timeframe="1Day",limit=800,start=(datetime.now(timezone.utc)-timedelta(days=900)).strftime("%Y-%m-%dT%H:%M:%SZ"),adjustment="all")
                            if bars:
                                z=pd.DataFrame(bars)
                                if not z.empty and "t" in z.columns and "c" in z.columns:
                                    rx=pd.to_datetime(z["t"],utc=True,errors="coerce").dt.tz_localize(None).dt.normalize()
                                    rq=pd.Series(pd.to_numeric(z["c"],errors="coerce").to_numpy(),index=rx).dropna()
                                    rq=rq[~rq.index.isna()]
                                    rq=rq[~rq.index.duplicated(keep="last")]
                                    if not rq.empty and rq.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                                        q=pd.concat([q.loc[q.index<rq.index.min()],rq]).sort_index()
                                        q=q[~q.index.duplicated(keep="last")]
                                        refreshed=True
                        except Exception as exc:
                            logger.warning("Alpaca recent-tail refresh failed for %s: %s",sym,exc)
                    if not refreshed and sym in ("SPY","QQQ"):
                        try:
                            response=requests.get("https://stooq.com/q/d/l/",
                                params={"s":sym.lower()+".us","i":"d"},
                                headers={"User-Agent":"Mozilla/5.0"},timeout=18)
                            response.raise_for_status()
                            d=pd.read_csv(io.StringIO(response.text))
                            if {"Date","Close"}.issubset(d.columns):
                                rx=pd.to_datetime(d["Date"],errors="coerce").dt.normalize()
                                rq=pd.Series(pd.to_numeric(d["Close"],errors="coerce").to_numpy(),index=rx).dropna()
                                rq=rq[~rq.index.isna()]
                                rq=rq[~rq.index.duplicated(keep="last")]
                                if not rq.empty and rq.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                                    q=pd.concat([q.loc[q.index<rq.index.min()],rq]).sort_index()
                                    q=q[~q.index.duplicated(keep="last")]
                                    refreshed=True
                        except Exception as exc:
                            logger.warning("Stooq recent-tail refresh failed for %s: %s",sym,exc)
                    if not refreshed and sym=="^VIX":
                        try:
                            response=requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                                headers={"User-Agent":"Mozilla/5.0"},timeout=18)
                            response.raise_for_status()
                            d=pd.read_csv(io.StringIO(response.text))
                            dc=next((c for c in d.columns if c.strip().lower()=="date"),None)
                            cc=next((c for c in d.columns if c.strip().lower() in ("close","vix close")),None)
                            if dc and cc:
                                rx=pd.to_datetime(d[dc],errors="coerce").dt.normalize()
                                rq=pd.Series(pd.to_numeric(d[cc],errors="coerce").to_numpy(),index=rx).dropna()
                                rq=rq[~rq.index.isna()]
                                rq=rq[~rq.index.duplicated(keep="last")]
                                if not rq.empty and rq.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                                    q=pd.concat([q.loc[q.index<rq.index.min()],rq]).sort_index()
                                    q=q[~q.index.duplicated(keep="last")]
                                    refreshed=True
                        except Exception as exc:
                            logger.warning("CBOE recent-tail refresh failed for VIX: %s",exc)
                    latest=q.index.max() if not q.empty else None
                    fresh=latest is not None and latest>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10)
                    source_map[sym]="Repository baseline + refreshed recent tail" if refreshed and fresh else "Repository historical baseline (stale tail)"
                    if not fresh:
                        logger.warning("Historical baseline for %s ends %s; recent refresh unavailable",sym,latest)
                    return sym,q
            except Exception as exc:
                logger.warning("Repository historical baseline unavailable for %s: %s",sym,exc)

            # Yahoo can silently return only recent rows for a long period request
            # on hosted workers. Fetch bounded 5-year chunks, validate coverage, then
            # combine. Never treat a short recent window as long-run crash history.
            start_dt=datetime(1990,1,1,tzinfo=timezone.utc)
            end_dt=datetime.now(timezone.utc)+timedelta(days=1)
            for host in ("query1.finance.yahoo.com","query2.finance.yahoo.com"):
                try:
                    chunks=[]
                    chunk_start=start_dt
                    while chunk_start < end_dt:
                        chunk_end=min(chunk_start+timedelta(days=365*5+2),end_dt)
                        url=f"https://{host}/v8/finance/chart/{requests.utils.quote(sym,safe='')}"
                        response=requests.get(
                            url,
                            params={"period1":int(chunk_start.timestamp()),"period2":int(chunk_end.timestamp()),
                                    "interval":"1d","events":"div,splits"},
                            headers={"User-Agent":"Mozilla/5.0 (compatible; CrashReplay/1.0)"},
                            timeout=15,
                        )
                        response.raise_for_status()
                        obj=response.json().get("chart",{})
                        if obj.get("error"):
                            raise RuntimeError(str(obj["error"]))
                        result=(obj.get("result") or [None])[0]
                        if result:
                            stamps=result.get("timestamp") or []
                            indicators=result.get("indicators",{})
                            adj=(indicators.get("adjclose") or [{}])[0]
                            quote=(indicators.get("quote") or [{}])[0]
                            closes=adj.get("adjclose") or quote.get("close") or []
                            if stamps and closes and len(stamps)==len(closes):
                                ix=pd.to_datetime(stamps,unit="s",utc=True,errors="coerce").tz_localize(None).normalize()
                                q=pd.Series(pd.to_numeric(closes,errors="coerce"),index=ix).dropna()
                                q=q[~q.index.isna()]
                                q=q[~q.index.duplicated(keep="last")].sort_index()
                                if not q.empty:
                                    chunks.append(q)
                        chunk_start=chunk_end
                    if chunks:
                        q=pd.concat(chunks).sort_index()
                        q=q[~q.index.duplicated(keep="last")]
                        expected_start=pd.Timestamp("2000-01-01") if sym=="SPY" else pd.Timestamp("1999-03-01") if sym=="QQQ" else pd.Timestamp("1990-01-01")
                        # Permit modest listing/provider gaps, but reject recent-only data.
                        if len(q)>=250 and q.index.min()<=expected_start+pd.Timedelta(days=400):
                            source_map[sym]=f"Yahoo Finance chunked history ({host})"
                            return sym,q
                        logger.warning("Yahoo chunks for %s did not cover expected history: %s to %s (%d rows)",sym,q.index.min() if len(q) else None,q.index.max() if len(q) else None,len(q))
                    else:
                        logger.warning("Yahoo chunked history returned no data for %s via %s",sym,host)
                except Exception as exc:
                    logger.warning("Yahoo chunked history failed for %s via %s: %s",sym,host,exc)
            # Try the existing shared Yahoo loader as a final Yahoo-specific path.
            try:
                chart=yahoo_get_chart(sym,interval="1d",period="max")
                if chart and chart.get("timestamps") and chart.get("close"):
                    ix=pd.to_datetime(chart["timestamps"],unit="s",utc=True,errors="coerce").tz_localize(None).normalize()
                    q=pd.Series(pd.to_numeric(chart["close"],errors="coerce"),index=ix).dropna()
                    q=q[~q.index.isna()]
                    q=q[~q.index.duplicated(keep="last")].sort_index()
                    expected_start=pd.Timestamp("2000-01-01") if sym=="SPY" else pd.Timestamp("1999-03-01") if sym=="QQQ" else pd.Timestamp("1990-01-01")
                    if len(q)>=250 and q.index.min()<=expected_start+pd.Timedelta(days=400) and q.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                        source_map[sym]="Yahoo Finance max history"
                        return sym,q
                    logger.warning("Yahoo max history for %s has incomplete or stale coverage: %s to %s",sym,q.index.min() if len(q) else None,q.index.max() if len(q) else None)
            except Exception as exc:
                logger.warning("Yahoo max-history fallback failed for %s: %s",sym,exc)
            # Independent fallbacks: Stooq daily history for SPY/QQQ, CBOE official VIX CSV.
            try:
                if sym in ("SPY","QQQ"):
                    response=requests.get("https://stooq.com/q/d/l/",params={"s":sym.lower()+".us","i":"d"},
                        headers={"User-Agent":"Mozilla/5.0"},timeout=18)
                    response.raise_for_status()
                    d=pd.read_csv(io.StringIO(response.text))
                    if {"Date","Close"}.issubset(d.columns):
                        ix=pd.to_datetime(d["Date"],errors="coerce").dt.normalize()
                        q=pd.Series(pd.to_numeric(d["Close"],errors="coerce").to_numpy(),index=ix).dropna()
                        q=q[~q.index.isna()]
                        q=q[~q.index.duplicated(keep="last")].sort_index()
                        expected_start=pd.Timestamp("2000-01-01") if sym=="SPY" else pd.Timestamp("1999-03-01")
                        if len(q)>=250 and q.index.min()<=expected_start+pd.Timedelta(days=400) and q.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                            source_map[sym]="Stooq daily history fallback"
                            return sym,q
                        logger.warning("Stooq long history for %s has incomplete or stale coverage: %s to %s",sym,q.index.min() if len(q) else None,q.index.max() if len(q) else None)
                        logger.warning("Stooq history for %s is recent-only or incomplete: %s to %s (%d rows)",sym,q.index.min() if len(q) else None,q.index.max() if len(q) else None,len(q))
                else:
                    response=requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                        headers={"User-Agent":"Mozilla/5.0"},timeout=18)
                    response.raise_for_status()
                    d=pd.read_csv(io.StringIO(response.text))
                    dc=next((c for c in d.columns if c.strip().lower()=="date"),None)
                    cc=next((c for c in d.columns if c.strip().lower() in ("close","vix close")),None)
                    if dc and cc:
                        ix=pd.to_datetime(d[dc],errors="coerce").dt.normalize()
                        q=pd.Series(pd.to_numeric(d[cc],errors="coerce").to_numpy(),index=ix).dropna()
                        q=q[~q.index.isna()]
                        q=q[~q.index.duplicated(keep="last")].sort_index()
                        if len(q)>=250:
                            source_map[sym]="CBOE official VIX history fallback"
                            return sym,q
            except Exception as exc:
                logger.warning("Independent long-history fallback failed for %s: %s",sym,exc)
            raise RuntimeError("Long-run history unavailable for "+sym+" (providers and repository baseline failed)")
        if p is None:
            with ThreadPoolExecutor(max_workers=3) as pool:
                series=dict(pool.map(_load_long,("SPY","QQQ","^VIX")))
            p=pd.concat(series,axis=1).sort_index()
            p=p[~p.index.duplicated(keep="last")]
            # Persist only after source coverage is verified across all crash eras.
            if not p.empty and p.index.min()<=required_start+pd.Timedelta(days=400) and p.index.max()>=pd.Timestamp.now().normalize()-pd.Timedelta(days=10):
                try:
                    os.makedirs(os.path.dirname(cache_path),exist_ok=True)
                    p[["SPY","QQQ","^VIX"]].rename_axis("date").to_csv(cache_path,float_format="%.8g")
                    logger.info("Saved verified historical validation cache: %s (%s to %s; %d rows)",cache_path,p.index.min(),p.index.max(),len(p))
                except Exception as exc:
                    logger.warning("Could not persist historical validation cache: %s",exc)
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
        # FRED macro series can be missing on hosted deployments even when market history is complete.
        # Renormalize weights over the components available on each date instead of dropping decades.
        score_parts=pd.DataFrame({"credit":credit,"volatility":vol,"momentum":momentum,"drawdown":drawdown,"liquidity":liquidity},index=p.index)
        score_weights=pd.Series({"credit":.30,"volatility":.20,"momentum":.25,"drawdown":.15,"liquidity":.10})
        weighted=score_parts.mul(score_weights,axis=1)
        available_weights=score_parts.notna().mul(score_weights,axis=1).sum(axis=1)
        p["replay_score"]=weighted.sum(axis=1,min_count=1).div(available_weights.replace(0,np.nan))
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
        payload={"available":True,"episodes":results,"method":"Time-varying broad-US-market proxy: high-yield credit spreads, official VIX, QQQ 6M momentum, SPY 1Y drawdown and NFCI. 60/100 is exploratory, not calibrated.",
          "coverage_start":p.index.min().strftime("%Y-%m-%d") if not p.empty else None,
          "coverage_end":p.index.max().strftime("%Y-%m-%d") if not p.empty else None,
          "data_sources":source_map,"observations":int(len(p)),
          "threshold":60,"warning":"Broad US market stress replay, not an AI-specific model or calibrated crash probability. Retrospective threshold check, not out-of-sample validation."}
    except Exception as exc:
        logger.warning("Historical crash replay unavailable: %s",exc)
        payload={"available":False,"episodes":[],"warning":"Historical replay unavailable: "+str(exc)[:220],
          "method":"Requires long-run SPY, QQQ, VIX and historical macro series."}
    _HIST_CACHE["payload"]=payload; _HIST_CACHE["ts"]=now
    return payload


def _historical_production_replay(fred_data):
    """Replay the production crash score day-by-day when all production inputs are historically available.

    Uses the same component formulas and weights as _score(). Historical AI Fundamental /
    AI Financing inputs are reconstructed from SEC XBRL quarterly/annual facts where possible.
    This is a period-end financial replay, not yet a filing-date point-in-time backtest.
    """
    now=time.time()
    cached=_PROD_CACHE["payload"]
    cache_ttl=HIST_CACHE_TTL if cached and cached.get("available") else 15*60
    if cached is not None and now-_PROD_CACHE["ts"]<cache_ttl:
        return cached
    try:
        from concurrent.futures import ThreadPoolExecutor
        source_map={}
        market_symbols=("SPY","QQQ","RSP","IWM","SOXX","^VIX")
        cik_map={"MSFT":"0000789019","GOOGL":"0001652044","AMZN":"0001018724","META":"0001326801","ORCL":"0001341439"}

        def _load_long(sym):
            try:
                chart=yahoo_get_chart(sym,interval="1d",period="max")
                if chart and chart.get("timestamps") and chart.get("close"):
                    ix=pd.to_datetime(chart["timestamps"],unit="s",utc=True,errors="coerce").tz_localize(None).normalize()
                    q=pd.Series(pd.to_numeric(chart["close"],errors="coerce"),index=ix).dropna()
                    q=q[~q.index.isna()]
                    q=q[~q.index.duplicated(keep="last")].sort_index()
                    if len(q)>=250:
                        source_map[sym]="Yahoo Finance max history"
                        return sym,q
            except Exception as exc:
                logger.warning("Yahoo long history failed for %s: %s",sym,exc)
            if sym in ("SPY","QQQ"):
                try:
                    response=requests.get("https://stooq.com/q/d/l/",params={"s":sym.lower()+".us","i":"d"},
                        headers={"User-Agent":"Mozilla/5.0"},timeout=18)
                    response.raise_for_status()
                    d=pd.read_csv(io.StringIO(response.text))
                    if {"Date","Close"}.issubset(d.columns):
                        ix=pd.to_datetime(d["Date"],errors="coerce").dt.normalize()
                        q=pd.Series(pd.to_numeric(d["Close"],errors="coerce").to_numpy(),index=ix).dropna()
                        q=q[~q.index.isna()]
                        q=q[~q.index.duplicated(keep="last")].sort_index()
                        if len(q)>=250:
                            source_map[sym]="Stooq daily history fallback"
                            return sym,q
                except Exception as exc:
                    logger.warning("Stooq long-history fallback failed for %s: %s",sym,exc)
            if sym=="^VIX":
                try:
                    response=requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                        headers={"User-Agent":"Mozilla/5.0"},timeout=18)
                    response.raise_for_status()
                    d=pd.read_csv(io.StringIO(response.text))
                    dc=next((c for c in d.columns if c.strip().lower()=="date"),None)
                    cc=next((c for c in d.columns if c.strip().lower() in ("close","vix close")),None)
                    if dc and cc:
                        ix=pd.to_datetime(d[dc],errors="coerce").dt.normalize()
                        q=pd.Series(pd.to_numeric(d[cc],errors="coerce").to_numpy(),index=ix).dropna()
                        q=q[~q.index.isna()]
                        q=q[~q.index.duplicated(keep="last")].sort_index()
                        if len(q)>=250:
                            source_map[sym]="CBOE official VIX history fallback"
                            return sym,q
                except Exception as exc:
                    logger.warning("CBOE VIX fallback failed: %s",exc)
            raise RuntimeError("Long-run history unavailable for "+sym)

        with ThreadPoolExecutor(max_workers=2) as pool:
            series=dict(pool.map(_load_long,market_symbols))
        p=pd.concat(series,axis=1).sort_index()
        p=p[~p.index.duplicated(keep="last")]

        for key in ("hy_oas","nfci","dfii10","unrate"):
            z=fred_data.get(key,pd.Series(dtype=float)).copy()
            z.index=pd.to_datetime(z.index,errors="coerce").tz_localize(None).normalize()
            p[key]=pd.to_numeric(z.reindex(p.index).ffill(),errors="coerce")

        def _sec_fact_payload(ticker):
            url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_map[ticker]}.json"
            r=requests.get(url,headers={"User-Agent":"TV-GPT-ALP-V2 research contact"},timeout=20)
            r.raise_for_status()
            return r.json()

        def _pick_fact(facts,candidates):
            for tag in candidates:
                node=facts.get("us-gaap",{}).get(tag)
                if node and "USD" in node.get("units",{}):
                    return node["units"]["USD"],tag
            return None,None

        def _flow_quarters(rows):
            q={}; annual={}
            for x in rows or []:
                try:
                    if not x.get("start") or not x.get("end"): continue
                    st=pd.Timestamp(x["start"]).normalize(); en=pd.Timestamp(x["end"]).normalize()
                    dur=(en-st).days; val=float(x["val"]); filed=pd.Timestamp(x.get("filed","1900-01-01"))
                    if 70<=dur<=110:
                        old=q.get(en)
                        if old is None or filed>old[1]: q[en]=(val,filed)
                    elif 300<=dur<=380 and x.get("form") in ("10-K","20-F"):
                        old=annual.get(en)
                        if old is None or filed>old[1]: annual[en]=(val,filed)
                except Exception: continue
            qv={d:v[0] for d,v in q.items()}
            for yend,(av,_) in annual.items():
                qs=sorted([d for d in qv if d<yend])
                if len(qs)>=3: qv[yend]=av-sum(qv[d] for d in qs[-3:])
            return pd.Series(qv,dtype=float).sort_index()

        def _instant_series(rows):
            out={}
            for x in rows or []:
                try:
                    en=pd.Timestamp(x["end"]).normalize(); val=float(x["val"]); filed=pd.Timestamp(x.get("filed","1900-01-01"))
                    old=out.get(en)
                    if old is None or filed>old[1]: out[en]=(val,filed)
                except Exception: continue
            return pd.Series({d:v[0] for d,v in out.items()},dtype=float).sort_index()

        def _company_snapshot(ticker):
            facts=_sec_fact_payload(ticker)["facts"]
            rev_rows,_=_pick_fact(facts,["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues","SalesRevenueNet"])
            cap_rows,_=_pick_fact(facts,["PaymentsToAcquirePropertyPlantAndEquipment","PaymentsToAcquirePropertyPlantAndEquipmentGross"])
            if rev_rows is None or cap_rows is None: return pd.DataFrame()
            rev=_flow_quarters(rev_rows); cap=_flow_quarters(cap_rows).abs()
            pieces=[]
            for tag in ("LongTermDebtAndFinanceLeaseObligations","LongTermDebtAndFinanceLeaseObligationsCurrent",
                        "LongTermDebtCurrent","LongTermDebtAndFinanceLeaseObligationsNoncurrent","LongTermDebtNoncurrent"):
                node=facts.get("us-gaap",{}).get(tag)
                if node and "USD" in node.get("units",{}): pieces.append(_instant_series(node["units"]["USD"]).rename(tag))
            debt=pd.concat(pieces,axis=1).sum(axis=1,min_count=1) if pieces else pd.Series(dtype=float)
            rows=[]
            for d in sorted(set(rev.index)|set(cap.index)):
                r=rev.loc[:d].tail(4); c=cap.loc[:d].tail(4)
                if len(r)<4 or len(c)<4: continue
                prev_r=rev.loc[:d].iloc[:-4].tail(4); prev_c=cap.loc[:d].iloc[:-4].tail(4)
                if len(prev_r)<4 or len(prev_c)<4: continue
                dn=debt.loc[:d].dropna()
                if len(dn)<5: continue
                rt=float(r.sum()); ct=float(c.sum()); rp=float(prev_r.sum()); cp=float(prev_c.sum())
                dn_now=float(dn.iloc[-1]); dn_prev=float(dn.iloc[-5])
                rows.append({"date":d,"capex_revenue":ct/rt*100 if rt>0 else None,
                    "capex_growth_gap":((ct/cp)-1)*100-((rt/rp)-1)*100 if cp>0 and rp>0 else None,
                    "debt_growth_gap":((dn_now/dn_prev)-1)*100-((rt/rp)-1)*100 if dn_prev>0 and rp>0 else None})
            return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

        with ThreadPoolExecutor(max_workers=1) as pool:
            fund_results=dict(zip(cik_map.keys(),pool.map(_company_snapshot,cik_map.keys())))
        fund_parts={k:v for k,v in fund_results.items() if v is not None and not v.empty}
        if not fund_parts: raise RuntimeError("SEC historical hyperscaler fundamentals unavailable")
        fund=pd.concat(fund_parts,axis=1)
        fund.columns=pd.MultiIndex.from_tuples(fund.columns)
        fund_agg=pd.DataFrame(index=fund.index)
        for metric in ("capex_revenue","capex_growth_gap","debt_growth_gap"):
            cols=[c for c in fund.columns if c[1]==metric]
            fund_agg[metric]=fund[cols].mean(axis=1,min_count=3)
        fund_agg=fund_agg.sort_index().ffill()

        hy=p["hy_oas"]; hy_med=hy.rolling(252,min_periods=60).median(); hy_chg=hy.diff(20)
        credit=(0.65*((hy/hy_med.clip(lower=.5)-.85)*100)+0.35*(hy_chg.fillna(0)*35)).clip(0,100)
        nf=p["nfci"]; nf_mean=nf.rolling(104,min_periods=26).mean(); nf_std=nf.rolling(104,min_periods=26).std()
        liquidity=(((nf-nf_mean)/nf_std.clip(lower=.15)+.5)*35).clip(0,100)
        real=p["dfii10"]; real_score=(((real-1)*22)+real.diff(60).clip(lower=0)*10).clip(0,100)
        vol=((p["^VIX"]-16)*4).clip(0,100)
        breadth=pd.concat([(50-(p[sym]/p["SPY"]).pct_change(63)*2.5).clip(0,100) for sym in ("RSP","IWM","SOXX")],axis=1).mean(axis=1)
        un=p["unrate"]; m=un.rolling(3).mean(); sahm=m-m.rolling(12).min()
        recession=(sahm*120).clip(0,100)
        recession=(recession+un.diff(3)*18).clip(0,100)

        cgap=fund_agg["capex_growth_gap"].reindex(p.index).ffill()
        dgap=fund_agg["debt_growth_gap"].reindex(p.index).ffill()
        ratio=fund_agg["capex_revenue"].reindex(p.index).ffill()
        ai=(35+cgap.mul(2).clip(-20,35)+(ratio.sub(15)*1.5).clip(-15,25)).clip(0,100)
        afin=(35+dgap.mul(2.5).clip(-15,45)).clip(0,100)
        ai_fund=(0.65*ai+0.35*afin).clip(0,100)

        production=(.18*ai_fund+.14*afin+.20*credit+.10*liquidity+.14*breadth+
                    .10*real_score+.05*vol+.09*recession).clip(0,100)
        p["production_score"]=production
        p=p.dropna(subset=["SPY","QQQ","RSP","IWM","SOXX","^VIX","hy_oas","nfci","dfii10","unrate","production_score"])

        episodes=[
          {"name":"Dot-com bust","symbol":"QQQ","peak":"2000-03-10"},
          {"name":"Global financial crisis","symbol":"SPY","peak":"2007-10-09"},
          {"name":"COVID shock","symbol":"SPY","peak":"2020-02-19"},
          {"name":"2022 bear market","symbol":"SPY","peak":"2022-01-03"},
        ]
        results=[]
        for e in episodes:
            s=e["symbol"]; peak_date=pd.Timestamp(e["peak"]); q=p[s].dropna()
            if q.empty or q.index.min()>peak_date or q.index.max()<peak_date:
                results.append({"episode":e["name"],"peak_date":e["peak"],"status":"Insufficient exact-production coverage",
                    "coverage_start":p.index.min().strftime("%Y-%m-%d") if not p.empty else None})
                continue
            peak_ix=q.index[q.index.get_indexer([peak_date],method="nearest")[0]]; peak_price=float(q.loc[peak_ix])
            future=q.loc[peak_ix:].iloc[1:253]; breach=future[future<=peak_price*.80]
            if breach.empty:
                results.append({"episode":e["name"],"peak_date":peak_ix.strftime("%Y-%m-%d"),"status":"20% threshold not found in 12M"})
                continue
            breach_date=breach.index[0]
            pre=p.loc[(p.index>=breach_date-pd.Timedelta(days=90))&(p.index<breach_date)]
            crossed=pre[pre["production_score"]>=60]; first=crossed.index[0] if not crossed.empty else None
            results.append({"episode":e["name"],"peak_date":peak_ix.strftime("%Y-%m-%d"),
              "20pct_date":breach_date.strftime("%Y-%m-%d"),"days_to_20pct":int((breach_date-peak_ix).days),
              "score_at_peak":round(float(p.loc[peak_ix,"production_score"]),1),
              "max_score_pre_breach":round(float(pre["production_score"].max()),1) if not pre.empty else None,
              "first_signal_date":first.strftime("%Y-%m-%d") if first is not None else None,
              "lead_days":int((breach_date-first).days) if first is not None else None,
              "status":"Signal before -20% threshold" if first is not None else "No 60+ signal in 90D pre-breach window"})

        payload={"available":True,"episodes":results,"threshold":60,"exact_production":True,
          "method":"Production crash score replay: identical _score() formulas and weights evaluated day-by-day; SEC XBRL period-end financial reconstruction for AI Fundamental/AI Financing.",
          "production_formula":"AI Fundamental 18% + AI Financing 14% + Credit 20% + Liquidity 10% + Market Breadth 14% + Real Rates 10% + Volatility 5% + Recession 9%",
          "coverage_start":p.index.min().strftime("%Y-%m-%d") if not p.empty else None,
          "coverage_end":p.index.max().strftime("%Y-%m-%d") if not p.empty else None,
          "observations":int(len(p)),"data_sources":source_map,
          "fundamental_source":"SEC EDGAR XBRL companyfacts; period-end financial facts; aggregate requires at least 3 hyperscalers",
          "point_in_time":False,
          "warning":"Production-formula replay, not calibrated probability. Financial facts use reporting period end rather than filing/publication date, so this is not yet a strict point-in-time backtest."}
    except Exception as exc:
        logger.warning("Historical production-score replay unavailable: %s",exc)
        payload={"available":False,"episodes":[],"warning":"Historical production-score replay unavailable: "+str(exc)[:260],
          "method":"Requires historical SPY/QQQ/RSP/IWM/SOXX/VIX, macro series and SEC XBRL hyperscaler financial facts."}
    _PROD_CACHE["payload"]=payload; _PROD_CACHE["ts"]=now
    return payload


def start_production_replay(fred_data=None):
    """Run the expensive production replay only from an explicit background job."""
    global _PROD_THREAD
    with _PROD_LOCK:
        if _PROD_JOB.get("status") == "running":
            return False, "already_running"
        if _PROD_CACHE.get("payload") is not None:
            _PROD_JOB["status"] = "complete"
            _PROD_JOB["result"] = _PROD_CACHE["payload"]
            return False, "cached"
        _PROD_JOB.update({
            "status":"running","result":None,"error":None,
            "started_at":datetime.now(timezone.utc).isoformat(),"finished_at":None
        })
        def _worker():
            try:
                replay_fred=fred_data
                if replay_fred is None:
                    replay_fred={k:_fred(v) for k,v in FRED.items()}
                result=_historical_production_replay(replay_fred)
                with _PROD_LOCK:
                    _PROD_JOB["result"]=result
                    _PROD_JOB["status"]="complete" if result.get("available") else "failed"
                    _PROD_JOB["error"]=None if result.get("available") else result.get("warning")
                    _PROD_JOB["finished_at"]=datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                logger.exception("Production crash replay worker failed")
                with _PROD_LOCK:
                    _PROD_JOB["status"]="failed"
                    _PROD_JOB["error"]=str(exc)[:260]
                    _PROD_JOB["finished_at"]=datetime.now(timezone.utc).isoformat()
        _PROD_THREAD=threading.Thread(target=_worker,name="ai-crash-production-replay",daemon=True)
        _PROD_THREAD.start()
        return True, "started"


def production_replay_status():
    with _PROD_LOCK:
        result=_PROD_JOB.get("result")
        if result is None and _PROD_CACHE.get("payload") is not None:
            result=_PROD_CACHE["payload"]
        return {
            "status":_PROD_JOB.get("status","idle"),
            "started_at":_PROD_JOB.get("started_at"),
            "finished_at":_PROD_JOB.get("finished_at"),
            "error":_PROD_JOB.get("error"),
            "result":result,
        }

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
        fund_rows=fund.get("rows",[]) or []
        fundamental_count=int(fund.get("companies_with_complete_statements",sum(1 for row in fund_rows if row.get("coverage_status")=="COMPLETE")))
        revenue_count=int(fund.get("companies_with_revenue",sum(1 for row in fund_rows if row.get("revenue_ttm") is not None)))
        metric_coverage=fund.get("metric_coverage",{})
        period_dates=[pd.to_datetime(row.get("revenue_period_end"),errors="coerce") for row in fund_rows if row.get("revenue_period_end")]
        latest_fund_period=max((d for d in period_dates if pd.notna(d)),default=None)
        fundamental_period_age_days=max(0,int((pd.Timestamp.now().normalize()-latest_fund_period.normalize()).days)) if latest_fund_period is not None else None
        quality_warnings=[]
        if market_age_days>5: quality_warnings.append("Market prices may be stale.")
        if fundamental_count<len(HYPERSCALERS): quality_warnings.append("Complete hyperscaler financial coverage is "+str(fundamental_count)+"/"+str(len(HYPERSCALERS))+"; missing metrics use neutral defaults.")
        if fundamental_period_age_days is None or fundamental_period_age_days>180: quality_warnings.append("Hyperscaler financial period-end data is missing or older than 180 days; verify latest filings.")
        missing_macro=[k for k,v in f.items() if v is None or len(v.dropna())==0]
        if missing_macro: quality_warnings.append("Missing macro series: "+", ".join(missing_macro))
        volatility_source = core["details"].get("volatility_source", "unknown")
        if "VIXY ETF proxy" in volatility_source:
            quality_warnings.append("Official VIX index unavailable; using Alpaca VIXY ETF proxy. VIXY is not the VIX index.")
        # Reliability is separate from the risk score: a high score with weak/stale inputs
        # must not be presented as equally trustworthy. This is a data-confidence measure,
        # not a probability of a crash.
        market_conf=100.0 if market_age_days<=1 else 90.0 if market_age_days<=3 else 70.0 if market_age_days<=5 else 35.0
        macro_conf=100.0*(len(f)-len(missing_macro))/max(len(FRED),1)
        fund_conf={0:0.0,1:30.0,2:50.0,3:70.0,4:85.0,5:100.0}.get(min(fundamental_count,5),100.0)
        if fundamental_period_age_days is None: fund_conf=min(fund_conf,10.0)
        elif fundamental_period_age_days>180: fund_conf=min(fund_conf,35.0)
        elif fundamental_period_age_days>120: fund_conf=min(fund_conf,60.0)
        vol_conf=100.0 if "VIXY ETF proxy" not in volatility_source else 60.0
        data_reliability=round(_clip(.35*market_conf+.25*macro_conf+.25*fund_conf+.15*vol_conf),1)
        v2=_v2_overlay(s,comps,hist)
        signal_confidence=round(_clip(.60*float(v2["persistence_20d"])+.40*float(v2["confirmation_score"])),1)
        reliability_label="HIGH" if data_reliability>=80 else "MEDIUM" if data_reliability>=60 else "LOW"
        data_quality={"market_latest_date":market_date.strftime("%Y-%m-%d"),"market_age_days":market_age_days,
          "volatility_source":volatility_source,
          "fundamental_companies_covered":fundamental_count,"fundamental_companies_with_revenue":revenue_count,
          "fundamental_companies_expected":len(HYPERSCALERS),"fundamental_period_end_latest":latest_fund_period.strftime("%Y-%m-%d") if latest_fund_period is not None else None,
          "fundamental_period_age_days":fundamental_period_age_days,"fundamental_metric_coverage":metric_coverage,
          "macro_series_covered":len(f)-len(missing_macro),"macro_series_expected":len(FRED),
          "warnings":quality_warnings}
        # Separate broad-US-market risk from AI-specific concentration/funding risk.
        market_weights={"Credit":.27,"Liquidity":.14,"Market Breadth":.22,"Real Rates":.14,"Volatility":.10,"Recession":.13}
        market_score=_clip(sum(comps[k]*w for k,w in market_weights.items()))
        market_drivers=[k for k,v in sorted(((k,comps[k]) for k in market_weights),key=lambda x:x[1],reverse=True)[:3] if v>=40]
        payload={"as_of":datetime.now(timezone.utc).isoformat(),"score":s,"regime":_regime(s),"drivers":drivers,
          "us_market":{"score":round(market_score,1),"regime":_regime(market_score),"drivers":market_drivers,
            "components":{k:round(comps[k],1) for k in market_weights},
            "method":"Separate broad-market index using credit, liquidity, breadth, real rates, volatility and recession factors; not a calibrated probability."},
          "components":factors,"confirmations":{"credit":comps["Credit"]>=70,"recession":comps["Recession"]>=70,
          "breadth":comps["Market Breadth"]>=70,"liquidity":comps["Liquidity"]>=70},
          "details":core["details"],"fundamentals":fund,"data_quality":data_quality,"history":hist,
          "reliability":{"data_score":data_reliability,"data_label":reliability_label,"signal_confidence":signal_confidence,
            "note":"Data score measures input quality; signal confidence requires persistent or multi-block stress. Neither is a calibrated crash probability."},
          "v2":v2,
          "historical_replay":_historical_crash_replay(f),
          "methodology":{"weights":{"AI Fundamental":18,"AI Financing":14,"Credit":20,"Liquidity":10,"Market Breadth":14,"Real Rates":10,"Volatility":5,"Recession":9},
          "note":"Early-warning monitor, not a crash-date predictor. Hyperscaler capex is a proxy, not AI-only capex."}}
        _CACHE["payload"]=payload; _CACHE["ts"]=now; return payload
