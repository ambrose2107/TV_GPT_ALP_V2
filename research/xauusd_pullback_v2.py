"""Pullback Breakout V2 research candidate."""
from dataclasses import dataclass
import numpy as np, pandas as pd
from research.xauusd_pullback_v1 import atr
@dataclass
class PullbackV2Config:
    ema_fast:int=50; ema_slow:int=200; atr_len:int=14; pullback_bars:int=3; breakout_lookback:int=5
    atr_stop:float=1.2; rr:float=2.0; risk_pct:float=.5; cooldown_bars:int=8
    session_start_utc:int=13; session_end_utc:int=21; min_ema_gap_atr:float=.10
    breakout_buffer_atr:float=.15; min_body_atr:float=.40; min_atr_pct:float=.00125; max_atr_pct:float=.02
def build_signals(df,cfg=PullbackV2Config()):
    """Build Pullback V2 signals with vectorized indicator/condition work.

    The old implementation evaluated every candle in Python for every optimizer
    candidate. On the small Render instance that became the dominant CPU cost.
    Rolling/EMA calculations below preserve the same signal rules while leaving
    only the cooldown pass as a small loop over actual signal candidates.
    """
    x=df.copy()
    close=x["Close"]
    open_=x["Open"]

    x["ema_fast"]=close.ewm(span=cfg.ema_fast,adjust=False).mean()
    x["ema_slow"]=close.ewm(span=cfg.ema_slow,adjust=False).mean()
    x["atr"]=atr(x,cfg.atr_len)

    n_pull=int(cfg.pullback_bars)
    n_break=int(cfg.breakout_lookback)

    # Only completed candles immediately before the signal candle are used.
    down=(close<open_).rolling(n_pull,min_periods=n_pull).sum().shift(1)
    up=(close>open_).rolling(n_pull,min_periods=n_pull).sum().shift(1)
    prior_high=x["High"].rolling(n_break,min_periods=n_break).max().shift(1)
    prior_low=x["Low"].rolling(n_break,min_periods=n_break).min().shift(1)

    a=x["atr"]
    ef=x["ema_fast"]
    es=x["ema_slow"]
    ap=a/close.abs().clip(lower=1e-9)
    body=(close-open_).abs()/a.replace(0,np.nan)
    hours=x.index.hour.to_numpy()
    session=(hours>=cfg.session_start_utc)&(hours<cfg.session_end_utc)

    valid=(
        session
        & a.notna().to_numpy()
        & (a.to_numpy()>0)
        & (ap.to_numpy()>=cfg.min_atr_pct)
        & (ap.to_numpy()<=cfg.max_atr_pct)
        & ((ef-es).abs()/a>=cfg.min_ema_gap_atr)
        & (body>=cfg.min_body_atr)
    )

    long_mask=(
        valid
        & (close.to_numpy()>es.to_numpy())
        & (ef.to_numpy()>es.to_numpy())
        & (down.to_numpy()>=1)
        & (close.to_numpy()>prior_high.to_numpy()+cfg.breakout_buffer_atr*a.to_numpy())
    )
    short_mask=(
        valid
        & (close.to_numpy()<es.to_numpy())
        & (ef.to_numpy()<es.to_numpy())
        & (up.to_numpy()>=1)
        & (close.to_numpy()<prior_low.to_numpy()-cfg.breakout_buffer_atr*a.to_numpy())
    )

    signal=np.zeros(len(x),dtype=np.int8)
    signal[long_mask]=1
    signal[short_mask]=-1

    # Cooldown is stateful by design. Iterate only over candidate signals,
    # not over every candle.
    candidates=np.flatnonzero(signal)
    if cfg.cooldown_bars>0 and len(candidates):
        keep=[]
        last=-10**9
        cooldown=int(cfg.cooldown_bars)
        for i in candidates:
            if i-last>=cooldown:
                keep.append(i)
                last=i
        filtered=np.zeros_like(signal)
        filtered[np.asarray(keep,dtype=int)]=signal[np.asarray(keep,dtype=int)]
        signal=filtered

    x["signal"]=signal
    x["sl"]=np.nan
    x["tp"]=np.nan
    x["reason"]=""

    idx=np.flatnonzero(signal)
    if len(idx):
        av=a.to_numpy()
        cv=close.to_numpy()
        sl=np.full(len(x),np.nan,dtype=float)
        tp=np.full(len(x),np.nan,dtype=float)
        sl[idx]=cv[idx]-signal[idx]*0.0  # initialize without branching
        long_idx=idx[signal[idx]==1]
        short_idx=idx[signal[idx]==-1]
        sl[long_idx]=cv[long_idx]-cfg.atr_stop*av[long_idx]
        tp[long_idx]=cv[long_idx]+cfg.rr*cfg.atr_stop*av[long_idx]
        sl[short_idx]=cv[short_idx]+cfg.atr_stop*av[short_idx]
        tp[short_idx]=cv[short_idx]-cfg.rr*cfg.atr_stop*av[short_idx]
        x["sl"]=sl
        x["tp"]=tp
        reasons=np.empty(len(x),dtype=object)
        reasons[:]=""
        reasons[long_idx]="trend-separated | pullback | displacement breakout"
        reasons[short_idx]="trend-separated | pullback | displacement breakout"
        x["reason"]=reasons

    return x

def backtest(df,cfg=PullbackV2Config(),initial_equity=10000.):
    s=build_signals(df,cfg);eq=float(initial_equity);pos=None;trades=[];curve=[]
    for i in range(len(s)):
        row=s.iloc[i];ts=s.index[i]
        if pos:
            hi,lo=float(row.High),float(row.Low);sl=lo<=pos["sl"] if pos["side"]==1 else hi>=pos["sl"];tp=hi>=pos["tp"] if pos["side"]==1 else lo<=pos["tp"]
            if sl or tp:
                r=-1. if sl else cfg.rr;eq*=1+cfg.risk_pct/100*r;trades.append({"entry_time":str(pos["entry_time"]),"exit_time":str(ts),"side":pos["side"],"entry":pos["entry"],"exit_price":pos["sl"] if sl else pos["tp"],"sl":pos["sl"],"tp":pos["tp"],"R":r,"equity":eq,"reason":pos["reason"]+" | "+("SL" if sl else "TP")});pos=None
        if pos is None and i+1<len(s) and int(row.signal)!=0:
            nxt=s.iloc[i+1];entry=float(nxt.Open);side=int(row.signal);stop=float(row.sl)
            if (side==1 and stop<entry) or (side==-1 and stop>entry):pos={"side":side,"entry_time":s.index[i+1],"entry":entry,"sl":stop,"tp":entry+side*cfg.rr*abs(entry-stop),"reason":str(row.reason)}
        curve.append(eq)
    t=pd.DataFrame(trades);e=pd.Series(curve,index=s.index[:len(curve)],dtype=float)
    if t.empty:m={"num_trades":0,"win_rate":0.,"profit_factor":0.,"expectancy_R":0.,"total_R":0.,"max_drawdown_pct":0.,"total_return_pct":0.,"final_equity":eq}
    else:
        gw=float(t.loc[t.R>0,"R"].sum());gl=float(-t.loc[t.R<0,"R"].sum());m={"num_trades":len(t),"win_rate":round(float((t.R>0).mean()*100),2),"profit_factor":round(gw/gl,2) if gl else float("inf"),"expectancy_R":round(float(t.R.mean()),3),"total_R":round(float(t.R.sum()),2),"max_drawdown_pct":round(float((e/e.cummax()-1).min()*100),2),"total_return_pct":round((eq/initial_equity-1)*100,2),"final_equity":round(eq,2)}
    return {"signals":s,"trades":t,"equity_curve":e,"metrics":m}
