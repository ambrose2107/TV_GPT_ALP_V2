"""Pullback Breakout V2 research engine.

The public functions still return pandas objects for the Strategy Lab UI, but
optimizer/backtest hot paths use compact NumPy arrays and avoid constructing
trade/equity DataFrames unless details are explicitly requested.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from research.xauusd_pullback_v1 import atr


@dataclass
class PullbackV2Config:
    ema_fast:int=50; ema_slow:int=200; atr_len:int=14; pullback_bars:int=3; breakout_lookback:int=5
    atr_stop:float=1.2; rr:float=2.0; risk_pct:float=.5; cooldown_bars:int=8
    session_start_utc:int=13; session_end_utc:int=21; min_ema_gap_atr:float=.10
    breakout_buffer_atr:float=.15; min_body_atr:float=.40; min_atr_pct:float=.00125; max_atr_pct:float=.02


def _ema_np(values, span):
    # Pandas' ewm is implemented in optimized native code and is substantially
    # cheaper than a Python EMA loop.
    return pd.Series(values).ewm(span=int(span), adjust=False).mean().to_numpy(dtype=np.float32)


def _rolling_mean_np(values, n):
    n=int(n)
    out=np.full(len(values), np.nan, dtype=np.float32)
    if n<=0 or len(values)<n:
        return out
    cs=np.cumsum(values, dtype=np.float64)
    out[n-1:]=((cs[n-1:] - np.r_[0.0, cs[:-n]]) / n).astype(np.float32)
    return out


def _rolling_sum_bool(values, n):
    return _rolling_mean_np(values.astype(np.float32), n) * n


def _rolling_extreme(values, n, is_max=True):
    # Optimizer windows are small (normally 4-6). sliding_window_view keeps
    # this operation in NumPy/C without pandas Series/DataFrame allocation.
    n=int(n)
    out=np.full(len(values), np.nan, dtype=np.float32)
    if n<=0 or len(values)<n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    w=sliding_window_view(values, n)
    out[n-1:]=(np.max(w,axis=1) if is_max else np.min(w,axis=1)).astype(np.float32)
    return out


def _signal_arrays(df, cfg):
    """Return compact arrays needed by the backtester."""
    o=df["Open"].to_numpy(dtype=np.float32, copy=False)
    h=df["High"].to_numpy(dtype=np.float32, copy=False)
    l=df["Low"].to_numpy(dtype=np.float32, copy=False)
    c=df["Close"].to_numpy(dtype=np.float32, copy=False)

    ef=_ema_np(c,cfg.ema_fast)
    es=_ema_np(c,cfg.ema_slow)

    prev_c=np.empty_like(c)
    prev_c[0]=c[0]
    prev_c[1:]=c[:-1]
    tr=np.maximum.reduce((h-l,np.abs(h-prev_c),np.abs(l-prev_c)))
    a=_rolling_mean_np(tr,cfg.atr_len)

    n_pull=int(cfg.pullback_bars)
    n_break=int(cfg.breakout_lookback)
    down=_rolling_sum_bool(c<o,n_pull)
    up=_rolling_sum_bool(c>o,n_pull)
    # Breakout levels must use completed bars only. Including the current bar
    # makes `close > current_high + buffer` impossible (and similarly for shorts).
    prior_high_raw=_rolling_extreme(h,n_break,True)
    prior_low_raw=_rolling_extreme(l,n_break,False)
    prior_high=np.empty_like(prior_high_raw); prior_high[:1]=np.nan; prior_high[1:]=prior_high_raw[:-1]
    prior_low=np.empty_like(prior_low_raw); prior_low[:1]=np.nan; prior_low[1:]=prior_low_raw[:-1]

    ap=a/np.maximum(np.abs(c),1e-9)
    body=np.divide(np.abs(c-o),a,out=np.full(len(c),np.nan,dtype=np.float32),where=a>0)

    idx=df.index
    hours=idx.hour.to_numpy()
    session=(hours>=cfg.session_start_utc)&(hours<cfg.session_end_utc)

    valid=(
        session & np.isfinite(a) & (a>0)
        & (ap>=cfg.min_atr_pct) & (ap<=cfg.max_atr_pct)
        & (np.abs(ef-es)/a>=cfg.min_ema_gap_atr)
        & (body>=cfg.min_body_atr)
    )
    long_mask=valid&(c>es)&(ef>es)&(down>=1)&(c>prior_high+cfg.breakout_buffer_atr*a)
    short_mask=valid&(c<es)&(ef<es)&(up>=1)&(c<prior_low-cfg.breakout_buffer_atr*a)

    signal=np.zeros(len(c),dtype=np.int8)
    signal[long_mask]=1
    signal[short_mask]=-1

    candidates=np.flatnonzero(signal)
    if cfg.cooldown_bars>0 and len(candidates)>1:
        keep=np.empty(len(candidates),dtype=np.int32)
        k=0; last=-10**9; cd=int(cfg.cooldown_bars)
        for i in candidates:
            if i-last>=cd:
                keep[k]=i; k+=1; last=i
        filtered=np.zeros_like(signal)
        kept=keep[:k]
        filtered[kept]=signal[kept]
        signal=filtered

    return o,h,l,c,a,signal


def _metrics_fast(df,cfg,initial_equity=10000.):
    """Metrics-only backtest. No pandas rows, trades or equity curve are built."""
    o,h,l,c,a,signal=_signal_arrays(df,cfg)
    n=len(c)
    eq=float(initial_equity)
    peak=eq
    max_dd=0.0
    pos_side=0
    entry=stop=target=0.0
    trades=0; wins=0; total_r=0.0; gross_win=0.0; gross_loss=0.0

    for i in range(n):
        if pos_side:
            if pos_side==1:
                hit_sl=l[i]<=stop
                hit_tp=h[i]>=target
            else:
                hit_sl=h[i]>=stop
                hit_tp=l[i]<=target
            if hit_sl or hit_tp:
                r=-1.0 if hit_sl else cfg.rr
                eq*=1.0+(cfg.risk_pct/100.0)*r
                trades+=1; total_r+=r
                if r>0: wins+=1; gross_win+=r
                else: gross_loss-=r
                if eq>peak: peak=eq
                dd=(eq/peak-1.0)*100.0
                if dd<max_dd: max_dd=dd
                pos_side=0

        if pos_side==0 and i+1<n and signal[i]:
            en=o[i+1]
            side=int(signal[i])
            st=c[i]-side*cfg.atr_stop*a[i]
            if (side==1 and st<en) or (side==-1 and st>en):
                pos_side=side; entry=en; stop=st
                target=en+side*cfg.rr*abs(en-st)

    pf=(gross_win/gross_loss) if gross_loss else (float("inf") if gross_win else 0.0)
    expectancy=(total_r/trades) if trades else 0.0
    return {
        "num_trades":int(trades),
        "win_rate":round((wins/trades)*100,2) if trades else 0.0,
        "profit_factor":round(pf,2) if np.isfinite(pf) else float("inf"),
        "expectancy_R":round(expectancy,3),
        "total_R":round(total_r,2),
        "max_drawdown_pct":round(max_dd,2),
        "total_return_pct":round((eq/initial_equity-1)*100,2),
        "final_equity":round(eq,2),
    }


def build_signals(df,cfg=PullbackV2Config()):
    """UI/debug representation. Kept separate from the optimizer hot path."""
    o,h,l,c,a,signal=_signal_arrays(df,cfg)
    x=df.copy()
    x["ema_fast"]=pd.Series(_ema_np(c,cfg.ema_fast),index=x.index)
    x["ema_slow"]=pd.Series(_ema_np(c,cfg.ema_slow),index=x.index)
    x["atr"]=pd.Series(a,index=x.index)
    x["signal"]=signal
    x["sl"]=np.nan; x["tp"]=np.nan; x["reason"]=""

    idx=np.flatnonzero(signal)
    if len(idx):
        sl=np.full(len(x),np.nan,dtype=np.float32)
        tp=np.full(len(x),np.nan,dtype=np.float32)
        long_idx=idx[signal[idx]==1]; short_idx=idx[signal[idx]==-1]
        sl[long_idx]=c[long_idx]-cfg.atr_stop*a[long_idx]
        tp[long_idx]=c[long_idx]+cfg.rr*cfg.atr_stop*a[long_idx]
        sl[short_idx]=c[short_idx]+cfg.atr_stop*a[short_idx]
        tp[short_idx]=c[short_idx]-cfg.rr*cfg.atr_stop*a[short_idx]
        x["sl"]=sl; x["tp"]=tp
        x.loc[x.index[long_idx],"reason"]="trend-separated | pullback | displacement breakout"
        x.loc[x.index[short_idx],"reason"]="trend-separated | pullback | displacement breakout"
    return x


def backtest(df,cfg=PullbackV2Config(),initial_equity=10000.,details=True):
    """Backtest. Optimizer should use details=False to avoid DataFrame churn."""
    if not details:
        return {"signals":None,"trades":None,"equity_curve":None,
                "metrics":_metrics_fast(df,cfg,initial_equity)}

    s=build_signals(df,cfg)
    eq=float(initial_equity); pos=None; trades=[]; curve=[]
    for i in range(len(s)):
        row=s.iloc[i]; ts=s.index[i]
        if pos:
            hi,lo=float(row.High),float(row.Low)
            sl=lo<=pos["sl"] if pos["side"]==1 else hi>=pos["sl"]
            tp=hi>=pos["tp"] if pos["side"]==1 else lo<=pos["tp"]
            if sl or tp:
                r=-1. if sl else cfg.rr
                eq*=1+cfg.risk_pct/100*r
                trades.append({"entry_time":str(pos["entry_time"]),"exit_time":str(ts),"side":pos["side"],
                               "entry":pos["entry"],"exit_price":pos["sl"] if sl else pos["tp"],
                               "sl":pos["sl"],"tp":pos["tp"],"R":r,"equity":eq,
                               "reason":pos["reason"]+" | "+("SL" if sl else "TP")})
                pos=None
        if pos is None and i+1<len(s) and int(row.signal)!=0:
            nxt=s.iloc[i+1];entry=float(nxt.Open);side=int(row.signal);stop=float(row.sl)
            if (side==1 and stop<entry) or (side==-1 and stop>entry):
                pos={"side":side,"entry_time":s.index[i+1],"entry":entry,"sl":stop,
                     "tp":entry+side*cfg.rr*abs(entry-stop),"reason":str(row.reason)}
        curve.append(eq)

    t=pd.DataFrame(trades)
    e=pd.Series(curve,index=s.index[:len(curve)],dtype=float)
    if t.empty:
        m={"num_trades":0,"win_rate":0.,"profit_factor":0.,"expectancy_R":0.,"total_R":0.,"max_drawdown_pct":0.,"total_return_pct":0.,"final_equity":eq}
    else:
        gw=float(t.loc[t.R>0,"R"].sum()); gl=float(-t.loc[t.R<0,"R"].sum())
        m={"num_trades":len(t),"win_rate":round(float((t.R>0).mean()*100),2),
           "profit_factor":round(gw/gl,2) if gl else float("inf"),
           "expectancy_R":round(float(t.R.mean()),3),"total_R":round(float(t.R.sum()),2),
           "max_drawdown_pct":round(float((e/e.cummax()-1).min()*100),2),
           "total_return_pct":round((eq/initial_equity-1)*100,2),"final_equity":round(eq,2)}
    return {"signals":s,"trades":t,"equity_curve":e,"metrics":m}
