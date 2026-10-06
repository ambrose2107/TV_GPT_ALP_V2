"""Gold Aroon + Money Flow Confluence V2 research backtest."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AroonMoneyFlowV2Config:
    aroon_len: int = 14
    cmf_len: int = 20
    flow_buffer: float = 0.05
    confirm_bars: int = 5
    atr_len: int = 14
    atr_stop: float = 1.5
    rr: float = 2.0
    risk_pct: float = 0.5
    use_adx: bool = False
    adx_len: int = 14
    adx_min: float = 20.0
    cooldown_bars: int = 0
    long_only: bool = False


def _atr(df, n):
    h,l,c=df["High"],df["Low"],df["Close"]
    pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/int(n),adjust=False,min_periods=int(n)).mean()


def _aroon(df,n):
    n=int(n)
    high_age=df["High"].rolling(n+1,min_periods=n+1).apply(lambda x: len(x)-1-int(np.argmax(x)),raw=True)
    low_age=df["Low"].rolling(n+1,min_periods=n+1).apply(lambda x: len(x)-1-int(np.argmin(x)),raw=True)
    return 100.0*(n-high_age)/n, 100.0*(n-low_age)/n


def _cmf(df,n):
    hl=df["High"]-df["Low"]
    mfv=((2.0*df["Close"]-df["High"]-df["Low"])/hl.replace(0,np.nan))*df["Volume"]
    vol=df["Volume"].where(hl>0,0.0)
    return mfv.rolling(n,min_periods=n).sum()/vol.rolling(n,min_periods=n).sum()


def _adx(df,n):
    h,l,c=df["High"],df["Low"],df["Close"]
    up=h.diff(); down=-l.diff()
    plus=up.where((up>down)&(up>0),0.0)
    minus=down.where((down>up)&(down>0),0.0)
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1.0/n,adjust=False,min_periods=n).mean()
    pdi=100*plus.ewm(alpha=1.0/n,adjust=False,min_periods=n).mean()/atr
    mdi=100*minus.ewm(alpha=1.0/n,adjust=False,min_periods=n).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1.0/n,adjust=False,min_periods=n).mean()


def build_signals(df,cfg=AroonMoneyFlowV2Config()):
    x=df.copy()
    up,dn=_aroon(x,cfg.aroon_len)
    x["aroon_up"],x["aroon_down"]=up,dn
    x["aroon"]=up-dn
    x["cmf"]=_cmf(x,cfg.cmf_len)
    x["atr"]=_atr(x,cfg.atr_len)

    bull_flip=(x["aroon"]>0)&(x["aroon"].shift(1)<=0)
    bear_flip=(x["aroon"]<0)&(x["aroon"].shift(1)>=0)
    bull_id=bull_flip.cumsum()
    bear_id=bear_flip.cumsum()
    bull_age=x.groupby(bull_id).cumcount().where(bull_id>0)
    bear_age=x.groupby(bear_id).cumcount().where(bear_id>0)

    agree_bull=(x["aroon"]>0)&(x["cmf"]>cfg.flow_buffer)
    agree_bear=(x["aroon"]<0)&(x["cmf"]<-cfg.flow_buffer)
    long_sig=agree_bull&~agree_bull.shift(1).fillna(False)&(bull_age<=cfg.confirm_bars)
    short_sig=agree_bear&~agree_bear.shift(1).fillna(False)&(bear_age<=cfg.confirm_bars)

    if cfg.use_adx:
        long_sig &= _adx(x,cfg.adx_len).shift(1)>=cfg.adx_min
        short_sig &= _adx(x,cfg.adx_len).shift(1)>=cfg.adx_min
    volume_ok=x["Volume"].rolling(cfg.cmf_len,min_periods=cfg.cmf_len).sum()>0
    long_sig &= volume_ok
    short_sig &= volume_ok
    if cfg.long_only: short_sig &= False

    sig=np.zeros(len(x),dtype=np.int8)
    sig[long_sig.fillna(False).to_numpy()]=1
    sig[short_sig.fillna(False).to_numpy()]=-1
    if cfg.cooldown_bars>0:
        candidates=np.flatnonzero(sig)
        last=-10**9
        for i in candidates:
            if i-last<cfg.cooldown_bars: sig[i]=0
            else: last=i
    x["signal"]=sig
    x["sl"]=np.nan;x["tp"]=np.nan;x["reason"]=""
    x.loc[long_sig.fillna(False),"reason"]="Aroon bullish + CMF buying"
    x.loc[short_sig.fillna(False),"reason"]="Aroon bearish + CMF selling"
    x.loc[x["signal"]==1,"sl"]=x["Close"]-cfg.atr_stop*x["atr"]
    x.loc[x["signal"]==1,"tp"]=x["Close"]+cfg.rr*cfg.atr_stop*x["atr"]
    x.loc[x["signal"]==-1,"sl"]=x["Close"]+cfg.atr_stop*x["atr"]
    x.loc[x["signal"]==-1,"tp"]=x["Close"]-cfg.rr*cfg.atr_stop*x["atr"]
    return x


def backtest(df,cfg=AroonMoneyFlowV2Config(),initial_equity=10000.0,details=True):
    s=build_signals(df,cfg)
    eq=initial_equity; pos=None; trades=[]; curve=[]
    for i in range(len(s)):
        row=s.iloc[i]; ts=s.index[i]
        if pos:
            hi,lo=float(row.High),float(row.Low)
            hit_sl=lo<=pos["sl"] if pos["side"]==1 else hi>=pos["sl"]
            hit_tp=hi>=pos["tp"] if pos["side"]==1 else lo<=pos["tp"]
            if hit_sl or hit_tp:
                r=-1.0 if hit_sl else cfg.rr
                eq*=1+(cfg.risk_pct/100)*r
                trades.append({"entry_time":str(pos["entry_time"]),"exit_time":str(ts),"side":pos["side"],
                               "entry":pos["entry"],"exit_price":pos["sl"] if hit_sl else pos["tp"],
                               "sl":pos["sl"],"tp":pos["tp"],"R":r,"equity":eq,
                               "reason":pos["reason"]+" | "+("SL" if hit_sl else "TP")})
                pos=None
        if pos is None and i+1<len(s) and int(row.signal)!=0:
            nxt=s.iloc[i+1]; entry=float(nxt.Open); side=int(row.signal)
            stop=float(row.sl)
            if np.isfinite(stop) and ((side==1 and stop<entry) or (side==-1 and stop>entry)):
                pos={"side":side,"entry_time":s.index[i+1],"entry":entry,"sl":stop,
                     "tp":entry+side*cfg.rr*abs(entry-stop),"reason":str(row.reason)}
        curve.append(eq)
    t=pd.DataFrame(trades)
    e=pd.Series(curve,index=s.index,dtype=float)
    if t.empty:
        m={"num_trades":0,"win_rate":0.0,"profit_factor":0.0,"expectancy_R":0.0,"total_R":0.0,
           "max_drawdown_pct":0.0,"total_return_pct":0.0,"final_equity":round(eq,2)}
    else:
        gw=float(t.loc[t.R>0,"R"].sum()); gl=float(-t.loc[t.R<0,"R"].sum())
        m={"num_trades":len(t),"win_rate":round(float((t.R>0).mean()*100),2),
           "profit_factor":round(gw/gl,2) if gl else float("inf"),
           "expectancy_R":round(float(t.R.mean()),3),"total_R":round(float(t.R.sum()),2),
           "max_drawdown_pct":round(float((e/e.cummax()-1).min()*100),2),
           "total_return_pct":round((eq/initial_equity-1)*100,2),"final_equity":round(eq,2)}
    return {"signals":s,"trades":t,"equity_curve":e,"metrics":m}
