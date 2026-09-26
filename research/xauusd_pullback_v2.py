"""Pullback Breakout V2 research candidate."""
from dataclasses import dataclass
import numpy as np, pandas as pd
from research.xauusd_pullback_v1 import atr
@dataclass
class PullbackV2Config:
    ema_fast:int=50; ema_slow:int=200; atr_len:int=14; pullback_bars:int=3; breakout_lookback:int=5
    atr_stop:float=1.2; rr:float=2.0; risk_pct:float=.5; cooldown_bars:int=8
    session_start_utc:int=13; session_end_utc:int=21; min_ema_gap_atr:float=.10
    breakout_buffer_atr:float=.05; min_body_atr:float=.20; min_atr_pct:float=.0005; max_atr_pct:float=.02
def build_signals(df,cfg=PullbackV2Config()):
    x=df.copy();x["ema_fast"]=x.Close.ewm(span=cfg.ema_fast,adjust=False).mean();x["ema_slow"]=x.Close.ewm(span=cfg.ema_slow,adjust=False).mean();x["atr"]=atr(x,cfg.atr_len)
    x["signal"]=0;x["sl"]=np.nan;x["tp"]=np.nan;x["reason"]=""
    hours=x.index.hour.to_numpy();session=(hours>=cfg.session_start_utc)&(hours<cfg.session_end_utc)
    for i in range(max(cfg.ema_slow,cfg.breakout_lookback+cfg.pullback_bars+2,cfg.atr_len+2),len(x)):
        if not session[i]:continue
        a=float(x.atr.iloc[i]);close=float(x.Close.iloc[i]);ef=float(x.ema_fast.iloc[i]);es=float(x.ema_slow.iloc[i])
        if not np.isfinite(a) or a<=0:continue
        ap=a/max(abs(close),1e-9)
        if not(cfg.min_atr_pct<=ap<=cfg.max_atr_pct) or abs(ef-es)/a<cfg.min_ema_gap_atr:continue
        prev=x.iloc[i-cfg.pullback_bars:i];down=int((prev.Close<prev.Open).sum());up=int((prev.Close>prev.Open).sum())
        ph=float(x.High.iloc[i-cfg.breakout_lookback:i].max());pl=float(x.Low.iloc[i-cfg.breakout_lookback:i].min())
        if abs(float(x.Close.iloc[i])-float(x.Open.iloc[i]))/a<cfg.min_body_atr:continue
        if close>es and ef>es and down>=1 and close>ph+cfg.breakout_buffer_atr*a:
            x.iat[i,x.columns.get_loc("signal")]=1;x.iat[i,x.columns.get_loc("sl")]=close-cfg.atr_stop*a;x.iat[i,x.columns.get_loc("tp")]=close+cfg.rr*cfg.atr_stop*a;x.iat[i,x.columns.get_loc("reason")]="trend-separated | pullback | displacement breakout"
        elif close<es and ef<es and up>=1 and close<pl-cfg.breakout_buffer_atr*a:
            x.iat[i,x.columns.get_loc("signal")]=-1;x.iat[i,x.columns.get_loc("sl")]=close+cfg.atr_stop*a;x.iat[i,x.columns.get_loc("tp")]=close-cfg.rr*cfg.atr_stop*a;x.iat[i,x.columns.get_loc("reason")]="trend-separated | pullback | displacement breakout"
    last=-10**9
    for i in np.flatnonzero(x.signal.to_numpy()!=0):
        if i-last<cfg.cooldown_bars:x.iat[i,x.columns.get_loc("signal")]=0;x.iat[i,x.columns.get_loc("sl")]=np.nan;x.iat[i,x.columns.get_loc("tp")]=np.nan
        else:last=i
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
