"""EMA 20/50 Third-Retest V2: distinct retests plus rejection quality."""
from dataclasses import dataclass
import numpy as np,pandas as pd
from research.xauusd_ema_retest_v1 import _resample_ohlc,_atr,_in_overlap
@dataclass
class EMARetestV2Config:
    ema_fast:int=20;ema_slow:int=50;rr:float=2.5;risk_pct:float=1.;retest_number:int=3;atr_len:int=14;sl_buffer_atr:float=.10
    overlap_start_utc_hour:int=13;overlap_start_utc_minute:int=30;overlap_end_utc_hour:int=16;overlap_end_utc_minute:int=0
    one_trade_per_day:bool=True;min_gap_atr:float=.10;min_rejection_body_atr:float=.10;separation_bars:int=1
def build_signals(m5,cfg=EMARetestV2Config()):
    m15=_resample_ohlc(m5,"15min");h1=_resample_ohlc(m5,"1h");m15["ema20"]=m15.Close.ewm(span=20,adjust=False).mean();m15["ema50"]=m15.Close.ewm(span=50,adjust=False).mean();m15["atr"]=_atr(m15,cfg.atr_len)
    h1["ema20"]=h1.Close.ewm(span=20,adjust=False).mean();h1["ema50"]=h1.Close.ewm(span=50,adjust=False).mean();hb=pd.Series(np.where(h1.ema20>h1.ema50,1,np.where(h1.ema20<h1.ema50,-1,0)),index=h1.index).shift(1).reindex(m15.index,method="ffill").fillna(0).astype(int)
    bull=(m15.ema20>m15.ema50)&(m15.ema20.shift(1)<=m15.ema50.shift(1));bear=(m15.ema20<m15.ema50)&(m15.ema20.shift(1)>=m15.ema50.shift(1));m15["signal"]=0;m15["retest_count"]=0;m15["sl_ref"]=np.nan;m15["reason"]=""
    direction=count=away=0
    for i in range(max(50,cfg.atr_len)+2,len(m15)):
        if bull.iloc[i]:direction=1;count=away=0;continue
        if bear.iloc[i]:direction=-1;count=away=0;continue
        if (direction==1 and m15.ema20.iloc[i]<=m15.ema50.iloc[i]) or (direction==-1 and m15.ema20.iloc[i]>=m15.ema50.iloc[i]):direction=count=away=0;continue
        if not direction:continue
        ef=float(m15.ema20.iloc[i]);es=float(m15.ema50.iloc[i]);a=float(m15.atr.iloc[i])
        if not np.isfinite(a) or a<=0:continue
        if direction==1 and float(m15.Low.iloc[i])>ef+.15*a:away+=1
        if direction==-1 and float(m15.High.iloc[i])<ef-.15*a:away+=1
        clean=(float(m15.Low.iloc[i])<=ef and float(m15.Low.iloc[i])>es and float(m15.Close.iloc[i])>ef) if direction==1 else (float(m15.High.iloc[i])>=ef and float(m15.High.iloc[i])<es and float(m15.Close.iloc[i])<ef)
        if clean and away>=cfg.separation_bars and abs(ef-es)/a>=cfg.min_gap_atr and abs(float(m15.Close.iloc[i])-float(m15.Open.iloc[i]))/a>=cfg.min_rejection_body_atr:
            count+=1;away=0;m15.iat[i,m15.columns.get_loc("retest_count")]=count
            if count==cfg.retest_number and int(hb.iloc[i])==direction and _in_overlap(m15.index[i],cfg):
                sl=es-cfg.sl_buffer_atr*a if direction==1 else es+cfg.sl_buffer_atr*a;m15.iat[i,m15.columns.get_loc("signal")]=direction;m15.iat[i,m15.columns.get_loc("sl_ref")]=sl;m15.iat[i,m15.columns.get_loc("reason")]=f"distinct retest #{count} | 1H aligned | rejection"
    return m15
def backtest(m5,cfg=EMARetestV2Config(),initial_equity=10000.):
    s=build_signals(m5,cfg);eq=float(initial_equity);pos=None;trades=[];curve=[];days=set()
    for i in range(len(s)):
        row=s.iloc[i];ts=s.index[i]
        if pos:
            hi,lo=float(row.High),float(row.Low);sl=lo<=pos["sl"] if pos["side"]==1 else hi>=pos["tp"];tp=hi>=pos["tp"] if pos["side"]==1 else lo<=pos["tp"]
            if sl or tp:
                r=-1. if sl else cfg.rr;eq+=pos["risk_cash"]*r;trades.append({"entry_time":str(pos["entry_time"]),"exit_time":str(ts),"side":"LONG" if pos["side"]==1 else "SHORT","entry":pos["entry"],"exit_price":pos["sl"] if sl else pos["tp"],"sl":pos["sl"],"tp":pos["tp"],"R":r,"equity":eq,"reason":pos["reason"]+" | "+("SL" if sl else "TP")});pos=None
        if pos is None and i+1<len(s) and int(row.signal)!=0:
            day=ts.date().isoformat()
            if cfg.one_trade_per_day and day in days:curve.append(eq);continue
            nxt=s.iloc[i+1];entry=float(nxt.Open);side=int(row.signal);stop=float(row.sl_ref)
            if (side==1 and stop<entry) or (side==-1 and stop>entry):
                risk=abs(entry-stop);pos={"side":side,"entry_time":s.index[i+1],"entry":entry,"sl":stop,"tp":entry+side*cfg.rr*risk,"risk_cash":eq*cfg.risk_pct/100,"reason":str(row.reason)};days.add(day)
        curve.append(eq)
    t=pd.DataFrame(trades);e=pd.Series(curve,index=s.index[:len(curve)],dtype=float)
    if t.empty:m={"num_trades":0,"win_rate":0.,"profit_factor":0.,"expectancy_R":0.,"total_R":0.,"max_drawdown_pct":0.,"total_return_pct":0.,"final_equity":eq}
    else:
        gw=float(t.loc[t.R>0,"R"].sum());gl=float(-t.loc[t.R<0,"R"].sum());m={"num_trades":len(t),"win_rate":round(float((t.R>0).mean()*100),2),"profit_factor":round(gw/gl,2) if gl else float("inf"),"expectancy_R":round(float(t.R.mean()),3),"total_R":round(float(t.R.sum()),2),"max_drawdown_pct":round(float((e/e.cummax()-1).min()*100),2),"total_return_pct":round((eq/initial_equity-1)*100,2),"final_equity":round(eq,2)}
    return {"signals":s,"trades":t,"equity_curve":e,"metrics":m}
