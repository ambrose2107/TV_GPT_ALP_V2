"""Triple RSI V2: original setup plus reversal confirmation."""
from dataclasses import dataclass
import numpy as np,pandas as pd
from research.xauusd_triple_rsi_v1 import _rsi,_metrics
@dataclass
class TripleRSIV2Config:
    rsi_len:int=5;rsi_entry_max:float=30.;rsi_three_days_ago_max:float=60.;rsi_exit_level:float=50.;ma_len:int=200;risk_pct:float=.5;initial_equity:float=10000.;require_reversal:bool=True;max_hold_days:int=0
def backtest(df,cfg=TripleRSIV2Config()):
    x=df.copy();x.index=pd.to_datetime(x.index).sort_values()
    if getattr(x.index,"tz",None) is not None:x.index=x.index.tz_convert("UTC").tz_localize(None)
    d=x[["Open","High","Low","Close"]].resample("1D").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
    d["rsi"]=_rsi(d.Close,cfg.rsi_len)
    d["ma200"]=d.Close.rolling(cfg.ma_len,min_periods=cfg.ma_len).mean()
    d["fall3"]=(d.rsi<d.rsi.shift(1))&(d.rsi.shift(1)<d.rsi.shift(2))
    d["rsi3"]=d.rsi.shift(3)<cfg.rsi_three_days_ago_max
    d["cross"]=(d.rsi>cfg.rsi_exit_level)&(d.rsi.shift(1)<=cfg.rsi_exit_level)
    d["reversal"]=d.Close>=d.Close.shift(1)

    trades=[];signals=[];eq=float(cfg.initial_equity);curve=[];pos=None;hold=0
    for i in range(len(d)):
        row=d.iloc[i]
        # Execute a prior day's confirmed signal at today's OPEN.
        if pos is not None:
            hold+=1
            if bool(row.cross) or (cfg.max_hold_days>0 and hold>=cfg.max_hold_days):
                ep=float(row.Close)
                ret=ep/pos["entry"]-1.0
                eq*=1.0+ret
                r_value=ret/(cfg.risk_pct/100.0) if cfg.risk_pct>0 else 0.0
                trades.append({
                    "entry_time":pos["entry_time"].isoformat(),
                    "exit_time":row.name.isoformat(),
                    "side":"LONG","entry":pos["entry"],"exit_price":ep,
                    "sl":np.nan,"tp":np.nan,"R":r_value,
                    "return_pct":ret*100.0,"equity":eq,
                    "reason":"rsi5_cross_above_50" if bool(row.cross) else "max_hold"
                })
                pos=None;hold=0

        # A signal is confirmed on today's close; execution is next day's open.
        valid=(
            np.isfinite(row.rsi) and np.isfinite(row.ma200)
            and bool(row.fall3) and bool(row.rsi3)
            and float(row.rsi)<cfg.rsi_entry_max
            and float(row.Close)>float(row.ma200)
            and (not cfg.require_reversal or bool(row.reversal))
        )
        signals.append({
            "signal_time":row.name.isoformat(),
            "rsi5":float(row.rsi) if np.isfinite(row.rsi) else np.nan,
            "ma200":float(row.ma200) if np.isfinite(row.ma200) else np.nan,
            "reversal":bool(row.reversal),"signal":bool(valid)
        })

        if pos is None and valid and i+1<len(d):
            nxt=d.iloc[i+1]
            pos={"entry_time":nxt.name,"entry":float(nxt.Open)}
            hold=0

        curve.append(eq)

    # Do not fabricate a closing price for an open position. It remains
    # unclosed and is excluded from realized-trade metrics.
    return {
        "data":d,
        "signals":pd.DataFrame(signals),
        "trades":trades,
        "equity_curve":curve,
        "metrics":_metrics(trades,curve,cfg.initial_equity,cfg.risk_pct)
    }
