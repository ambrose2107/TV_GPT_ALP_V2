"""GPT XAUUSD Confluence V4

5m execution strategy using 4H/1H/15m context.
Design goals: no look-ahead, explicit confluence scoring, liquidity sweeps,
FVG/IFVG, supply/demand, Fibonacci retracement/extension and ATR risk.

This is a research/backtest engine, not a guarantee of profitability.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

FIBS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.65, 0.705, 0.786, 0.886, 1.0, 1.272, 1.618)

@dataclass
class V4Config:
    atr_len: int = 14
    pivot: int = 3
    fib_low: float = 0.618
    fib_high: float = 0.786
    fib_key: float = 0.705
    min_score: int = 12
    strong_score: int = 16
    sweep_lookback: int = 20
    sweep_window: int = 6
    fvg_lookback: int = 12
    zone_atr: float = 0.35
    sl_atr: float = 0.20
    min_rr: float = 1.5
    max_hold_bars: int = 72
    cooldown_bars: int = 6
    risk_pct: float = 0.5


def atr(df, n=14):
    prev = df.Close.shift(1)
    tr = pd.concat([df.High-df.Low, (df.High-prev).abs(), (df.Low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def pivots(df, p=3):
    hi = df.High.eq(df.High.rolling(2*p+1, center=True).max())
    lo = df.Low.eq(df.Low.rolling(2*p+1, center=True).min())
    return hi.fillna(False), lo.fillna(False)


def structure(df, p=3):
    ph, pl = pivots(df, p)
    sh = df.High.where(ph).ffill()
    sl = df.Low.where(pl).ffill()
    # A close above/below the latest confirmed pivot is the structural break.
    trend = pd.Series(index=df.index, dtype=object)
    trend[df.Close > sh.shift(1)] = 'bullish'
    trend[df.Close < sl.shift(1)] = 'bearish'
    trend = trend.ffill().fillna('neutral')
    return ph, pl, sh, sl, trend


def fvg_state(df):
    bull = df.Low > df.High.shift(2)
    bear = df.High < df.Low.shift(2)
    # latest unfilled gap bounds; IFVG is represented when price crosses through
    # a previously active gap and then respects it from the opposite side.
    blo = pd.Series(np.nan, index=df.index); bhi = pd.Series(np.nan, index=df.index)
    slo = pd.Series(np.nan, index=df.index); shi = pd.Series(np.nan, index=df.index)
    b_active = None; s_active = None
    for i in range(len(df)):
        if bull.iloc[i]: b_active = (float(df.High.iloc[i-2]), float(df.Low.iloc[i]))
        if bear.iloc[i]: s_active = (float(df.High.iloc[i]), float(df.Low.iloc[i-2]))
        if b_active and float(df.Close.iloc[i]) < b_active[0]: b_active = None
        if s_active and float(df.Close.iloc[i]) > s_active[1]: s_active = None
        if b_active: blo.iloc[i], bhi.iloc[i] = b_active
        if s_active: slo.iloc[i], shi.iloc[i] = s_active
    return bull, bear, blo, bhi, slo, shi


def resample(df, rule):
    return df.resample(rule, label='right', closed='right').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()


def _load_alpaca_5m(symbol='GLD', n_bars=30000, feed=None):
    """Load 5-minute US ETF bars from Alpaca Market Data with pagination."""
    import os
    import requests

    api_key = os.environ.get('ALPACA_API_KEY', '')
    secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
    if not api_key or not secret_key:
        raise RuntimeError('ALPACA_API_KEY / ALPACA_SECRET_KEY are not configured on Render.')

    symbol = str(symbol).upper().strip()
    feed = (feed or os.environ.get('ALPACA_DATA_FEED', 'sip')).lower()
    if feed not in ('sip', 'iex'):
        raise ValueError("ALPACA_DATA_FEED must be 'sip' or 'iex'.")

    # Alpaca's Basic plan restricts recent SIP data; keeping the end at least
    # 20 minutes old also makes the backtest deterministic while the market is open.
    end_ts = pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=20)
    # ~78 regular-session 5m bars/day. Add ample calendar slack for weekends/holidays.
    calendar_days = max(30, int(np.ceil(max(1000, n_bars) / 78.0 * 7.0 / 5.0)) + 14)
    start_ts = end_ts - pd.Timedelta(days=calendar_days)

    url = f'https://data.alpaca.markets/v2/stocks/{symbol}/bars'
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Accept': 'application/json',
    }

    last_error = None
    for selected_feed in ([feed, 'iex'] if feed == 'sip' else ['iex']):
        rows = []
        page_token = None
        try:
            while True:
                params = {
                    'timeframe': '5Min',
                    'start': start_ts.isoformat().replace('+00:00', 'Z'),
                    'end': end_ts.isoformat().replace('+00:00', 'Z'),
                    'limit': 10000,
                    'adjustment': 'raw',
                    'feed': selected_feed,
                    'sort': 'asc',
                }
                if page_token:
                    params['page_token'] = page_token

                resp = requests.get(url, headers=headers, params=params, timeout=30)
                if resp.status_code >= 400:
                    detail = resp.text[:500].replace('\\n', ' ')
                    raise RuntimeError(f'Alpaca HTTP {resp.status_code} ({selected_feed}): {detail}')
                payload = resp.json()
                page_rows = payload.get('bars') or []
                rows.extend(page_rows)
                page_token = payload.get('next_page_token')
                if not page_token or len(rows) >= n_bars:
                    break

            if not rows:
                raise RuntimeError(f'Alpaca returned no {selected_feed} 5m bars for {symbol}.')

            frame = pd.DataFrame(rows)
            frame['timestamp'] = pd.to_datetime(frame['t'], utc=True)
            frame = frame.rename(columns={
                'o':'Open', 'h':'High', 'l':'Low', 'c':'Close', 'v':'Volume'
            })
            frame = frame.set_index('timestamp')[['Open','High','Low','Close','Volume']]
            frame = frame[~frame.index.duplicated(keep='last')].sort_index()
            frame = frame.apply(pd.to_numeric, errors='coerce').dropna()
            if frame.empty:
                raise RuntimeError(f'Alpaca returned unusable {selected_feed} bars for {symbol}.')
            return frame.tail(n_bars)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f'Alpaca 5m data failed for {symbol}: {last_error}')


def load_data(use_live=True, n_bars=30000, symbol='GLD', data_source='alpaca'):
    if not use_live:
        raise ValueError('V4 live backtest requires market data; use_live=false is intentionally disabled.')

    symbol = str(symbol or 'GLD').upper().strip()
    data_source = str(data_source or 'alpaca').lower().strip()

    if data_source == 'alpaca':
        raw = _load_alpaca_5m(symbol, n_bars)
    elif data_source == 'yahoo':
        # Kept as an optional research provider, but the V4 page defaults to Alpaca/GLD.
        import yfinance as yf
        raw = yf.download(symbol, period='60d', interval='5m', progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            raise RuntimeError(f'Yahoo returned no 5m market data for {symbol}.')
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[['Open','High','Low','Close','Volume']].dropna().tail(n_bars)
    else:
        raise ValueError("data_source must be 'alpaca' or 'yahoo'.")

    raw = raw[['Open','High','Low','Close','Volume']].dropna().tail(n_bars)
    raw.index = pd.to_datetime(raw.index)
    return {
        'm5': raw,
        'm15': resample(raw, '15min'),
        'h1': resample(raw, '1h'),
        'h4': resample(raw, '4h'),
    }

def _fib_zone(low, high, side, a, b):
    rng = high-low
    if side == 'long': return high-rng*b, high-rng*a
    return low+rng*a, low+rng*b


def _nearest(value, levels, tol):
    return any(abs(value-x) <= tol for x in levels if np.isfinite(x))


def build_signals(data, cfg=V4Config()):
    m5, m15, h1, h4 = (data[k].copy() for k in ('m5','m15','h1','h4'))
    for d in (m5,m15,h1,h4): d['atr']=atr(d,cfg.atr_len)
    _,_,sh15,sl15,tr15 = structure(m15,cfg.pivot)
    _,_,sh1,sl1,tr1 = structure(h1,cfg.pivot)
    _,_,sh4,sl4,tr4 = structure(h4,cfg.pivot)
    _,_,_,_,_,_ = fvg_state(m15)
    bull5,bear5,sh5,sl5,tr5 = structure(m5,cfg.pivot)
    _,_,fvg_lo_b,fvg_hi_b,fvg_lo_s,fvg_hi_s = fvg_state(m15)

    out=m5.copy(); out['signal']=0; out['score']=0; out['sl']=np.nan; out['tp1']=np.nan; out['tp2']=np.nan; out['tp3']=np.nan; out['reason']=''
    # Shift HTF features one completed bar before mapping to 5m: no future leakage.
    h1a=tr1.shift(1).reindex(m5.index,method='ffill'); h4a=tr4.shift(1).reindex(m5.index,method='ffill')
    m15a=m15.reindex(m5.index,method='ffill'); tr15a=tr15.shift(1).reindex(m5.index,method='ffill')
    sh15a=sh15.shift(1).reindex(m5.index,method='ffill'); sl15a=sl15.shift(1).reindex(m5.index,method='ffill')
    atr15a=m15.atr.shift(1).reindex(m5.index,method='ffill')
    # Confirmed 15m FVG state is also shifted before mapping.
    fbl=fvg_lo_b.shift(1).reindex(m5.index,method='ffill'); fbh=fvg_hi_b.shift(1).reindex(m5.index,method='ffill')
    fsl=fvg_lo_s.shift(1).reindex(m5.index,method='ffill'); fsh=fvg_hi_s.shift(1).reindex(m5.index,method='ffill')

    last_low=np.nan; last_high=np.nan; last_long_sweep=-999; last_short_sweep=-999; last_trade=-999
    for i in range(max(30,cfg.pivot*3),len(out)):
        price=float(out.Close.iloc[i]); a=float(out.atr.iloc[i])
        if not np.isfinite(a) or a<=0: continue
        # confirmed 15m impulse from most recent structural leg
        hi=float(sh15a.iloc[i]) if np.isfinite(sh15a.iloc[i]) else np.nan
        lo=float(sl15a.iloc[i]) if np.isfinite(sl15a.iloc[i]) else np.nan
        if not np.isfinite(hi) or not np.isfinite(lo) or hi<=lo: continue
        side15='long' if tr15a.iloc[i]=='bullish' else 'short' if tr15a.iloc[i]=='bearish' else None
        if side15 is None: continue
        # HTF alignment is descriptive context, not a hard AND gate.
        score=0; reasons=[]
        if side15=='long':
            if h1a.iloc[i]=='bullish': score+=2; reasons.append('1H bullish')
            if h4a.iloc[i]=='bullish': score+=1; reasons.append('4H bullish')
            zlo,zhi=_fib_zone(lo,hi,'long',cfg.fib_low,cfg.fib_high)
            in_fib=zlo <= price <= zhi
            if in_fib: score+=2; reasons.append('0.618-0.786')
            key=hi-(hi-lo)*cfg.fib_key
            if abs(price-key)<=cfg.zone_atr*a: score+=1; reasons.append('0.705')
            # Liquidity sweep: wick through a recent low then close back above it.
            recent=float(out.Low.iloc[max(0,i-cfg.sweep_lookback):i].min())
            if out.Low.iloc[i] < recent and price > recent: last_long_sweep=i; score+=3; reasons.append('sell-side sweep')
            elif i-last_long_sweep<=cfg.sweep_window: score+=2; reasons.append('recent sweep')
            # Demand / order-block proxy: last bearish 5m candle near the zone.
            if in_fib and out.Close.iloc[i-1] < out.Open.iloc[i-1] and out.Low.iloc[i-1] <= zhi: score+=2; reasons.append('demand/OB')
            # FVG / IFVG confluence.
            if np.isfinite(fbl.iloc[i]) and fbl.iloc[i] <= price <= fbh.iloc[i]: score+=2; reasons.append('bull FVG')
            if np.isfinite(fsl.iloc[i]) and price >= fsh.iloc[i] and i-last_long_sweep<=cfg.sweep_window: score+=2; reasons.append('IFVG reaction')
            # 5m MSS/BOS proxy: close above prior confirmed swing high after sweep.
            if price > sh5.iloc[i-1] if np.isfinite(sh5.iloc[i-1]) else False:
                score+=3; reasons.append('5m BOS/MSS')
            # displacement proxy.
            if out.Close.iloc[i]-out.Open.iloc[i] > 0.7*a: score+=2; reasons.append('bull displacement')
            sl=min(float(out.Low.iloc[i]), recent)-cfg.sl_atr*a
            risk=price-sl; tp1=hi; tp2=price+risk*2; tp3=price+risk*3
            valid=score>=cfg.min_score and risk>0 and (tp1-price)/risk>=cfg.min_rr
            if valid and i-last_trade>=cfg.cooldown_bars:
                out.iloc[i,out.columns.get_loc('signal')]=1; out.iloc[i,out.columns.get_loc('score')]=score; out.iloc[i,out.columns.get_loc('sl')]=sl; out.iloc[i,out.columns.get_loc('tp1')]=tp1; out.iloc[i,out.columns.get_loc('tp2')]=tp2; out.iloc[i,out.columns.get_loc('tp3')]=tp3; out.iloc[i,out.columns.get_loc('reason')]=' | '.join(reasons); last_trade=i
        else:
            if h1a.iloc[i]=='bearish': score+=2; reasons.append('1H bearish')
            if h4a.iloc[i]=='bearish': score+=1; reasons.append('4H bearish')
            zlo,zhi=_fib_zone(lo,hi,'short',cfg.fib_low,cfg.fib_high)
            in_fib=zlo <= price <= zhi
            if in_fib: score+=2; reasons.append('0.618-0.786')
            key=lo+(hi-lo)*cfg.fib_key
            if abs(price-key)<=cfg.zone_atr*a: score+=1; reasons.append('0.705')
            recent=float(out.High.iloc[max(0,i-cfg.sweep_lookback):i].max())
            if out.High.iloc[i] > recent and price < recent: last_short_sweep=i; score+=3; reasons.append('buy-side sweep')
            elif i-last_short_sweep<=cfg.sweep_window: score+=2; reasons.append('recent sweep')
            if in_fib and out.Close.iloc[i-1] > out.Open.iloc[i-1] and out.High.iloc[i-1] >= zlo: score+=2; reasons.append('supply/OB')
            if np.isfinite(fsl.iloc[i]) and fsl.iloc[i] <= price <= fsh.iloc[i]: score+=2; reasons.append('bear FVG')
            if np.isfinite(fbl.iloc[i]) and price <= fbl.iloc[i] and i-last_short_sweep<=cfg.sweep_window: score+=2; reasons.append('IFVG reaction')
            if price < sl5.iloc[i-1] if np.isfinite(sl5.iloc[i-1]) else False: score+=3; reasons.append('5m BOS/MSS')
            if out.Open.iloc[i]-out.Close.iloc[i] > 0.7*a: score+=2; reasons.append('bear displacement')
            sl=max(float(out.High.iloc[i]),recent)+cfg.sl_atr*a
            risk=sl-price; tp1=lo; tp2=price-risk*2; tp3=price-risk*3
            valid=score>=cfg.min_score and risk>0 and (price-tp1)/risk>=cfg.min_rr
            if valid and i-last_trade>=cfg.cooldown_bars:
                out.iloc[i,out.columns.get_loc('signal')]=-1; out.iloc[i,out.columns.get_loc('score')]=score; out.iloc[i,out.columns.get_loc('sl')]=sl; out.iloc[i,out.columns.get_loc('tp1')]=tp1; out.iloc[i,out.columns.get_loc('tp2')]=tp2; out.iloc[i,out.columns.get_loc('tp3')]=tp3; out.iloc[i,out.columns.get_loc('reason')]=' | '.join(reasons); last_trade=i
    return out


def backtest(data,cfg=V4Config(),initial_equity=10000):
    df=build_signals(data,cfg); equity=initial_equity; trades=[]; eq=[]; pos=None
    for i,(ts,r) in enumerate(df.iterrows()):
        if pos is None and r.signal!=0:
            pos={'side':int(r.signal),'entry':float(r.Close),'sl':float(r.sl),'tp1':float(r.tp1),'tp2':float(r.tp2),'tp3':float(r.tp3),'remaining':1.0,'realized':0.0,'age':0,'score':int(r.score),'reason':r.reason,'entry_time':ts}
        if pos:
            pos['age']+=1; side=pos['side']; px=float(r.Close); hi=float(r.High); lo=float(r.Low); risk=abs(pos['entry']-pos['sl']);
            hit=[]
            if side==1:
                if lo<=pos['sl']: hit.append(('SL',pos['sl'],pos['remaining']))
                else:
                    for name,tp in [('TP1',pos['tp1']),('TP2',pos['tp2']),('TP3',pos['tp3'])]:
                        if hi>=tp and pos['remaining']>0: hit.append((name,tp,0.25 if name!='TP3' else pos['remaining'])); pos['remaining']-=0.25 if name!='TP3' else pos['remaining'];
            else:
                if hi>=pos['sl']: hit.append(('SL',pos['sl'],pos['remaining']))
                else:
                    for name,tp in [('TP1',pos['tp1']),('TP2',pos['tp2']),('TP3',pos['tp3'])]:
                        if lo<=tp and pos['remaining']>0: hit.append((name,tp,0.25 if name!='TP3' else pos['remaining'])); pos['remaining']-=0.25 if name!='TP3' else pos['remaining']
            for name,fill,frac in hit:
                R=((fill-pos['entry'])/risk if side==1 else (pos['entry']-fill)/risk); pos['realized']+=R*frac
                if name=='TP1' and pos['remaining']>0: pos['sl']=pos['entry']
            if hit and hit[-1][0]=='SL' or pos['remaining']<=1e-9 or pos['age']>=cfg.max_hold_bars:
                if not hit and pos['age']>=cfg.max_hold_bars:
                    R=((px-pos['entry'])/risk if side==1 else (pos['entry']-px)/risk); pos['realized']+=R*pos['remaining']
                equity*=1+cfg.risk_pct/100*pos['realized']; trades.append({**pos,'exit_time':ts,'R':pos['realized'],'equity':equity}); pos=None
        eq.append(equity)
    if pos:
        px=float(df.Close.iloc[-1]); risk=abs(pos['entry']-pos['sl']); R=((px-pos['entry'])/risk if pos['side']==1 else (pos['entry']-px)/risk)*pos['remaining']+pos['realized']; equity*=1+cfg.risk_pct/100*R; trades.append({**pos,'exit_time':df.index[-1],'R':R,'equity':equity})
    t=pd.DataFrame(trades); e=pd.Series(eq,index=df.index)
    if t.empty: metrics={'num_trades':0,'win_rate':0.0,'profit_factor':0.0,'total_R':0.0,'expectancy_R':0.0,'max_drawdown_pct':0.0,'total_return_pct':0.0,'final_equity':equity}
    else:
        wins=t[t.R>0].R.sum(); losses=-t[t.R<0].R.sum(); peak=e.cummax(); dd=(e/peak-1).min()*100
        metrics={'num_trades':int(len(t)),'win_rate':round(float((t.R>0).mean()*100),2),'profit_factor':round(float(wins/losses),2) if losses else float('inf'),'total_R':round(float(t.R.sum()),2),'expectancy_R':round(float(t.R.mean()),3),'max_drawdown_pct':round(float(dd),2),'total_return_pct':round(float((equity/initial_equity-1)*100),2),'final_equity':round(float(equity),2)}
    return {'signals':df,'trades':t,'equity_curve':e,'metrics':metrics}
