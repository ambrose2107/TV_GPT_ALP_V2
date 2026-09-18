"""GPT XAUUSD Confluence V4

5m execution strategy using 4H/1H/15m context.
Design goals: no look-ahead, explicit confluence scoring, liquidity sweeps,
FVG/IFVG, supply/demand, Fibonacci retracement/extension and ATR risk.

This is a research/backtest engine, not a guarantee of profitability.
"""
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("xauusd_confluence_v4")

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
    # A centered pivot needs p bars on the right to be confirmed. Delay the
    # level by p bars before using it so the backtest cannot see future price.
    sh = df.High.where(ph).ffill().shift(p)
    sl = df.Low.where(pl).ffill().shift(p)
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
        # A 3-candle FVG is undefined for the first two rows. Avoid negative
        # iloc indexing, which otherwise reads candles from the dataframe tail.
        if i >= 2:
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
    """Load 5-minute US ETF bars from Alpaca with pagination and bounded memory."""
    import os
    import requests

    n_bars = max(100, min(int(n_bars), 60000))
    api_key = os.environ.get('ALPACA_API_KEY', '')
    secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
    if not api_key or not secret_key:
        raise RuntimeError('ALPACA_API_KEY / ALPACA_SECRET_KEY are not configured on Render.')

    symbol = str(symbol).upper().strip()
    feed = (feed or os.environ.get('ALPACA_DATA_FEED', 'sip')).lower()
    if feed not in ('sip', 'iex'):
        raise ValueError("ALPACA_DATA_FEED must be 'sip' or 'iex'.")

    end_ts = pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=20)
    # Request only the period needed for the requested sample plus warm-up.
    # This avoids a large 30-day query when testing only 50-200 bars.
    calendar_days = max(5, int(np.ceil(n_bars / 78.0 * 7.0 / 5.0)) + 3)
    start_ts = end_ts - pd.Timedelta(days=calendar_days)

    url = f'https://data.alpaca.markets/v2/stocks/{symbol}/bars'
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Accept': 'application/json',
    }
    logger.info('[DATA] Alpaca request symbol=%s timeframe=5Min requested=%d window=%s..%s feed=%s',
                symbol, n_bars, start_ts.isoformat(), end_ts.isoformat(), feed)

    last_error = None
    for selected_feed in ([feed, 'iex'] if feed == 'sip' else ['iex']):
        rows = []
        page_token = None
        pages = 0
        try:
            while True:
                params = {
                    'timeframe': '5Min',
                    'start': start_ts.isoformat().replace('+00:00', 'Z'),
                    'end': end_ts.isoformat().replace('+00:00', 'Z'),
                    'limit': min(5000, n_bars),
                    'adjustment': 'raw',
                    'feed': selected_feed,
                    'sort': 'asc',
                }
                if page_token:
                    params['page_token'] = page_token

                # Fail fast so a slow feed can fall back to IEX before the
                # Render/proxy request timeout is reached.
                resp = requests.get(url, headers=headers, params=params, timeout=12)
                status = resp.status_code
                if status >= 400:
                    detail = resp.text[:500].replace('\\n', ' ')
                    raise RuntimeError(f'Alpaca HTTP {status} ({selected_feed}): {detail}')
                payload = resp.json()
                page_rows = payload.get('bars') or []
                pages += 1

                # Store compact tuples instead of retaining thousands of JSON dicts.
                for b in page_rows:
                    try:
                        rows.append((b['t'], b['o'], b['h'], b['l'], b['c'], b.get('v', 0)))
                    except (KeyError, TypeError):
                        continue

                page_token = payload.get('next_page_token')
                logger.info('[DATA] Alpaca page=%d rows=%d total=%d feed=%s',
                            pages, len(page_rows), len(rows), selected_feed)

                del page_rows, payload, resp
                if not page_token or len(rows) >= n_bars:
                    break

            if not rows:
                raise RuntimeError(f'Alpaca returned no {selected_feed} 5m bars for {symbol}.')

            frame = pd.DataFrame(rows, columns=['timestamp','Open','High','Low','Close','Volume'])
            del rows
            frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True)
            frame = frame.set_index('timestamp')
            frame = frame[~frame.index.duplicated(keep='last')].sort_index()
            for col in ('Open','High','Low','Close'):
                frame[col] = pd.to_numeric(frame[col], errors='coerce').astype('float64')
            frame['Volume'] = pd.to_numeric(frame['Volume'], errors='coerce').fillna(0).astype('int64')
            frame = frame.dropna(subset=['Open','High','Low','Close'])
            frame = frame.tail(n_bars)

            logger.info('[DATA] Alpaca success feed=%s pages=%d bars=%d start=%s end=%s',
                        selected_feed, pages, len(frame),
                        frame.index.min(), frame.index.max())
            return frame
        except Exception as exc:
            last_error = exc
            logger.exception('[DATA] Alpaca failed feed=%s symbol=%s', selected_feed, symbol)

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
    m15 = resample(raw, '15min')
    h1 = resample(raw, '1h')
    h4 = resample(raw, '4h')
    logger.info('[DATA] Final dataset symbol=%s source=%s m5=%d m15=%d h1=%d h4=%d',
                symbol, data_source, len(raw), len(m15), len(h1), len(h4))
    return {'m5': raw, 'm15': m15, 'h1': h1, 'h4': h4}

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
    bull5,bear5,sh5,sl5,tr5 = structure(m5,cfg.pivot)
    _,_,fvg_lo_b,fvg_hi_b,fvg_lo_s,fvg_hi_s = fvg_state(m15)

    out=m5.copy(); out['signal']=0; out['score']=0; out['sl']=np.nan; out['tp1']=np.nan; out['tp2']=np.nan; out['tp3']=np.nan; out['reason']=''
    # Shift HTF features one completed bar before mapping to 5m: no future leakage.
    h1a=tr1.shift(1).reindex(m5.index,method='ffill'); h4a=tr4.shift(1).reindex(m5.index,method='ffill')
    tr15a=tr15.shift(1).reindex(m5.index,method='ffill')
    sh15a=sh15.shift(1).reindex(m5.index,method='ffill'); sl15a=sl15.shift(1).reindex(m5.index,method='ffill')
    atr15a=m15.atr.shift(1).reindex(m5.index,method='ffill')
    # Confirmed 15m FVG state is also shifted before mapping.
    fbl=fvg_lo_b.shift(1).reindex(m5.index,method='ffill'); fbh=fvg_hi_b.shift(1).reindex(m5.index,method='ffill')
    fsl=fvg_lo_s.shift(1).reindex(m5.index,method='ffill'); fsh=fvg_hi_s.shift(1).reindex(m5.index,method='ffill')

    last_long_sweep=-999; last_short_sweep=-999; last_trade=-999
    diag={'bars':0,'side15':0,'score_ge_min':0,'risk_valid':0,'rr_valid':0,'signals':0,'max_score':0}
    for i in range(max(30,cfg.pivot*3),len(out)):
        diag['bars'] += 1
        price=float(out.Close.iloc[i]); a=float(out.atr.iloc[i])
        if not np.isfinite(a) or a<=0: continue
        # confirmed 15m impulse from most recent structural leg
        hi=float(sh15a.iloc[i]) if np.isfinite(sh15a.iloc[i]) else np.nan
        lo=float(sl15a.iloc[i]) if np.isfinite(sl15a.iloc[i]) else np.nan
        if not np.isfinite(hi) or not np.isfinite(lo) or hi<=lo: continue
        side15='long' if tr15a.iloc[i]=='bullish' else 'short' if tr15a.iloc[i]=='bearish' else None
        if side15 is None: continue
        diag['side15'] += 1
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
            prev_sh5 = sh5.iloc[i-1]
            if np.isfinite(prev_sh5) and price > float(prev_sh5):
                score+=3; reasons.append('5m BOS/MSS')
            # displacement proxy.
            if out.Close.iloc[i]-out.Open.iloc[i] > 0.7*a: score+=2; reasons.append('bull displacement')
            sl=min(float(out.Low.iloc[i]), recent)-cfg.sl_atr*a
            risk=price-sl; tp1=hi; tp2=price+risk*2; tp3=price+risk*3
            rr=(tp1-price)/risk if risk>0 else -np.inf
            diag['max_score']=max(diag['max_score'], int(score))
            if score>=cfg.min_score: diag['score_ge_min'] += 1
            if risk>0: diag['risk_valid'] += 1
            if rr>=cfg.min_rr: diag['rr_valid'] += 1
            valid=score>=cfg.min_score and risk>0 and rr>=cfg.min_rr
            if valid and i-last_trade>=cfg.cooldown_bars:
                out.iloc[i,out.columns.get_loc('signal')]=1; out.iloc[i,out.columns.get_loc('score')]=score; diag['signals'] += 1; out.iloc[i,out.columns.get_loc('sl')]=sl; out.iloc[i,out.columns.get_loc('tp1')]=tp1; out.iloc[i,out.columns.get_loc('tp2')]=tp2; out.iloc[i,out.columns.get_loc('tp3')]=tp3; out.iloc[i,out.columns.get_loc('reason')]=' | '.join(reasons); last_trade=i
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
            prev_sl5 = sl5.iloc[i-1]
            if np.isfinite(prev_sl5) and price < float(prev_sl5):
                score+=3; reasons.append('5m BOS/MSS')
            if out.Open.iloc[i]-out.Close.iloc[i] > 0.7*a: score+=2; reasons.append('bear displacement')
            sl=max(float(out.High.iloc[i]),recent)+cfg.sl_atr*a
            risk=sl-price; tp1=lo; tp2=price-risk*2; tp3=price-risk*3
            rr=(price-tp1)/risk if risk>0 else -np.inf
            diag['max_score']=max(diag['max_score'], int(score))
            if score>=cfg.min_score: diag['score_ge_min'] += 1
            if risk>0: diag['risk_valid'] += 1
            if rr>=cfg.min_rr: diag['rr_valid'] += 1
            valid=score>=cfg.min_score and risk>0 and rr>=cfg.min_rr
            if valid and i-last_trade>=cfg.cooldown_bars:
                out.iloc[i,out.columns.get_loc('signal')]=-1; out.iloc[i,out.columns.get_loc('score')]=score; diag['signals'] += 1; out.iloc[i,out.columns.get_loc('sl')]=sl; out.iloc[i,out.columns.get_loc('tp1')]=tp1; out.iloc[i,out.columns.get_loc('tp2')]=tp2; out.iloc[i,out.columns.get_loc('tp3')]=tp3; out.iloc[i,out.columns.get_loc('reason')]=' | '.join(reasons); last_trade=i
    out.attrs['signal_diag'] = diag
    return out


def backtest(data, cfg=V4Config(), initial_equity=10000, signals=None):
    """Run the V4 backtest using immutable initial risk for all R calculations.

    The stop may move to breakeven after TP1. R-multiples must still be measured
    against the original entry-to-stop distance; recomputing risk from the moved
    stop would make risk zero and caused ZeroDivisionError.
    """
    df = signals.copy() if signals is not None else build_signals(data, cfg)
    equity = float(initial_equity)
    trades = []
    eq = []
    pos = None
    eps = 1e-12

    for ts, r in df.iterrows():
        opened_this_bar = False

        if pos is None and r.signal != 0:
            entry = float(r.Close)
            stop = float(r.sl)
            initial_risk = abs(entry - stop)

            # Defensive guard: malformed/degenerate signals must never enter.
            if (
                np.isfinite(entry)
                and np.isfinite(stop)
                and np.isfinite(initial_risk)
                and initial_risk > eps
            ):
                pos = {
                    'side': int(r.signal),
                    'entry': entry,
                    'sl': stop,
                    'initial_sl': stop,
                    'initial_risk': initial_risk,
                    'tp1': float(r.tp1),
                    'tp2': float(r.tp2),
                    'tp3': float(r.tp3),
                    'remaining': 1.0,
                    'realized': 0.0,
                    'age': 0,
                    'score': int(r.score),
                    'reason': r.reason,
                    'entry_time': ts,
                }
                opened_this_bar = True
            else:
                logger.warning(
                    '[BACKTEST] skipped zero/invalid-risk signal time=%s entry=%s sl=%s risk=%s',
                    ts, entry, stop, initial_risk
                )

        # Do not use the entry candle's already-known high/low to decide an exit.
        if pos is not None and not opened_this_bar:
            pos['age'] += 1
            side = pos['side']
            px = float(r.Close)
            hi = float(r.High)
            lo = float(r.Low)
            risk = float(pos['initial_risk'])

            if not np.isfinite(risk) or risk <= eps:
                logger.error(
                    '[BACKTEST] invalid stored initial risk time=%s entry=%s initial_sl=%s risk=%s',
                    ts, pos['entry'], pos.get('initial_sl'), risk
                )
                pos = None
                eq.append(equity)
                continue

            hit = []

            if side == 1:
                if lo <= pos['sl']:
                    hit.append(('SL', float(pos['sl']), float(pos['remaining'])))
                else:
                    for name, tp in [('TP1', pos['tp1']), ('TP2', pos['tp2']), ('TP3', pos['tp3'])]:
                        if hi >= tp and pos['remaining'] > eps:
                            frac = 0.25 if name != 'TP3' else pos['remaining']
                            frac = min(float(frac), float(pos['remaining']))
                            hit.append((name, float(tp), frac))
                            pos['remaining'] = max(0.0, pos['remaining'] - frac)
            else:
                if hi >= pos['sl']:
                    hit.append(('SL', float(pos['sl']), float(pos['remaining'])))
                else:
                    for name, tp in [('TP1', pos['tp1']), ('TP2', pos['tp2']), ('TP3', pos['tp3'])]:
                        if lo <= tp and pos['remaining'] > eps:
                            frac = 0.25 if name != 'TP3' else pos['remaining']
                            frac = min(float(frac), float(pos['remaining']))
                            hit.append((name, float(tp), frac))
                            pos['remaining'] = max(0.0, pos['remaining'] - frac)

            for name, fill, frac in hit:
                if side == 1:
                    R = (fill - pos['entry']) / risk
                else:
                    R = (pos['entry'] - fill) / risk
                pos['realized'] += R * frac

                # Move the live stop to BE, but NEVER change initial_risk.
                if name == 'TP1' and pos['remaining'] > eps:
                    pos['sl'] = pos['entry']

            stopped = bool(hit) and hit[-1][0] == 'SL'
            fully_exited = pos['remaining'] <= eps
            timed_out = pos['age'] >= cfg.max_hold_bars

            if stopped or fully_exited or timed_out:
                if not hit and timed_out and pos['remaining'] > eps:
                    if side == 1:
                        R = (px - pos['entry']) / risk
                    else:
                        R = (pos['entry'] - px) / risk
                    pos['realized'] += R * pos['remaining']
                    pos['remaining'] = 0.0

                equity *= 1 + (cfg.risk_pct / 100.0) * pos['realized']
                trades.append({
                    **pos,
                    'exit_time': ts,
                    'R': pos['realized'],
                    'equity': equity,
                })
                pos = None

        eq.append(equity)

    # Mark any still-open trade to the final close using original risk.
    if pos is not None:
        px = float(df.Close.iloc[-1])
        risk = float(pos['initial_risk'])
        if np.isfinite(risk) and risk > eps:
            if pos['side'] == 1:
                open_R = (px - pos['entry']) / risk
            else:
                open_R = (pos['entry'] - px) / risk
            total_R = pos['realized'] + open_R * pos['remaining']
        else:
            logger.error('[BACKTEST] invalid final initial risk; using realized R only')
            total_R = pos['realized']

        equity *= 1 + (cfg.risk_pct / 100.0) * total_R
        trades.append({
            **pos,
            'exit_time': df.index[-1],
            'R': total_R,
            'equity': equity,
        })

    t = pd.DataFrame(trades)
    e = pd.Series(eq, index=df.index, dtype='float64')

    if t.empty:
        metrics = {
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_R': 0.0,
            'expectancy_R': 0.0,
            'max_drawdown_pct': 0.0,
            'total_return_pct': 0.0,
            'final_equity': equity,
        }
    else:
        wins = t.loc[t.R > 0, 'R'].sum()
        losses = -t.loc[t.R < 0, 'R'].sum()
        peak = e.cummax()
        dd = (e / peak - 1).min() * 100
        metrics = {
            'num_trades': int(len(t)),
            'win_rate': round(float((t.R > 0).mean() * 100), 2),
            'profit_factor': round(float(wins / losses), 2) if losses else float('inf'),
            'total_R': round(float(t.R.sum()), 2),
            'expectancy_R': round(float(t.R.mean()), 3),
            'max_drawdown_pct': round(float(dd), 2),
            'total_return_pct': round(float((equity / initial_equity - 1) * 100), 2),
            'final_equity': round(float(equity), 2),
        }

    return {'signals': df, 'trades': t, 'equity_curve': e, 'metrics': metrics}


def _trade_metrics(trades):
    """Compact metrics for an optimizer segment."""
    if trades is None or trades.empty:
        return {'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'total_R': 0.0, 'expectancy_R': 0.0}
    r = pd.to_numeric(trades['R'], errors='coerce').dropna()
    if r.empty:
        return {'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'total_R': 0.0, 'expectancy_R': 0.0}
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (float('inf') if wins > 0 else 0.0)
    return {'trades': int(len(r)),
            'win_rate': round(float((r > 0).mean() * 100), 2),
            'profit_factor': pf,
            'total_R': float(r.sum()),
            'expectancy_R': float(r.mean())}


def _apply_signal_filters(df, min_score, min_rr, cooldown):
    """Cheaply filter a precomputed signal frame for optimizer execution."""
    out = df.copy()
    risk = (out['Close'] - out['sl']).abs()
    long_rr = (out['tp1'] - out['Close']) / risk
    short_rr = (out['Close'] - out['tp1']) / risk
    rr = np.where(out['signal'].to_numpy() > 0, long_rr.to_numpy(), short_rr.to_numpy())
    valid = (
        (out['signal'].to_numpy() != 0) &
        (pd.to_numeric(out['score'], errors='coerce').to_numpy() >= float(min_score)) &
        np.isfinite(risk.to_numpy()) & (risk.to_numpy() > 1e-12) &
        np.isfinite(rr) & (rr >= float(min_rr))
    )
    # Reproduce the signal cooldown without rebuilding all market structure.
    chosen = np.flatnonzero(valid)
    if cooldown > 1 and len(chosen):
        keep = np.zeros(len(out), dtype=bool)
        last = -10**9
        for i in chosen:
            if i - last >= int(cooldown):
                keep[i] = True
                last = i
        valid = keep
    out.loc[~valid, ['signal','score','sl','tp1','tp2','tp3']] = [0, 0, np.nan, np.nan, np.nan, np.nan]
    out.loc[~valid, 'reason'] = ''
    return out


def optimize(data, base_cfg=None, initial_equity=10000, min_trades=5):
    """Fast optimizer with cached signals and chronological 70/30 validation.

    Expensive market-structure/FVG calculations are built only once per SL-ATR
    value. Score, RR and cooldown combinations are then filtered from those
    cached signals, making the optimizer much faster than rerunning build_signals
    for every combination.

    The ranking emphasizes validation PF while requiring a meaningful train/test
    sample and also considering expectancy and drawdown. It does not guarantee
    future profitability.
    """
    base = base_cfg or V4Config()
    candidates = []
    scores = (8, 10, 12, 14)
    rrs = (1.0, 1.5, 2.0)
    sls = (0.10, 0.30)
    cooldowns = (3, 6)
    split_i = max(1, int(len(data['m5']) * 0.70))
    split_time = data['m5'].index[split_i]
    min_trades = max(2, int(min_trades))

    for sl_atr in sls:
        # Cache the expensive signal-generation pass for this SL setting.
        signal_cfg = V4Config(**{**base.__dict__, 'min_score': 0, 'min_rr': 0.0,
                                 'sl_atr': sl_atr, 'cooldown_bars': 0})
        cached = build_signals(data, signal_cfg)
        for min_score in scores:
            for min_rr in rrs:
                for cooldown in cooldowns:
                    cfg = V4Config(**{**base.__dict__, 'min_score': min_score,
                                      'min_rr': min_rr, 'sl_atr': sl_atr,
                                      'cooldown_bars': cooldown})
                    filtered = _apply_signal_filters(cached, min_score, min_rr, cooldown)
                    result = backtest(data, cfg, initial_equity=initial_equity, signals=filtered)
                    m = result['metrics']
                    trades = result['trades'].copy()
                    if not trades.empty:
                        train = trades[trades['entry_time'] < split_time]
                        test = trades[trades['entry_time'] >= split_time]
                    else:
                        train = trades
                        test = trades
                    tm = _trade_metrics(train)
                    vm = _trade_metrics(test)
                    pf_train = float(tm['profit_factor'])
                    pf_test = float(vm['profit_factor'])
                    rank_train_pf = min(pf_train, 5.0) if np.isfinite(pf_train) else 5.0
                    rank_test_pf = min(pf_test, 5.0) if np.isfinite(pf_test) else 5.0
                    dd = abs(float(m.get('max_drawdown_pct', 0.0)))
                    robust_score = (
                        0.50 * rank_test_pf +
                        0.25 * rank_train_pf +
                        0.20 * max(-2.0, min(2.0, vm['expectancy_R'])) -
                        0.05 * dd
                    )
                    eligible = tm['trades'] >= min_trades and vm['trades'] >= 2
                    candidates.append({
                        'min_score': int(min_score), 'min_rr': float(min_rr),
                        'sl_atr': float(sl_atr), 'cooldown_bars': int(cooldown),
                        'profit_factor': float(m['profit_factor']) if np.isfinite(m['profit_factor']) else None,
                        'train_pf': pf_train if np.isfinite(pf_train) else None,
                        'test_pf': pf_test if np.isfinite(pf_test) else None,
                        'train_trades': int(tm['trades']), 'test_trades': int(vm['trades']),
                        'num_trades': int(m['num_trades']),
                        'total_R': float(m['total_R']),
                        'expectancy_R': float(m['expectancy_R']),
                        'win_rate': float(m['win_rate']),
                        'max_drawdown_pct': float(m['max_drawdown_pct']),
                        'total_return_pct': float(m['total_return_pct']),
                        'robust_score': round(float(robust_score), 3),
                        'eligible': bool(eligible)
                    })

    eligible_rows = [x for x in candidates if x['eligible']]
    pool = eligible_rows if eligible_rows else candidates
    pool.sort(key=lambda x: (
        x['robust_score'],
        x['test_pf'] if x['test_pf'] is not None else -1,
        x['total_R']
    ), reverse=True)
    return {
        'tested': len(candidates),
        'eligible': len(eligible_rows),
        'min_trades': min_trades,
        'train_pct': 70,
        'test_pct': 30,
        'split_time': str(split_time),
        'method': 'Cached signals + 70/30 chronological validation; validation PF weighted with train PF, expectancy and drawdown',
        'results': pool[:20]
    }
