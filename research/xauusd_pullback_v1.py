"""Gold 5m Pullback Breakout V1.

A separate research strategy inspired by a documented four-phase pullback-window
framework: trend confirmation -> 1-3 candle pullback -> breakout confirmation ->
ATR-based risk management. It is deliberately independent of V4 SMC confluence.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from research.xauusd_confluence_v4 import load_data


@dataclass
class PullbackConfig:
    ema_fast: int = 50
    ema_slow: int = 200
    atr_len: int = 14
    pullback_bars: int = 3
    breakout_lookback: int = 5
    atr_stop: float = 1.2
    rr: float = 1.8
    risk_pct: float = 0.5
    cooldown_bars: int = 6
    session_start_utc: int = 13
    session_end_utc: int = 21


def atr(df, n):
    tr = pd.concat([
        df.High - df.Low,
        (df.High - df.Close.shift()).abs(),
        (df.Low - df.Close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def build_signals(df, cfg=PullbackConfig()):
    x = df.copy()
    x['ema_fast'] = x.Close.ewm(span=cfg.ema_fast, adjust=False).mean()
    x['ema_slow'] = x.Close.ewm(span=cfg.ema_slow, adjust=False).mean()
    x['atr'] = atr(x, cfg.atr_len)
    x['signal'] = 0
    x['sl'] = np.nan
    x['tp'] = np.nan
    x['reason'] = ''

    # DatetimeIndex.hour returns a NumPy array; keep the session mask as a
    # plain boolean array and index it with [i] (not .iloc).
    hours = x.index.hour.to_numpy()
    session = (hours >= cfg.session_start_utc) & (hours < cfg.session_end_utc)

    for i in range(max(cfg.ema_slow, cfg.breakout_lookback + cfg.pullback_bars + 2), len(x)):
        if not session[i]:
            continue
        a = float(x.atr.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue

        close = float(x.Close.iloc[i])
        ef = float(x.ema_fast.iloc[i])
        es = float(x.ema_slow.iloc[i])

        # Trend state must be known on the signal candle.
        bull = close > es and ef > es
        bear = close < es and ef < es

        # 1-3 completed counter-trend candles immediately before the breakout.
        n = cfg.pullback_bars
        prev = x.iloc[i-n:i]
        if len(prev) != n:
            continue
        down_count = int((prev.Close < prev.Open).sum())
        up_count = int((prev.Close > prev.Open).sum())

        prior_high = float(x.High.iloc[i-cfg.breakout_lookback:i].max())
        prior_low = float(x.Low.iloc[i-cfg.breakout_lookback:i].min())

        if bull and down_count >= 1 and close > prior_high:
            x.at[x.index[i], 'signal'] = 1
            x.at[x.index[i], 'sl'] = close - cfg.atr_stop * a
            x.at[x.index[i], 'tp'] = close + cfg.rr * cfg.atr_stop * a
            x.at[x.index[i], 'reason'] = 'EMA trend | pullback | upside breakout'
        elif bear and up_count >= 1 and close < prior_low:
            x.at[x.index[i], 'signal'] = -1
            x.at[x.index[i], 'sl'] = close + cfg.atr_stop * a
            x.at[x.index[i], 'tp'] = close - cfg.rr * cfg.atr_stop * a
            x.at[x.index[i], 'reason'] = 'EMA trend | pullback | downside breakout'

    # Cooldown is applied after signal generation.
    last = -10**9
    for i in np.flatnonzero(x.signal.to_numpy() != 0):
        if i - last < cfg.cooldown_bars:
            x.iloc[i, x.columns.get_loc('signal')] = 0
            x.iloc[i, x.columns.get_loc('sl')] = np.nan
            x.iloc[i, x.columns.get_loc('tp')] = np.nan
            x.iloc[i, x.columns.get_loc('reason')] = ''
        else:
            last = i
    return x


def backtest(df, cfg=PullbackConfig(), initial_equity=10000):
    s = build_signals(df, cfg)
    equity = float(initial_equity)
    pos = None
    trades = []
    curve = []

    for i in range(len(s)):
        row = s.iloc[i]
        ts = s.index[i]

        if pos is not None:
            hi, lo = float(row.High), float(row.Low)
            side = pos['side']
            hit_sl = lo <= pos['sl'] if side == 1 else hi >= pos['sl']
            hit_tp = hi >= pos['tp'] if side == 1 else lo <= pos['tp']
            r = None
            reason = None
            # Conservative same-bar ordering: stop wins if both are touched.
            if hit_sl:
                r = -1.0
                reason = 'SL'
            elif hit_tp:
                r = cfg.rr
                reason = 'TP'
            if r is not None:
                equity *= 1 + (cfg.risk_pct / 100.0) * r
                trades.append({
                    'entry_time': pos['entry_time'], 'exit_time': ts,
                    'side': side, 'entry': pos['entry'], 'sl': pos['sl'],
                    'tp': pos['tp'], 'R': r, 'equity': equity,
                    'reason': pos['reason'] + ' | ' + reason
                })
                pos = None

        # Signal is acted on at the NEXT bar open, avoiding same-bar lookahead.
        if pos is None and i + 1 < len(s) and int(row.signal) != 0:
            nxt = s.iloc[i + 1]
            entry = float(nxt.Open)
            if int(row.signal) == 1:
                stop = float(row.sl) + (entry - float(row.Close))
                target = entry + (entry - stop) * cfg.rr
            else:
                stop = float(row.sl) + (entry - float(row.Close))
                target = entry - (stop - entry) * cfg.rr
            if (int(row.signal) == 1 and stop < entry) or (int(row.signal) == -1 and stop > entry):
                pos = {
                    'side': int(row.signal), 'entry_time': s.index[i + 1],
                    'entry': entry, 'sl': stop, 'tp': target,
                    'reason': str(row.reason)
                }
        curve.append(equity)

    t = pd.DataFrame(trades)
    e = pd.Series(curve, index=s.index[:len(curve)], dtype='float64')
    if t.empty:
        metrics = {'num_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                   'total_R': 0.0, 'expectancy_R': 0.0, 'max_drawdown_pct': 0.0,
                   'total_return_pct': 0.0, 'final_equity': equity}
    else:
        wins = float(t.loc[t.R > 0, 'R'].sum())
        losses = float(-t.loc[t.R < 0, 'R'].sum())
        dd = float((e / e.cummax() - 1).min() * 100)
        metrics = {
            'num_trades': int(len(t)),
            'win_rate': round(float((t.R > 0).mean() * 100), 2),
            'profit_factor': round(wins / losses, 2) if losses else float('inf'),
            'total_R': round(float(t.R.sum()), 2),
            'expectancy_R': round(float(t.R.mean()), 3),
            'max_drawdown_pct': round(dd, 2),
            'total_return_pct': round(float((equity / initial_equity - 1) * 100), 2),
            'final_equity': round(equity, 2)
        }
    return {'signals': s, 'trades': t, 'equity_curve': e, 'metrics': metrics}


def optimize(df, base=None, initial_equity=10000, min_trades=20):
    base = base or PullbackConfig()
    rows = []
    for fast in (34, 50, 75):
        for slow in (150, 200, 250):
            if fast >= slow:
                continue
            for stop in (1.0, 1.2, 1.5):
                for rr in (1.5, 1.8, 2.2):
                    cfg = PullbackConfig(**{**base.__dict__, 'ema_fast': fast,
                                            'ema_slow': slow, 'atr_stop': stop, 'rr': rr})
                    r = backtest(df, cfg, initial_equity)
                    m = r['metrics']
                    rows.append({
                        'ema_fast': fast, 'ema_slow': slow, 'atr_stop': stop, 'rr': rr,
                        'num_trades': m['num_trades'], 'profit_factor': m['profit_factor'],
                        'total_R': m['total_R'], 'expectancy_R': m['expectancy_R'],
                        'win_rate': m['win_rate'], 'max_drawdown_pct': m['max_drawdown_pct'],
                        'total_return_pct': m['total_return_pct']
                    })
    eligible = [r for r in rows if r['num_trades'] >= int(min_trades)]
    pool = eligible or rows
    pool.sort(key=lambda r: (
        -999 if not np.isfinite(r['profit_factor']) else r['profit_factor'],
        r['expectancy_R'], r['total_R']
    ), reverse=True)
    return {'tested': len(rows), 'eligible': len(eligible), 'min_trades': int(min_trades),
            'results': pool[:20]}
