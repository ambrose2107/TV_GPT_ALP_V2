"""
fibonacci.py
Detects "clean impulsive" swing legs on the 15m chart (filtering out tiny
or choppy moves, per the spec's explicit requirement), and computes ALL
standard Fibonacci levels for that leg, including the 0.618-0.786 primary
zone, the 0.705 midpoint, and the 1.272/1.618 extensions used as TP3/TP4.
"""
import numpy as np
import pandas as pd

FIB_LEVELS = [0.0, 0.382, 0.5, 0.618, 0.705, 0.786, 1.0, 1.272, 1.618]


def leg_efficiency(df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """Net displacement / total path length over [start_idx, end_idx] -
    close to 1.0 = a clean, direct move; close to 0 = choppy back-and-forth."""
    if end_idx <= start_idx:
        return 0.0
    closes = df["Close"].values[start_idx:end_idx + 1]
    net = abs(closes[-1] - closes[0])
    path = np.sum(np.abs(np.diff(closes)))
    return net / path if path > 0 else 0.0


def detect_impulses(df15: pd.DataFrame, swing_high_confirmed: pd.Series, swing_low_confirmed: pd.Series,
                     atr_series: pd.Series, min_efficiency: float = 0.35, min_atr_mult: float = 3.0) -> dict:
    """
    Scans confirmed swing points in time order and, whenever a swing low is
    immediately followed by a swing high (or vice versa) that together form
    a leg meeting the size + efficiency thresholds, marks it as a "clean
    impulse". Returns forward-filled per-bar Series describing the CURRENT
    active impulse (so downstream code just reads the latest row):
        impulse_dir ('bullish'/'bearish'/None), impulse_low, impulse_high,
        impulse_low_time, impulse_high_time
    A new impulse only overwrites the active one once it is itself
    confirmed - no lookahead.
    """
    n = len(df15)
    sh = swing_high_confirmed
    sl = swing_low_confirmed

    # merge all confirmed pivots into one time-ordered sequence of (idx, type, price)
    points = []
    sh_idx = np.where(~sh.isna().values)[0]
    sl_idx = np.where(~sl.isna().values)[0]
    for i in sh_idx:
        points.append((i, "H", sh.values[i]))
    for i in sl_idx:
        points.append((i, "L", sl.values[i]))
    points.sort(key=lambda x: x[0])

    impulse_dir = np.full(n, None, dtype=object)
    impulse_low = np.full(n, np.nan)
    impulse_high = np.full(n, np.nan)
    impulse_low_i = np.full(n, -1, dtype=np.int64)
    impulse_high_i = np.full(n, -1, dtype=np.int64)

    cur_dir, cur_lo, cur_hi, cur_lo_i, cur_hi_i = None, np.nan, np.nan, -1, -1
    atr_v = atr_series.values

    for k in range(1, len(points)):
        i_prev, t_prev, p_prev = points[k - 1]
        i_cur, t_cur, p_cur = points[k]
        if t_prev == "L" and t_cur == "H" and p_cur > p_prev:
            size_ok = (p_cur - p_prev) > min_atr_mult * atr_v[i_cur] if not np.isnan(atr_v[i_cur]) else False
            eff = leg_efficiency(df15, i_prev, i_cur)
            if size_ok and eff >= min_efficiency:
                cur_dir, cur_lo, cur_hi, cur_lo_i, cur_hi_i = "bullish", p_prev, p_cur, i_prev, i_cur
        elif t_prev == "H" and t_cur == "L" and p_cur < p_prev:
            size_ok = (p_prev - p_cur) > min_atr_mult * atr_v[i_cur] if not np.isnan(atr_v[i_cur]) else False
            eff = leg_efficiency(df15, i_prev, i_cur)
            if size_ok and eff >= min_efficiency:
                cur_dir, cur_lo, cur_hi, cur_lo_i, cur_hi_i = "bearish", p_cur, p_prev, i_cur, i_prev

        # this impulse becomes "known" starting at bar i_cur (when the 2nd
        # pivot confirms) - fill forward from there until superseded
        if cur_dir is not None:
            fill_start = i_cur
            impulse_dir[fill_start:] = cur_dir
            impulse_low[fill_start:] = cur_lo
            impulse_high[fill_start:] = cur_hi
            impulse_low_i[fill_start:] = cur_lo_i
            impulse_high_i[fill_start:] = cur_hi_i

    return {
        "impulse_dir": pd.Series(impulse_dir, index=df15.index),
        "impulse_low": pd.Series(impulse_low, index=df15.index),
        "impulse_high": pd.Series(impulse_high, index=df15.index),
        "impulse_low_i": impulse_low_i,
        "impulse_high_i": impulse_high_i,
    }


def fib_levels_for_leg(low: float, high: float, direction: str) -> dict:
    """
    direction='bullish': leg ran low->high; retracement measured DOWN from
    high, extensions projected ABOVE high.
    direction='bearish': mirror.
    """
    rng = high - low
    levels = {}
    for pct in FIB_LEVELS:
        if direction == "bullish":
            levels[pct] = high - rng * pct
        else:
            levels[pct] = low + rng * pct
    return levels


def fib_zone_bounds(low: float, high: float, direction: str, zone_min: float = 0.618, zone_max: float = 0.786):
    levels = fib_levels_for_leg(low, high, direction)
    lo, hi = levels[zone_max], levels[zone_min]
    return (min(lo, hi), max(lo, hi))
