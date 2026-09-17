"""
confluence.py
The 2-3 confluence checks required alongside the Fib zone itself:
  - liquidity: a prior opposite swing point resting near the retracement
    zone (sell-side liquidity below for longs, buy-side above for shorts)
  - support/resistance: ANY other prior swing point (not the immediate
    liquidity one) overlapping the zone - a broader "this level mattered
    before" check
  - order block: the last opposite-colored candle immediately before the
    impulse began (classic SMC order block definition)

Also: 5-minute liquidity sweep detection (stop-hunt wick through a recent
swing extreme that closes back inside) and BOS/structure-shift check,
reused for the execution-timeframe confirmation.
"""
import numpy as np
import pandas as pd


def liquidity_confluence(zone_lo: float, zone_hi: float, liquidity_level: float, tolerance: float) -> bool:
    if liquidity_level is None or np.isnan(liquidity_level):
        return False
    return (zone_lo - tolerance) <= liquidity_level <= (zone_hi + tolerance)


def sr_confluence(zone_lo: float, zone_hi: float, other_levels: list, tolerance: float) -> bool:
    for lvl in other_levels:
        if lvl is None or (isinstance(lvl, float) and np.isnan(lvl)):
            continue
        if (zone_lo - tolerance) <= lvl <= (zone_hi + tolerance):
            return True
    return False


def find_order_block(df15: pd.DataFrame, leg_start_idx: int, direction: str, lookback: int = 3):
    """
    Bullish OB: the last down-close (bearish) candle in the few bars before
    the impulsive up-leg's starting swing low. Bearish OB: the last
    up-close candle before the impulsive down-leg's start. Returns
    (ob_low, ob_high) or (None, None) if none found.
    """
    start = max(0, leg_start_idx - lookback)
    window = df15.iloc[start:leg_start_idx + 1]
    if window.empty:
        return None, None

    if direction == "bullish":
        down_candles = window[window["Close"] < window["Open"]]
        if down_candles.empty:
            return None, None
        last = down_candles.iloc[-1]
    else:
        up_candles = window[window["Close"] > window["Open"]]
        if up_candles.empty:
            return None, None
        last = up_candles.iloc[-1]

    return float(last["Low"]), float(last["High"])


def sweep_and_bos(df5: pd.DataFrame, structure5: pd.DataFrame, lookback: int, sweep_window: int) -> dict:
    """
    5-minute confirmation primitives:
      - swept_low / swept_high: stop-hunt wick through the recent extreme
        that closes back inside
      - bull_confirm / bear_confirm: a BOS in the trade direction occurring
        within `sweep_window` bars of a matching-direction sweep
    """
    liq_high = df5["High"].rolling(lookback).max().shift(1)
    liq_low = df5["Low"].rolling(lookback).min().shift(1)

    swept_low = (df5["Low"] < liq_low) & (df5["Close"] > liq_low)
    swept_high = (df5["High"] > liq_high) & (df5["Close"] < liq_high)

    n = len(df5)
    last_bull_sweep = np.full(n, -10**9, dtype=np.int64)
    last_bear_sweep = np.full(n, -10**9, dtype=np.int64)
    sl_v, sh_v = swept_low.values, swept_high.values
    lb, lr = -10**9, -10**9
    for i in range(n):
        if sl_v[i]:
            lb = i
        if sh_v[i]:
            lr = i
        last_bull_sweep[i] = lb
        last_bear_sweep[i] = lr

    idx = np.arange(n)
    bull_confirm = structure5["bos_bull"].values & ((idx - last_bull_sweep) <= sweep_window)
    bear_confirm = structure5["bos_bear"].values & ((idx - last_bear_sweep) <= sweep_window)

    return {
        "swept_low": swept_low, "swept_high": swept_high,
        "liq_high": liq_high, "liq_low": liq_low,
        "bull_confirm": pd.Series(bull_confirm, index=df5.index),
        "bear_confirm": pd.Series(bear_confirm, index=df5.index),
        "last_bull_sweep_idx": last_bull_sweep, "last_bear_sweep_idx": last_bear_sweep,
    }
