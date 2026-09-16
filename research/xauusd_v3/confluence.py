"""
confluence.py
Implements the 8 confluence concepts, each as an independent, togglable
layer - same philosophy as the Pine v3 script:

  1. Supply & Demand zones   -> pivot + impulsive ATR move
  2. Volume POI              -> volume-spike-bar proxy (simplified from true
                                 tick-level volume profile, which isn't
                                 practical to compute bar-by-bar here either)
  3. Inefficiencies (FVG)    -> classic 3-candle fair value gap
  4. Liquidity               -> sweep of recent unmitigated swing extreme
  5. Fibonacci levels        -> ALL standard levels marked (not just one
                                 zone), plus the 0.5-0.786 zone as a filter
  6. Confluence score        -> counts how many of {1,2,3,4,5} align
                                 ("Accuracy Zones", operationalized honestly)
  7. DXY correlation         -> DXY's own EMA trend, inverse to gold
  8. Silver correlation      -> Silver's own EMA trend, same direction as gold

Every confluence layer returns boolean Series (long/short) so the caller
can log a diagnostic funnel (how many bars pass each filter) - this is
what actually answers "why did no trades fire".

Performance: the sequential zone-tracking loops (liquidity, supply/demand,
FVG, volume POI) are numba-JIT-compiled when available, with a pure-Python
fallback - same pattern as engine.py.
"""
import numpy as np
import pandas as pd

from .indicators import ema, atr, pivot_high, pivot_low

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrap



def fibonacci_confluence(df: pd.DataFrame, atr_series: pd.Series, swing_lookback: int,
                          zone_min: float, zone_max: float) -> dict:
    swing_high = df["High"].rolling(swing_lookback).max()
    swing_low = df["Low"].rolling(swing_lookback).min()
    fib_range = swing_high - swing_low

    zone_long_lo = swing_high - fib_range * zone_max
    zone_long_hi = swing_high - fib_range * zone_min
    zone_short_lo = swing_low + fib_range * zone_min
    zone_short_hi = swing_low + fib_range * zone_max

    long_ok = (df["Close"] >= zone_long_lo) & (df["Close"] <= zone_long_hi)
    short_ok = (df["Close"] >= zone_short_lo) & (df["Close"] <= zone_short_hi)

    return {
        "long": long_ok.fillna(False), "short": short_ok.fillna(False),
        "swing_high": swing_high, "swing_low": swing_low,
        "zone_long_lo": zone_long_lo, "zone_long_hi": zone_long_hi,
        "zone_short_lo": zone_short_lo, "zone_short_hi": zone_short_hi,
    }


def _liquidity_core(low, high, close, liq_high, liq_low, sweep_window):
    n = len(low)
    swept_low = np.zeros(n, dtype=np.bool_)
    swept_high = np.zeros(n, dtype=np.bool_)
    last_bull = np.empty(n, dtype=np.int64)
    last_bear = np.empty(n, dtype=np.int64)
    lb, lr = -1_000_000_000, -1_000_000_000
    for i in range(n):
        sl = (low[i] < liq_low[i]) and (close[i] > liq_low[i]) if not np.isnan(liq_low[i]) else False
        sh = (high[i] > liq_high[i]) and (close[i] < liq_high[i]) if not np.isnan(liq_high[i]) else False
        swept_low[i] = sl
        swept_high[i] = sh
        if sl:
            lb = i
        if sh:
            lr = i
        last_bull[i] = lb
        last_bear[i] = lr

    idx = np.arange(n)
    long_ok = (idx - last_bull) <= sweep_window
    short_ok = (idx - last_bear) <= sweep_window
    return swept_low, swept_high, long_ok, short_ok


_liquidity_core_jit = njit(cache=True)(_liquidity_core) if NUMBA_AVAILABLE else _liquidity_core


def liquidity_confluence(df: pd.DataFrame, lookback: int, sweep_window: int) -> dict:
    liq_high = df["High"].rolling(lookback).max().shift(1)
    liq_low = df["Low"].rolling(lookback).min().shift(1)

    swept_low, swept_high, long_ok, short_ok = _liquidity_core_jit(
        df["Low"].values.astype(np.float64), df["High"].values.astype(np.float64),
        df["Close"].values.astype(np.float64), liq_high.values.astype(np.float64),
        liq_low.values.astype(np.float64), int(sweep_window),
    )

    return {
        "long": pd.Series(long_ok, index=df.index), "short": pd.Series(short_ok, index=df.index),
        "liq_high": liq_high, "liq_low": liq_low,
        "swept_low": pd.Series(swept_low, index=df.index), "swept_high": pd.Series(swept_high, index=df.index),
    }


def _supply_demand_core(close, atr_v, piv_low_v, piv_high_v, impulse_atr_mult, zone_height_atr_mult):
    n = len(close)
    demand_lo = np.full(n, np.nan)
    demand_hi = np.full(n, np.nan)
    demand_valid = np.zeros(n, dtype=np.bool_)
    supply_lo = np.full(n, np.nan)
    supply_hi = np.full(n, np.nan)
    supply_valid = np.zeros(n, dtype=np.bool_)

    d_lo, d_hi, d_valid = np.nan, np.nan, False
    s_lo, s_hi, s_valid = np.nan, np.nan, False

    for i in range(n):
        pl = piv_low_v[i]
        if (not np.isnan(pl)) and (close[i] - pl) > impulse_atr_mult * atr_v[i]:
            d_lo = pl
            d_hi = pl + zone_height_atr_mult * atr_v[i]
            d_valid = True
        if d_valid and close[i] < d_lo:
            d_valid = False

        ph = piv_high_v[i]
        if (not np.isnan(ph)) and (ph - close[i]) > impulse_atr_mult * atr_v[i]:
            s_hi = ph
            s_lo = ph - zone_height_atr_mult * atr_v[i]
            s_valid = True
        if s_valid and close[i] > s_hi:
            s_valid = False

        demand_lo[i], demand_hi[i], demand_valid[i] = d_lo, d_hi, d_valid
        supply_lo[i], supply_hi[i], supply_valid[i] = s_lo, s_hi, s_valid

    return demand_lo, demand_hi, demand_valid, supply_lo, supply_hi, supply_valid


_supply_demand_core_jit = njit(cache=True)(_supply_demand_core) if NUMBA_AVAILABLE else _supply_demand_core


def supply_demand_confluence(df: pd.DataFrame, atr_series: pd.Series, pivot_left: int,
                              pivot_right: int, impulse_atr_mult: float, zone_height_atr_mult: float) -> dict:
    piv_low = pivot_low(df["Low"], pivot_left, pivot_right)
    piv_high = pivot_high(df["High"], pivot_left, pivot_right)

    demand_lo, demand_hi, demand_valid, supply_lo, supply_hi, supply_valid = _supply_demand_core_jit(
        df["Close"].values.astype(np.float64), atr_series.values.astype(np.float64),
        piv_low.values.astype(np.float64), piv_high.values.astype(np.float64),
        float(impulse_atr_mult), float(zone_height_atr_mult),
    )

    close_v = df["Close"].values
    long_ok = demand_valid & (close_v >= demand_lo) & (close_v <= demand_hi)
    short_ok = supply_valid & (close_v <= supply_hi) & (close_v >= supply_lo)

    return {
        "long": pd.Series(long_ok, index=df.index), "short": pd.Series(short_ok, index=df.index),
        "demand_lo": pd.Series(demand_lo, index=df.index), "demand_hi": pd.Series(demand_hi, index=df.index),
        "demand_valid": pd.Series(demand_valid, index=df.index),
        "supply_lo": pd.Series(supply_lo, index=df.index), "supply_hi": pd.Series(supply_hi, index=df.index),
        "supply_valid": pd.Series(supply_valid, index=df.index),
    }


def _fvg_core(high, low, close, bull_fvg, bear_fvg):
    n = len(close)
    fvg_bull_lo = np.full(n, np.nan)
    fvg_bull_hi = np.full(n, np.nan)
    fvg_bull_valid = np.zeros(n, dtype=np.bool_)
    fvg_bear_lo = np.full(n, np.nan)
    fvg_bear_hi = np.full(n, np.nan)
    fvg_bear_valid = np.zeros(n, dtype=np.bool_)

    b_lo, b_hi, b_valid = np.nan, np.nan, False
    r_lo, r_hi, r_valid = np.nan, np.nan, False

    for i in range(n):
        if bull_fvg[i]:
            b_lo, b_hi, b_valid = high[i - 2], low[i], True
        if b_valid and close[i] < b_lo:
            b_valid = False

        if bear_fvg[i]:
            r_hi, r_lo, r_valid = low[i - 2], high[i], True
        if r_valid and close[i] > r_hi:
            r_valid = False

        fvg_bull_lo[i], fvg_bull_hi[i], fvg_bull_valid[i] = b_lo, b_hi, b_valid
        fvg_bear_lo[i], fvg_bear_hi[i], fvg_bear_valid[i] = r_lo, r_hi, r_valid

    return fvg_bull_lo, fvg_bull_hi, fvg_bull_valid, fvg_bear_lo, fvg_bear_hi, fvg_bear_valid


_fvg_core_jit = njit(cache=True)(_fvg_core) if NUMBA_AVAILABLE else _fvg_core


def fvg_confluence(df: pd.DataFrame) -> dict:
    high, low, close = df["High"].values.astype(np.float64), df["Low"].values.astype(np.float64), df["Close"].values.astype(np.float64)
    n = len(df)

    bull_fvg = np.zeros(n, dtype=np.bool_)
    bear_fvg = np.zeros(n, dtype=np.bool_)
    bull_fvg[2:] = low[2:] > high[:-2]
    bear_fvg[2:] = high[2:] < low[:-2]

    (fvg_bull_lo, fvg_bull_hi, fvg_bull_valid,
     fvg_bear_lo, fvg_bear_hi, fvg_bear_valid) = _fvg_core_jit(high, low, close, bull_fvg, bear_fvg)

    long_ok = fvg_bull_valid & (close >= fvg_bull_lo) & (close <= fvg_bull_hi)
    short_ok = fvg_bear_valid & (close <= fvg_bear_hi) & (close >= fvg_bear_lo)

    return {
        "long": pd.Series(long_ok, index=df.index), "short": pd.Series(short_ok, index=df.index),
        "fvg_bull_lo": pd.Series(fvg_bull_lo, index=df.index), "fvg_bull_hi": pd.Series(fvg_bull_hi, index=df.index),
        "fvg_bull_valid": pd.Series(fvg_bull_valid, index=df.index),
        "fvg_bear_lo": pd.Series(fvg_bear_lo, index=df.index), "fvg_bear_hi": pd.Series(fvg_bear_hi, index=df.index),
        "fvg_bear_valid": pd.Series(fvg_bear_valid, index=df.index),
    }


def _poi_core(low, high, close, atr_v, is_hv, tolerance_atr):
    n = len(close)
    poi_lo = np.full(n, np.nan)
    poi_hi = np.full(n, np.nan)
    poi_valid = np.zeros(n, dtype=np.bool_)
    p_lo, p_hi, p_valid = np.nan, np.nan, False
    for i in range(n):
        if is_hv[i]:
            p_lo, p_hi, p_valid = low[i], high[i], True
        poi_lo[i], poi_hi[i], poi_valid[i] = p_lo, p_hi, p_valid

    confluence = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if poi_valid[i]:
            confluence[i] = (close[i] >= poi_lo[i] - tolerance_atr * atr_v[i]) and \
                             (close[i] <= poi_hi[i] + tolerance_atr * atr_v[i])
    return poi_lo, poi_hi, poi_valid, confluence


_poi_core_jit = njit(cache=True)(_poi_core) if NUMBA_AVAILABLE else _poi_core


def volume_poi_confluence(df: pd.DataFrame, atr_series: pd.Series, vol_lookback: int,
                           vol_mult: float, tolerance_atr: float) -> dict:
    vol_avg = df["Volume"].rolling(vol_lookback).mean()
    is_high_vol = (df["Volume"] > vol_avg * vol_mult).fillna(False)

    poi_lo, poi_hi, poi_valid, confluence = _poi_core_jit(
        df["Low"].values.astype(np.float64), df["High"].values.astype(np.float64),
        df["Close"].values.astype(np.float64), atr_series.values.astype(np.float64),
        is_high_vol.values.astype(np.bool_), float(tolerance_atr),
    )

    return {
        "confluence": pd.Series(confluence, index=df.index),  # direction-agnostic
        "poi_lo": pd.Series(poi_lo, index=df.index), "poi_hi": pd.Series(poi_hi, index=df.index),
        "poi_valid": pd.Series(poi_valid, index=df.index), "is_high_vol_bar": is_high_vol,
    }


def intermarket_confluence(gold_df: pd.DataFrame, other_df: pd.DataFrame,
                            fast_len: int, slow_len: int, inverse: bool) -> dict:
    """
    other_df must already be reindexed/aligned to gold_df's index.
    inverse=True (DXY): downtrend in `other` -> bullish confluence for gold.
    inverse=False (Silver): uptrend in `other` -> bullish confluence for gold.
    """
    if other_df is None:
        n = len(gold_df)
        na_series = pd.Series(np.full(n, False), index=gold_df.index)
        return {"long": na_series, "short": na_series, "available": False}

    fast = ema(other_df["Close"], fast_len)
    slow = ema(other_df["Close"], slow_len)
    other_up = fast > slow
    other_down = fast < slow

    if inverse:
        long_ok = other_down   # DXY rolling over -> bullish gold
        short_ok = other_up
    else:
        long_ok = other_up     # Silver rising with gold -> bullish confirmation
        short_ok = other_down

    return {"long": long_ok.fillna(False), "short": short_ok.fillna(False), "available": True,
            "trend_up": other_up, "trend_down": other_down}


def confluence_score(long_flags: dict, short_flags: dict) -> dict:
    """
    long_flags / short_flags: dict of {name: pd.Series[bool]} for the
    ZONE-based confluences only (supply/demand, fvg, poi, liquidity, fib) -
    intermarket (DXY/Silver) is kept separate since it's a different kind
    of confirmation (macro, not price-geometry).
    """
    long_score = sum(s.astype(int) for s in long_flags.values())
    short_score = sum(s.astype(int) for s in short_flags.values())
    return {"long_score": long_score, "short_score": short_score, "max_score": len(long_flags)}
