"""
indicators.py
Standard, textbook technical indicators. Vectorized with pandas/numpy.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """Vectorized fractal pivot high: True where `high` is the max in the
    centered [left, right] window. Confirmed only `right` bars later in
    practice (the caller should treat a pivot at bar i as known starting
    at bar i+right)."""
    window = left + right + 1
    roll_max = high.rolling(window, center=True).max()
    return high.where(high == roll_max)


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    roll_min = low.rolling(window, center=True).min()
    return low.where(low == roll_min)


FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]


def fibonacci_levels(swing_low: float, swing_high: float, uptrend: bool) -> dict:
    """
    Returns ALL standard Fibonacci levels (retracement + common extensions)
    for the leg between swing_low and swing_high, labeled by their
    percentage. `uptrend=True` means the leg ran low->high (retracement
    levels measured back down from the high); `uptrend=False` mirrors it
    (leg ran high->low, retracement measured back up from the low).
    """
    rng = swing_high - swing_low
    levels = {}
    for pct in FIB_LEVELS:
        if uptrend:
            price = swing_high - rng * pct
        else:
            price = swing_low + rng * pct
        levels[pct] = price
    return levels
