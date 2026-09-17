"""
structure.py
Market structure primitives shared by both timeframes:
  - confirmed swing highs/lows (fractal pivots)
  - structure trend: sequence of higher-highs/higher-lows (bullish) vs
    lower-highs/lower-lows (bearish)
  - BOS (break of structure): close breaks beyond the most recent relevant
    confirmed swing point

No lookahead: a pivot at bar i is only "known" starting at bar i+right
(the confirmation lag inherent to fractal pivots), and everything here
respects that by construction (rolling window + explicit shift).
"""
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def swing_points(df: pd.DataFrame, left: int, right: int) -> dict:
    """
    Returns confirmed swing highs/lows as sparse Series (NaN except at the
    confirming bar, i.e. `right` bars after the actual extreme) plus
    forward-filled "last known swing high/low" series for convenience.
    """
    window = left + right + 1
    roll_max = df["High"].rolling(window, center=True).max()
    roll_min = df["Low"].rolling(window, center=True).min()

    swing_high = df["High"].where(df["High"] == roll_max)
    swing_low = df["Low"].where(df["Low"] == roll_min)

    # a centered rolling window "knows" the future by `right` bars - shift
    # the confirmed value forward by `right` bars so it only becomes known
    # once those bars have actually printed (no lookahead)
    swing_high_confirmed = swing_high.shift(right)
    swing_low_confirmed = swing_low.shift(right)

    return {
        "swing_high": swing_high_confirmed,
        "swing_low": swing_low_confirmed,
        "last_swing_high": swing_high_confirmed.ffill(),
        "last_swing_low": swing_low_confirmed.ffill(),
    }


def market_structure(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """
    Classifies structure trend bar-by-bar using confirmed swing points:
      - bullish: most recent confirmed swing high > the one before it, AND
                 most recent confirmed swing low > the one before it (HH+HL)
      - bearish: mirror (LH+LL)
      - neutral: anything else (mixed / insufficient history)

    Also flags BOS events:
      - bos_bull: close breaks above the last confirmed swing high
      - bos_bear: close breaks below the last confirmed swing low
    (Interpret as continuation-BOS or reversal-MSS/CHoCH based on the
    prevailing trend at the time - the caller decides which meaning
    applies for its purpose.)
    """
    sp = swing_points(df, left, right)
    out = df.copy()
    out["swing_high"] = sp["swing_high"]
    out["swing_low"] = sp["swing_low"]
    out["last_swing_high"] = sp["last_swing_high"]
    out["last_swing_low"] = sp["last_swing_low"]

    sh_series = sp["swing_high"].dropna()
    sl_series = sp["swing_low"].dropna()

    prev_swing_high = pd.Series(np.nan, index=out.index)
    prev_swing_low = pd.Series(np.nan, index=out.index)

    if len(sh_series) >= 2:
        shifted = sh_series.shift(1)
        prev_swing_high = shifted.reindex(out.index).ffill()
    if len(sl_series) >= 2:
        shifted = sl_series.shift(1)
        prev_swing_low = shifted.reindex(out.index).ffill()

    out["prev_swing_high"] = prev_swing_high
    out["prev_swing_low"] = prev_swing_low

    higher_high = out["last_swing_high"] > out["prev_swing_high"]
    higher_low = out["last_swing_low"] > out["prev_swing_low"]
    lower_high = out["last_swing_high"] < out["prev_swing_high"]
    lower_low = out["last_swing_low"] < out["prev_swing_low"]

    trend = pd.Series("neutral", index=out.index)
    trend[higher_high.fillna(False) & higher_low.fillna(False)] = "bullish"
    trend[lower_high.fillna(False) & lower_low.fillna(False)] = "bearish"
    out["structure_trend"] = trend

    out["bos_bull"] = out["Close"] > out["last_swing_high"].shift(1)
    out["bos_bear"] = out["Close"] < out["last_swing_low"].shift(1)

    return out
