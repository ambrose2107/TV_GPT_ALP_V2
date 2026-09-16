"""
research/strategies/indicators.py
Standard, textbook technical indicators shared by all registered strategies.
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
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def swing_levels(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    high, low = df["High"], df["Low"]
    window = left + right + 1
    roll_max = high.rolling(window, center=True).max()
    roll_min = low.rolling(window, center=True).min()
    swing_high = high.where(high == roll_max)
    swing_low = low.where(low == roll_min)
    out = pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low}, index=df.index)
    out["nearest_res"] = out["swing_high"].ffill()
    out["nearest_sup"] = out["swing_low"].ffill()
    return out


def rolling_vol_percentile(atr_series: pd.Series, lookback: int = 200) -> pd.Series:
    return atr_series.rolling(lookback, min_periods=20).rank(pct=True).fillna(0.5)
