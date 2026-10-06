"""
Gold Aroon Money Flow Confluence V2.

Port of the user's TradingView "Aroon Money Flow Confluence" signal logic
into the dashboard backtest registry.  The backtest engine executes a signal
at the next bar open and supplies the ATR stop/target.

Core:
- Aroon oscillator: Aroon Up - Aroon Down, length 14 by default.
- Chaikin Money Flow, length 20 by default.
- Long when Aroon is bullish and CMF crosses above the positive flow buffer.
- Short when Aroon is bearish and CMF crosses below the negative flow buffer.
- Aroon flip must have occurred within the confirmation window.
- Optional ADX strength filter.
- No lookahead / no future-bar information.

The TradingView script has HTF confirmation, smoothing, three targets and
alert/trade-state management.  Those are intentionally not duplicated here
because the current V2 backtest engine owns execution, stop/target and trade
accounting.  The strategy exposes the signal-side parameters that can be
tested without turning the model into an overfit parameter grid.
"""
import pandas as pd

from .base import register
from .indicators import atr


def _aroon(series, length):
    # TradingView-compatible Aroon value over length + 1 bars.
    roll = series.rolling(length + 1, min_periods=length + 1)
    # Position of most recent extreme: 0 = current bar, length = oldest bar.
    def age(values, find_max=True):
        if find_max:
            pos = len(values) - 1 - values.argmax()
        else:
            pos = len(values) - 1 - values.argmin()
        return pos

    up_age = roll.apply(lambda x: age(x, True), raw=True)
    dn_age = roll.apply(lambda x: age(x, False), raw=True)
    up = 100.0 * (length - up_age) / length
    dn = 100.0 * (length - dn_age) / length
    return up, dn


def _cmf(df, length):
    hl = df["High"] - df["Low"]
    mfv = ((2.0 * df["Close"] - df["High"] - df["Low"]) / hl.replace(0, float("nan"))) * df["Volume"]
    vol = df["Volume"].where(hl > 0, 0.0)
    return mfv.rolling(length, min_periods=length).sum() / vol.rolling(length, min_periods=length).sum()


def _adx(df, length):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr_w
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr_w
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _generate_signals(df, params):
    d = df.copy()
    aroon_len = int(params["aroon_len"])
    cmf_len = int(params["cmf_len"])
    flow_buffer = float(params["flow_buffer"])
    confirm_bars = int(params["confirm_bars"])

    up, dn = _aroon(d["High"], aroon_len)
    d["aroon_up"] = up
    d["aroon_down"] = dn
    d["aroon"] = up - dn
    d["cmf"] = _cmf(d, cmf_len)
    d["atr"] = atr(d, int(params["atr_len"]))

    # Detect the Aroon zero-line flip without using future bars.
    bull_flip = (d["aroon"] > 0) & (d["aroon"].shift(1) <= 0)
    bear_flip = (d["aroon"] < 0) & (d["aroon"].shift(1) >= 0)

    # Bars since the latest Aroon flip.  This mirrors the Pine confirmation
    # concept while remaining vectorized and cheap on Render.
    bull_age = bull_flip.astype(int).groupby(bull_flip.cumsum()).cumcount()
    bear_age = bear_flip.astype(int).groupby(bear_flip.cumsum()).cumcount()
    have_bull = bull_flip.cumsum() > 0
    have_bear = bear_flip.cumsum() > 0
    bull_age = bull_age.where(have_bull)
    bear_age = bear_age.where(have_bear)

    agree_bull = (d["aroon"] > 0) & (d["cmf"] > flow_buffer)
    agree_bear = (d["aroon"] < 0) & (d["cmf"] < -flow_buffer)

    # Pine emits a signal when money flow starts agreeing, provided the
    # Aroon flip happened recently.
    long_cond = agree_bull & ~agree_bull.shift(1).fillna(False) & (bull_age <= confirm_bars)
    short_cond = agree_bear & ~agree_bear.shift(1).fillna(False) & (bear_age <= confirm_bars)

    if params.get("use_adx"):
        d["adx"] = _adx(d, int(params["adx_len"]))
        adx_ok = d["adx"].shift(1) >= float(params["adx_min"])
        long_cond &= adx_ok
        short_cond &= adx_ok

    # Zero volume means CMF cannot represent money flow; do not manufacture
    # signals from it.
    if "Volume" in d.columns:
        long_cond &= d["Volume"].rolling(cmf_len, min_periods=cmf_len).sum() > 0
        short_cond &= d["Volume"].rolling(cmf_len, min_periods=cmf_len).sum() > 0

    d["signal"] = 0
    d.loc[long_cond.fillna(False), "signal"] = 1
    d.loc[short_cond.fillna(False), "signal"] = -1
    return d


register({
    "id": "gold_aroon_money_flow_v2",
    "name": "Gold Aroon + Money Flow Confluence V2",
    "description": "Aroon zero-line trend flip confirmed by Chaikin Money Flow; designed for Gold/XAUUSD or GC=F backtesting.",
    "default_params": {
        "aroon_len": 14,
        "cmf_len": 20,
        "flow_buffer": 0.05,
        "confirm_bars": 5,
        "atr_len": 14,
        "atr_mult_sl": 1.5,
        "r_multiple": 2.0,
        "use_adx": False,
        "adx_len": 14,
        "adx_min": 20.0,
        "use_trailing": False,
        "breakeven_at_R": 1.0,
        "trail_atr_mult": 1.2,
    },
    "param_schema": {
        "aroon_len": {"type": "int", "min": 5, "max": 50, "step": 1},
        "cmf_len": {"type": "int", "min": 5, "max": 60, "step": 1},
        "flow_buffer": {"type": "float", "min": 0.0, "max": 0.20, "step": 0.01},
        "confirm_bars": {"type": "int", "min": 0, "max": 20, "step": 1},
        "atr_mult_sl": {"type": "float", "min": 0.75, "max": 3.0, "step": 0.25},
        "r_multiple": {"type": "float", "min": 1.0, "max": 4.0, "step": 0.25},
        "use_adx": {"type": "bool"},
        "adx_len": {"type": "int", "min": 5, "max": 30, "step": 1},
        "adx_min": {"type": "float", "min": 10.0, "max": 40.0, "step": 1.0},
        "use_trailing": {"type": "bool"},
    },
    "generate_signals": _generate_signals,
})
