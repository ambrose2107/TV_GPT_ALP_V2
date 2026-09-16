"""
research/strategies/ema_crossover.py
The same 9/21 EMA crossover + 50 EMA trend filter used by
research/backtester.py and research/strategy_signals.py, generalized so it
can run on any timeframe/interval (not just daily) via the shared backtest
engine, for apples-to-apples comparison against other registered strategies.
"""
from .base import register
from .indicators import ema


def _generate_signals(df, params):
    d = df.copy()
    d["ema_f"] = ema(d["Close"], int(params["ema_fast"]))
    d["ema_s"] = ema(d["Close"], int(params["ema_slow"]))
    d["ema_trend"] = ema(d["Close"], int(params["ema_trend"]))

    f_prev, s_prev = d["ema_f"].shift(1), d["ema_s"].shift(1)
    bullish_cross = (f_prev <= s_prev) & (d["ema_f"] > d["ema_s"])
    bearish_cross = (f_prev >= s_prev) & (d["ema_f"] < d["ema_s"])

    trend_up = d["Close"] > d["ema_trend"]
    trend_down = d["Close"] < d["ema_trend"]

    d["signal"] = 0
    d.loc[bullish_cross & trend_up, "signal"] = 1
    if params.get("allow_shorts"):
        d.loc[bearish_cross & trend_down, "signal"] = -1
    return d


register({
    "id": "ema_crossover",
    "name": "EMA 9/21/50 Crossover",
    "description": "Classic fast/slow EMA crossover filtered by a longer-term trend EMA.",
    "default_params": {
        "ema_fast": 9, "ema_slow": 21, "ema_trend": 50,
        "atr_len": 14, "atr_mult_sl": 1.5, "r_multiple": 2.0,
        "allow_shorts": False,
        "use_trailing": True, "breakeven_at_R": 1.0, "trail_atr_mult": 1.5,
    },
    "param_schema": {
        "ema_fast": {"type": "int", "min": 3, "max": 30, "step": 1},
        "ema_slow": {"type": "int", "min": 10, "max": 60, "step": 1},
        "ema_trend": {"type": "int", "min": 20, "max": 200, "step": 5},
        "atr_mult_sl": {"type": "float", "min": 0.5, "max": 4.0, "step": 0.1},
        "r_multiple": {"type": "float", "min": 1.0, "max": 5.0, "step": 0.1},
        "allow_shorts": {"type": "bool"},
        "use_trailing": {"type": "bool"},
    },
    "generate_signals": _generate_signals,
})
