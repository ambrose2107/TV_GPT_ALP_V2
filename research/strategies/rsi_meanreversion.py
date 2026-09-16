"""
research/strategies/rsi_meanreversion.py
Simple RSI mean-reversion: fade extreme RSI readings, exit on RSI reverting
to the midline (or via the shared ATR stop / R-multiple target). Useful as
a contrasting style vs. the trend-following strategies for comparison.
"""
from .base import register
from .indicators import rsi, atr


def _generate_signals(df, params):
    d = df.copy()
    d["rsi"] = rsi(d["Close"], int(params["rsi_len"]))
    d["atr"] = atr(d, int(params["atr_len"]))

    rsi_prev = d["rsi"].shift(1)
    long_cond = (rsi_prev >= params["rsi_extreme_low"]) & (d["rsi"] < params["rsi_extreme_low"])
    short_cond = (rsi_prev <= params["rsi_extreme_high"]) & (d["rsi"] > params["rsi_extreme_high"])

    d["signal"] = 0
    d.loc[long_cond, "signal"] = 1
    if params.get("allow_shorts"):
        d.loc[short_cond, "signal"] = -1
    return d


register({
    "id": "rsi_meanreversion",
    "name": "RSI Mean Reversion",
    "description": "Fades extreme RSI readings, betting on a snap-back toward the mean.",
    "default_params": {
        "rsi_len": 14, "rsi_extreme_low": 20, "rsi_extreme_high": 80,
        "atr_len": 14, "atr_mult_sl": 1.2, "r_multiple": 1.5,
        "allow_shorts": True,
        "use_trailing": False, "breakeven_at_R": 1.0, "trail_atr_mult": 1.2,
    },
    "param_schema": {
        "rsi_len": {"type": "int", "min": 5, "max": 30, "step": 1},
        "rsi_extreme_low": {"type": "int", "min": 5, "max": 30, "step": 1},
        "rsi_extreme_high": {"type": "int", "min": 70, "max": 95, "step": 1},
        "atr_mult_sl": {"type": "float", "min": 0.5, "max": 4.0, "step": 0.1},
        "r_multiple": {"type": "float", "min": 0.5, "max": 4.0, "step": 0.1},
        "allow_shorts": {"type": "bool"},
    },
    "generate_signals": _generate_signals,
})
