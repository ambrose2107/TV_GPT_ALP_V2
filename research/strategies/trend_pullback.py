"""
research/strategies/trend_pullback.py
EMA trend filter + RSI pullback-and-resume entry, with optional swing
support/resistance confluence and a volatility (chop) filter.

This is the same logic built for the standalone XAUUSD script, adapted to
the dashboard's strategy registry interface.
"""
from .base import register
from .indicators import ema, rsi, atr, swing_levels, rolling_vol_percentile


def _generate_signals(df, params):
    d = df.copy()
    d["ema_fast"] = ema(d["Close"], int(params["ema_fast"]))
    d["ema_slow"] = ema(d["Close"], int(params["ema_slow"]))
    d["rsi"] = rsi(d["Close"], int(params["rsi_len"]))
    d["atr"] = atr(d, int(params["atr_len"]))
    d["trend_up"] = d["ema_fast"] > d["ema_slow"]
    d["trend_down"] = d["ema_fast"] < d["ema_slow"]

    rsi_prev = d["rsi"].shift(1)
    long_trigger = (rsi_prev < params["rsi_oversold"]) & (d["rsi"] >= params["rsi_oversold"])
    short_trigger = (rsi_prev > params["rsi_overbought"]) & (d["rsi"] <= params["rsi_overbought"])

    long_cond = d["trend_up"] & long_trigger
    short_cond = d["trend_down"] & short_trigger

    if params.get("use_vol_filter"):
        d["vol_pctile"] = rolling_vol_percentile(d["atr"])
        vol_ok = d["vol_pctile"] >= params["vol_percentile_min"]
        long_cond &= vol_ok
        short_cond &= vol_ok

    if params.get("use_bounce_filter"):
        sw = swing_levels(d, left=int(params["swing_left"]), right=int(params["swing_right"]))
        d = d.join(sw)
        tol = params["bounce_zone_atr_tol"] * d["atr"]
        near_sup = (d["Close"] - d["nearest_sup"]).abs() <= tol
        near_res = (d["nearest_res"] - d["Close"]).abs() <= tol
        long_cond &= near_sup.fillna(False)
        short_cond &= near_res.fillna(False)

    d["signal"] = 0
    d.loc[long_cond, "signal"] = 1
    d.loc[short_cond, "signal"] = -1
    return d


register({
    "id": "trend_pullback",
    "name": "EMA Trend + RSI Pullback",
    "description": "Trades pullbacks (RSI oversold/overbought resuming) within an EMA-defined trend.",
    "default_params": {
        "ema_fast": 20, "ema_slow": 50, "rsi_len": 14,
        "rsi_oversold": 35, "rsi_overbought": 65,
        "atr_len": 14, "atr_mult_sl": 1.5, "r_multiple": 2.0,
        "use_bounce_filter": True, "bounce_zone_atr_tol": 0.6,
        "swing_left": 5, "swing_right": 5,
        "use_vol_filter": True, "vol_percentile_min": 0.25,
        "use_trailing": True, "breakeven_at_R": 1.0, "trail_atr_mult": 1.2,
    },
    "param_schema": {
        "ema_fast": {"type": "int", "min": 5, "max": 50, "step": 1},
        "ema_slow": {"type": "int", "min": 20, "max": 200, "step": 5},
        "rsi_len": {"type": "int", "min": 5, "max": 30, "step": 1},
        "rsi_oversold": {"type": "int", "min": 10, "max": 45, "step": 1},
        "rsi_overbought": {"type": "int", "min": 55, "max": 90, "step": 1},
        "atr_mult_sl": {"type": "float", "min": 0.5, "max": 4.0, "step": 0.1},
        "r_multiple": {"type": "float", "min": 1.0, "max": 5.0, "step": 0.1},
        "use_bounce_filter": {"type": "bool"},
        "use_vol_filter": {"type": "bool"},
        "use_trailing": {"type": "bool"},
    },
    "generate_signals": _generate_signals,
})
