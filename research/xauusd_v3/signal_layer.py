"""
signal.py
Combines the base EMA/RSI signal with all optional confluence filters, and
- critically - logs a DIAGNOSTIC FUNNEL: how many candidate bars survive
each successive filter. This is what actually answers "why did no trades
fire" instead of guessing: if the funnel shows 400 raw signals -> 6 after
adding 4 filters, you know exactly which filter(s) did the damage.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .indicators import ema, rsi, atr
from .confluence import (
    fibonacci_confluence, liquidity_confluence, supply_demand_confluence,
    fvg_confluence, volume_poi_confluence, intermarket_confluence, confluence_score,
)


@dataclass
class SignalConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    rsi_threshold: float = 5.0
    min_spacing: int = 8
    atr_len: int = 14
    allow_longs: bool = True
    allow_shorts: bool = True

    use_fib: bool = False
    fib_swing_lookback: int = 50
    fib_zone_min: float = 0.5
    fib_zone_max: float = 0.786

    use_liquidity: bool = False
    liq_lookback: int = 30
    liq_sweep_window: int = 6

    use_sd: bool = False
    sd_pivot_left: int = 5
    sd_pivot_right: int = 5
    sd_impulse_atr_mult: float = 2.0
    sd_zone_height_atr_mult: float = 1.0

    use_fvg: bool = False

    use_poi: bool = False
    poi_vol_lookback: int = 50
    poi_vol_mult: float = 2.0
    poi_tolerance_atr: float = 0.5

    use_accuracy_score: bool = False
    min_confluence_score: int = 2

    use_dxy: bool = False
    use_silver: bool = False
    macro_fast_len: int = 20
    macro_slow_len: int = 50


def build_signals(data: dict, cfg: SignalConfig) -> dict:
    """
    data: {'gold': df, 'dxy': df|None, 'silver': df|None}
    Returns {'df': enriched gold df with 'signal' column, 'funnel': dict,
             'confluence_data': dict (for plotting zones)}
    """
    gold = data["gold"].copy()
    gold["ema_f"] = ema(gold["Close"], cfg.ema_fast)
    gold["ema_s"] = ema(gold["Close"], cfg.ema_slow)
    gold["rsi"] = rsi(gold["Close"], cfg.rsi_len)
    gold["atr"] = atr(gold, cfg.atr_len)

    trend_up = gold["ema_f"] > gold["ema_s"]
    trend_down = gold["ema_f"] < gold["ema_s"]
    rsi_long = gold["rsi"] > (50 + cfg.rsi_threshold)
    rsi_short = gold["rsi"] < (50 - cfg.rsi_threshold)

    raw_dir = pd.Series(0, index=gold.index)
    raw_dir[trend_up & rsi_long] = 1
    raw_dir[trend_down & rsi_short] = -1

    funnel = {"1_raw_ema_rsi_signal": int((raw_dir != 0).sum())}

    # -- compute every confluence (always computed, so display/diagnostics
    #    work even when a filter is toggled off) --
    fib = fibonacci_confluence(gold, gold["atr"], cfg.fib_swing_lookback, cfg.fib_zone_min, cfg.fib_zone_max)
    liq = liquidity_confluence(gold, cfg.liq_lookback, cfg.liq_sweep_window)
    sd = supply_demand_confluence(gold, gold["atr"], cfg.sd_pivot_left, cfg.sd_pivot_right,
                                   cfg.sd_impulse_atr_mult, cfg.sd_zone_height_atr_mult)
    fvg = fvg_confluence(gold)
    poi = volume_poi_confluence(gold, gold["atr"], cfg.poi_vol_lookback, cfg.poi_vol_mult, cfg.poi_tolerance_atr)
    dxy_c = intermarket_confluence(gold, data.get("dxy"), cfg.macro_fast_len, cfg.macro_slow_len, inverse=True)
    silver_c = intermarket_confluence(gold, data.get("silver"), cfg.macro_fast_len, cfg.macro_slow_len, inverse=False)

    score = confluence_score(
        {"sd": sd["long"], "fvg": fvg["long"], "poi": poi["confluence"], "liq": liq["long"], "fib": fib["long"]},
        {"sd": sd["short"], "fvg": fvg["short"], "poi": poi["confluence"], "liq": liq["short"], "fib": fib["short"]},
    )

    def _apply(name, use_flag, long_series, short_series, current_dir, funnel_dict):
        if not use_flag:
            return current_dir
        ok_long = long_series.reindex(current_dir.index).fillna(False)
        ok_short = short_series.reindex(current_dir.index).fillna(False)
        new_dir = current_dir.copy()
        new_dir[(current_dir == 1) & (~ok_long)] = 0
        new_dir[(current_dir == -1) & (~ok_short)] = 0
        funnel_dict[f"after_{name}"] = int((new_dir != 0).sum())
        return new_dir

    d = raw_dir.copy()
    d = _apply("fib", cfg.use_fib, fib["long"], fib["short"], d, funnel)
    d = _apply("liquidity", cfg.use_liquidity, liq["long"], liq["short"], d, funnel)
    d = _apply("supply_demand", cfg.use_sd, sd["long"], sd["short"], d, funnel)
    d = _apply("fvg", cfg.use_fvg, fvg["long"], fvg["short"], d, funnel)
    d = _apply("volume_poi", cfg.use_poi, poi["confluence"], poi["confluence"], d, funnel)

    if cfg.use_accuracy_score:
        ok_long = (score["long_score"] >= cfg.min_confluence_score).reindex(d.index).fillna(False)
        ok_short = (score["short_score"] >= cfg.min_confluence_score).reindex(d.index).fillna(False)
        d[(d == 1) & (~ok_long)] = 0
        d[(d == -1) & (~ok_short)] = 0
        funnel["after_confluence_score"] = int((d != 0).sum())

    if cfg.use_dxy and dxy_c["available"]:
        d = _apply("dxy", True, dxy_c["long"], dxy_c["short"], d, funnel)
    elif cfg.use_dxy and not dxy_c["available"]:
        print("[signal] DXY filter requested but DXY data unavailable - filter skipped.")

    if cfg.use_silver and silver_c["available"]:
        d = _apply("silver", True, silver_c["long"], silver_c["short"], d, funnel)
    elif cfg.use_silver and not silver_c["available"]:
        print("[signal] Silver filter requested but Silver data unavailable - filter skipped.")

    if not cfg.allow_longs:
        d[d == 1] = 0
    if not cfg.allow_shorts:
        d[d == -1] = 0
    funnel["after_long_short_toggle"] = int((d != 0).sum())

    # transition-only + cooldown
    prev_dir = d.shift(1).fillna(0)
    transition = (d != 0) & (d != prev_dir)
    funnel["after_transition_only"] = int(transition.sum())

    signal = np.zeros(len(gold), dtype=int)
    last_fire = -10**9
    trans_idx = np.where(transition.values)[0]
    d_vals = d.values
    for i in trans_idx:
        if i - last_fire >= cfg.min_spacing:
            signal[i] = d_vals[i]
            last_fire = i
    gold["signal"] = signal
    funnel["after_cooldown_FINAL"] = int((signal != 0).sum())

    confluence_data = {
        "fib": fib, "liq": liq, "sd": sd, "fvg": fvg, "poi": poi,
        "dxy": dxy_c, "silver": silver_c, "score": score,
    }
    return {"df": gold, "funnel": funnel, "confluence_data": confluence_data}
