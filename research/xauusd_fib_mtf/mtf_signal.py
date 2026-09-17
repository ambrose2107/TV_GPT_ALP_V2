"""
mtf_signal.py
Orchestrates the full spec:
  15m: structure trend (bias) -> clean impulse detection -> Fibonacci zone
       (0.618-0.786) -> confluence gate (liquidity + S/R + order block,
       require >= min_confluences)
  5m:  liquidity sweep + BOS/structure-shift confirmation, in the same
       direction as the 15m bias, while price is in the 15m zone
  ->   entry on the 5m confirmation bar's close, with explicit SL and
       TP1-4 levels computed from the structural rules in the spec.

Logs a diagnostic funnel at every gate, same philosophy as the earlier
work: if you get zero trades, the funnel tells you exactly which gate did it.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .structure import atr, market_structure
from .fibonacci import detect_impulses, fib_levels_for_leg, fib_zone_bounds
from .confluence import liquidity_confluence, sr_confluence, find_order_block, sweep_and_bos


@dataclass
class MTFConfig:
    # structure detection
    pivot_left_15m: int = 3
    pivot_right_15m: int = 3
    pivot_left_5m: int = 3
    pivot_right_5m: int = 3
    atr_len: int = 14

    # impulse quality filter
    min_impulse_efficiency: float = 0.35
    min_impulse_atr_mult: float = 3.0

    # fib zone
    zone_min: float = 0.618
    zone_max: float = 0.786
    zone_tolerance_atr: float = 0.3   # how close a confluence level must be to the zone

    # confluence gate
    min_confluences: int = 2          # out of {liquidity, sr, order_block}
    sr_lookback_pivots: int = 10

    # 5m confirmation
    sweep_lookback_5m: int = 20
    sweep_window_5m: int = 6
    sl_buffer_atr_mult: float = 0.15
    min_spacing_5m: int = 6

    allow_longs: bool = True
    allow_shorts: bool = True


def _recent_swing_levels(structure15: pd.DataFrame, up_to_idx: int, n_pivots: int, kind: str) -> list:
    col = "swing_high" if kind == "high" else "swing_low"
    series = structure15[col].iloc[:up_to_idx + 1].dropna()
    return list(series.tail(n_pivots).values)


def build_mtf_signals(data: dict, cfg: MTFConfig) -> dict:
    """
    data: {'m5': df, 'm15': df}
    Returns {'df5': enriched 5m df with signal/sl/tp1-4 columns, 'df15':
    enriched 15m df, 'funnel': dict}
    """
    m15 = data["m15"].copy()
    m5 = data["m5"].copy()

    m15["atr"] = atr(m15, cfg.atr_len)
    m5["atr"] = atr(m5, cfg.atr_len)

    struct15 = market_structure(m15, cfg.pivot_left_15m, cfg.pivot_right_15m)
    struct15["atr"] = m15["atr"]
    struct5 = market_structure(m5, cfg.pivot_left_5m, cfg.pivot_right_5m)
    struct5["atr"] = m5["atr"]

    impulses = detect_impulses(m15, struct15["swing_high"], struct15["swing_low"], m15["atr"],
                                cfg.min_impulse_efficiency, cfg.min_impulse_atr_mult)
    struct15["impulse_dir"] = impulses["impulse_dir"]
    struct15["impulse_low"] = impulses["impulse_low"]
    struct15["impulse_high"] = impulses["impulse_high"]

    funnel = {}
    n15 = len(struct15)

    bias_bull = (struct15["structure_trend"] == "bullish").values
    bias_bear = (struct15["structure_trend"] == "bearish").values
    imp_bull = (struct15["impulse_dir"] == "bullish").values
    imp_bear = (struct15["impulse_dir"] == "bearish").values

    funnel["1_15m_bullish_bias"] = int(bias_bull.sum())
    funnel["1_15m_bearish_bias"] = int(bias_bear.sum())
    funnel["2_bullish_bias_AND_impulse"] = int((bias_bull & imp_bull).sum())
    funnel["2_bearish_bias_AND_impulse"] = int((bias_bear & imp_bear).sum())

    # ---- 15m zone + confluence gate: computed ONCE per unique impulse
    #      (not re-checked every bar against "is close exactly in the zone
    #      right now") and kept LIVE for that impulse's whole lifetime, so
    #      the 5m timeframe has a real window to react whenever price
    #      actually revisits the zone - not just the fleeting instant the
    #      15m candle's own close happens to sit inside it. ----
    close15 = m15["Close"].values
    imp_low_v = struct15["impulse_low"].values
    imp_high_v = struct15["impulse_high"].values
    atr15_v = struct15["atr"].values
    imp_low_i_arr = impulses["impulse_low_i"]
    imp_high_i_arr = impulses["impulse_high_i"]
    prev_swing_low_v = struct15["prev_swing_low"].values
    prev_swing_high_v = struct15["prev_swing_high"].values

    zone_gate_bull = np.zeros(n15, dtype=bool)
    zone_gate_bear = np.zeros(n15, dtype=bool)
    tp2_bull = np.full(n15, np.nan)
    tp2_bear = np.full(n15, np.nan)
    tp3_bull = np.full(n15, np.nan)
    tp3_bear = np.full(n15, np.nan)
    tp4_bull = np.full(n15, np.nan)
    tp4_bear = np.full(n15, np.nan)
    zone_lo_bull = np.full(n15, np.nan)
    zone_hi_bull = np.full(n15, np.nan)
    zone_lo_bear = np.full(n15, np.nan)
    zone_hi_bear = np.full(n15, np.nan)

    n_unique_impulses = {"bull": 0, "bear": 0}
    confluence_count_pass = {"bull": 0, "bear": 0}

    last_imp_high_i_seen_bull = -2
    last_imp_high_i_seen_bear = -2
    cur_gate_bull, cur_lo_bull, cur_hi_bull = False, np.nan, np.nan
    cur_gate_bear, cur_lo_bear, cur_hi_bear = False, np.nan, np.nan
    cur_tp2_bull = cur_tp3_bull = cur_tp4_bull = np.nan
    cur_tp2_bear = cur_tp3_bear = cur_tp4_bear = np.nan

    for i in range(n15):
        tol = cfg.zone_tolerance_atr * atr15_v[i] if not np.isnan(atr15_v[i]) else 0.0

        # -- bullish impulse: evaluate confluence ONCE when a NEW impulse
        #    (identified by its high-pivot bar index) is first seen --
        if bias_bull[i] and imp_bull[i] and not np.isnan(imp_low_v[i]) and imp_high_i_arr[i] != last_imp_high_i_seen_bull:
            last_imp_high_i_seen_bull = imp_high_i_arr[i]
            n_unique_impulses["bull"] += 1
            lo, hi = fib_zone_bounds(imp_low_v[i], imp_high_v[i], "bullish", cfg.zone_min, cfg.zone_max)
            liq = liquidity_confluence(lo, hi, prev_swing_low_v[i], tol)
            others = _recent_swing_levels(struct15, i, cfg.sr_lookback_pivots, "low") + \
                _recent_swing_levels(struct15, i, cfg.sr_lookback_pivots, "high")
            sr = sr_confluence(lo, hi, others, tol)
            ob_lo, ob_hi = find_order_block(m15, int(imp_low_i_arr[i]), "bullish")
            ob = sr_confluence(lo, hi, [ob_lo, ob_hi], tol) if ob_lo is not None else False
            n_conf = int(liq) + int(sr) + int(ob)
            if n_conf >= cfg.min_confluences:
                confluence_count_pass["bull"] += 1
                cur_gate_bull, cur_lo_bull, cur_hi_bull = True, lo, hi
                fibs = fib_levels_for_leg(imp_low_v[i], imp_high_v[i], "bullish")
                cur_tp2_bull, cur_tp3_bull, cur_tp4_bull = imp_high_v[i], fibs[1.272], fibs[1.618]
            else:
                cur_gate_bull = False

        # -- bearish impulse: mirror --
        if bias_bear[i] and imp_bear[i] and not np.isnan(imp_low_v[i]) and imp_high_i_arr[i] != last_imp_high_i_seen_bear:
            last_imp_high_i_seen_bear = imp_high_i_arr[i]
            n_unique_impulses["bear"] += 1
            lo, hi = fib_zone_bounds(imp_low_v[i], imp_high_v[i], "bearish", cfg.zone_min, cfg.zone_max)
            liq = liquidity_confluence(lo, hi, prev_swing_high_v[i], tol)
            others = _recent_swing_levels(struct15, i, cfg.sr_lookback_pivots, "low") + \
                _recent_swing_levels(struct15, i, cfg.sr_lookback_pivots, "high")
            sr = sr_confluence(lo, hi, others, tol)
            ob_lo, ob_hi = find_order_block(m15, int(imp_high_i_arr[i]), "bearish")
            ob = sr_confluence(lo, hi, [ob_lo, ob_hi], tol) if ob_lo is not None else False
            n_conf = int(liq) + int(sr) + int(ob)
            if n_conf >= cfg.min_confluences:
                confluence_count_pass["bear"] += 1
                cur_gate_bear, cur_lo_bear, cur_hi_bear = True, lo, hi
                fibs = fib_levels_for_leg(imp_low_v[i], imp_high_v[i], "bearish")
                cur_tp2_bear, cur_tp3_bear, cur_tp4_bear = imp_low_v[i], fibs[1.272], fibs[1.618]
            else:
                cur_gate_bear = False

        # invalidate a live zone if price closes back beyond the impulse's
        # own origin (the setup's premise has failed)
        if cur_gate_bull and not np.isnan(imp_low_v[i]) and close15[i] < imp_low_v[i]:
            cur_gate_bull = False
        if cur_gate_bear and not np.isnan(imp_high_v[i]) and close15[i] > imp_high_v[i]:
            cur_gate_bear = False

        zone_gate_bull[i], zone_lo_bull[i], zone_hi_bull[i] = cur_gate_bull, cur_lo_bull, cur_hi_bull
        zone_gate_bear[i], zone_lo_bear[i], zone_hi_bear[i] = cur_gate_bear, cur_lo_bear, cur_hi_bear
        tp2_bull[i], tp3_bull[i], tp4_bull[i] = cur_tp2_bull, cur_tp3_bull, cur_tp4_bull
        tp2_bear[i], tp3_bear[i], tp4_bear[i] = cur_tp2_bear, cur_tp3_bear, cur_tp4_bear

    funnel["3_unique_impulses_bull"] = n_unique_impulses["bull"]
    funnel["3_unique_impulses_bear"] = n_unique_impulses["bear"]
    funnel["4_confluence_gate_pass_bull"] = confluence_count_pass["bull"]
    funnel["4_confluence_gate_pass_bear"] = confluence_count_pass["bear"]

    struct15["zone_gate_bull"] = zone_gate_bull
    struct15["zone_gate_bear"] = zone_gate_bear
    struct15["zone_lo_bull"], struct15["zone_hi_bull"] = zone_lo_bull, zone_hi_bull
    struct15["zone_lo_bear"], struct15["zone_hi_bear"] = zone_lo_bear, zone_hi_bear
    struct15["tp2_bull"], struct15["tp3_bull"], struct15["tp4_bull"] = tp2_bull, tp3_bull, tp4_bull
    struct15["tp2_bear"], struct15["tp3_bear"], struct15["tp4_bear"] = tp2_bear, tp3_bear, tp4_bear

    # ---- forward-fill 15m setup state onto the 5m index ----
    cols_to_ffill = ["zone_gate_bull", "zone_gate_bear", "zone_lo_bull", "zone_hi_bull",
                      "zone_lo_bear", "zone_hi_bear", "tp2_bull", "tp3_bull", "tp4_bull",
                      "tp2_bear", "tp3_bear", "tp4_bear"]
    setup_5m = struct15[cols_to_ffill].reindex(m5.index, method="ffill")

    # ---- 5m confirmation: sweep + BOS ----
    sweep = sweep_and_bos(m5, struct5, cfg.sweep_lookback_5m, cfg.sweep_window_5m)

    close5 = m5["Close"].values
    atr5_v = m5["atr"].values
    low5 = m5["Low"].values
    high5 = m5["High"].values
    zone_gate_bull_5 = setup_5m["zone_gate_bull"].fillna(False).values.astype(bool)
    zone_gate_bear_5 = setup_5m["zone_gate_bear"].fillna(False).values.astype(bool)
    zone_lo_bull_5 = setup_5m["zone_lo_bull"].values
    zone_hi_bull_5 = setup_5m["zone_hi_bull"].values
    zone_lo_bear_5 = setup_5m["zone_lo_bear"].values
    zone_hi_bear_5 = setup_5m["zone_hi_bear"].values

    in_zone_5_bull = zone_gate_bull_5 & (close5 >= zone_lo_bull_5) & (close5 <= zone_hi_bull_5)
    in_zone_5_bear = zone_gate_bear_5 & (close5 >= zone_lo_bear_5) & (close5 <= zone_hi_bear_5)

    funnel["5_5m_in_zone_bull"] = int(in_zone_5_bull.sum())
    funnel["5_5m_in_zone_bear"] = int(in_zone_5_bear.sum())

    # IMPORTANT: the BOS confirmation naturally happens AS PRICE LEAVES the
    # zone during the resumption move, not while still sitting inside it -
    # so we check whether the SWEEP (the bar just before/at the reversal)
    # happened while price was in/near the zone, not whether the later BOS
    # bar itself is still inside the zone bounds.
    last_bull_sweep_idx_arr = sweep["last_bull_sweep_idx"]
    last_bear_sweep_idx_arr = sweep["last_bear_sweep_idx"]

    def _sweep_was_in_zone(sweep_idx_arr, zone_active, zone_lo, zone_hi, price_ref):
        n = len(price_ref)
        out = np.zeros(n, dtype=bool)
        for i in range(n):
            si = sweep_idx_arr[i]
            if 0 <= si < n and zone_active[si]:
                lo, hi = zone_lo[si], zone_hi[si]
                if not np.isnan(lo) and lo <= price_ref[si] <= hi:
                    out[i] = True
        return out

    sweep_in_zone_bull = _sweep_was_in_zone(last_bull_sweep_idx_arr, zone_gate_bull_5,
                                             zone_lo_bull_5, zone_hi_bull_5, low5)
    sweep_in_zone_bear = _sweep_was_in_zone(last_bear_sweep_idx_arr, zone_gate_bear_5,
                                             zone_lo_bear_5, zone_hi_bear_5, high5)

    bull_confirm = sweep["bull_confirm"].values & sweep_in_zone_bull
    bear_confirm = sweep["bear_confirm"].values & sweep_in_zone_bear

    if not cfg.allow_longs:
        bull_confirm[:] = False
    if not cfg.allow_shorts:
        bear_confirm[:] = False

    funnel["6_5m_sweep_plus_BOS_confirm_bull"] = int(bull_confirm.sum())
    funnel["6_5m_sweep_plus_BOS_confirm_bear"] = int(bear_confirm.sum())

    n5 = len(m5)
    signal = np.zeros(n5, dtype=int)
    sl_level = np.full(n5, np.nan)
    tp1 = np.full(n5, np.nan)
    tp2 = np.full(n5, np.nan)
    tp3 = np.full(n5, np.nan)
    tp4 = np.full(n5, np.nan)

    last_bull_sweep_idx = sweep["last_bull_sweep_idx"]
    last_bear_sweep_idx = sweep["last_bear_sweep_idx"]
    last_swing_high_5 = struct5["last_swing_high"].values
    last_swing_low_5 = struct5["last_swing_low"].values

    last_fire = -10**9
    for i in range(n5):
        if i - last_fire < cfg.min_spacing_5m:
            continue
        if bull_confirm[i]:
            sweep_i = last_bull_sweep_idx[i]
            sweep_low = low5[sweep_i] if 0 <= sweep_i < n5 else low5[i]
            sl = sweep_low - cfg.sl_buffer_atr_mult * (atr5_v[i] if not np.isnan(atr5_v[i]) else 0)
            t1 = last_swing_high_5[i]
            t2 = setup_5m["tp2_bull"].values[i]
            t3 = setup_5m["tp3_bull"].values[i]
            t4 = setup_5m["tp4_bull"].values[i]
            targets = sorted([v for v in [t1, t2, t3, t4] if not np.isnan(v) and v > close5[i]])
            if len(targets) >= 2 and not np.isnan(sl) and sl < close5[i]:
                signal[i] = 1
                sl_level[i] = sl
                padded = (targets + [targets[-1]] * 4)[:4]
                tp1[i], tp2[i], tp3[i], tp4[i] = padded
                last_fire = i
        elif bear_confirm[i]:
            sweep_i = last_bear_sweep_idx[i]
            sweep_high = high5[sweep_i] if 0 <= sweep_i < n5 else high5[i]
            sl = sweep_high + cfg.sl_buffer_atr_mult * (atr5_v[i] if not np.isnan(atr5_v[i]) else 0)
            t1 = last_swing_low_5[i]
            t2 = setup_5m["tp2_bear"].values[i]
            t3 = setup_5m["tp3_bear"].values[i]
            t4 = setup_5m["tp4_bear"].values[i]
            targets = sorted([v for v in [t1, t2, t3, t4] if not np.isnan(v) and v < close5[i]], reverse=True)
            if len(targets) >= 2 and not np.isnan(sl) and sl > close5[i]:
                signal[i] = -1
                sl_level[i] = sl
                padded = (targets + [targets[-1]] * 4)[:4]
                tp1[i], tp2[i], tp3[i], tp4[i] = padded
                last_fire = i

    funnel["7_min_2_valid_targets_and_sl"] = int((signal != 0).sum())

    m5["signal"] = signal
    m5["sl_level"] = sl_level
    m5["tp1"], m5["tp2"], m5["tp3"], m5["tp4"] = tp1, tp2, tp3, tp4

    return {"df5": m5, "df15": struct15, "funnel": funnel}
