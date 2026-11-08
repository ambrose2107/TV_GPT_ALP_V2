"""AGPro 1-2-3 Reversal strategy; confirmed pivots, no lookahead."""
import numpy as np
from .base import register
from .indicators import atr

DEFAULTS = {
    "left_bars": 2, "right_bars": 2, "atr_len": 14,
    "min_leg_atr": 0.65, "min_p3_hold_atr": 0.05,
    "min_retrace": 0.15, "max_retrace": 0.92,
    "max_structure_age": 160, "break_buffer_atr": 0.05,
    "pocket_width_atr": 0.30, "retest_window": 40,
    "invalid_buffer_atr": 0.15, "stop_buffer_atr": 0.15,
    "reward_risk": 2.0,
}


def _generate_signals(df, params, entry_mode):
    p = {**DEFAULTS, **(params or {})}
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing OHLC columns: " + ", ".join(sorted(missing)))
    d = df.copy()
    left, right = max(1, int(p["left_bars"])), max(1, int(p["right_bars"]))
    av = atr(d, max(2, int(p["atr_len"]))).to_numpy(dtype=float)
    hi, lo, cl = (d[k].to_numpy(dtype=float) for k in ("High", "Low", "Close"))
    sig = np.zeros(len(d), dtype=int)
    stops = np.full(len(d), np.nan)
    targets = np.full(len(d), np.nan)
    pt, pb, pp = [], [], []  # pivot type, bar, price
    direction = 0
    p1bar = p2bar = p3bar = None
    p1 = p2 = p3 = neck = np.nan
    broken = retested = False
    breakbar = None
    pocket_top = pocket_bottom = np.nan

    def add_pivot(kind, bar, price):
        changed = False
        if pt and pt[-1] == kind:
            better = price > pp[-1] if kind == 1 else price < pp[-1]
            if better:
                pb[-1], pp[-1], changed = bar, price, True
        else:
            pt.append(kind); pb.append(bar); pp.append(price); changed = True
        while len(pt) > 12:
            pt.pop(0); pb.pop(0); pp.pop(0)
        return changed

    for i in range(len(d)):
        a = av[i]
        if not np.isfinite(a) or a <= 0:
            continue
        changed = False
        pivot_bar = i - right
        if pivot_bar >= left:
            start = pivot_bar - left
            if hi[pivot_bar] == np.nanmax(hi[start:i + 1]):
                changed = add_pivot(1, pivot_bar, hi[pivot_bar]) or changed
            if lo[pivot_bar] == np.nanmin(lo[start:i + 1]):
                changed = add_pivot(-1, pivot_bar, lo[pivot_bar]) or changed

        if changed and len(pt) >= 3:
            t1, t2, t3 = pt[-3:]
            b1, b2, b3 = pb[-3:]
            q1, q2, q3 = pp[-3:]
            candidate = 1 if (t1, t2, t3) == (-1, 1, -1) else (-1 if (t1, t2, t3) == (1, -1, 1) else 0)
            if candidate and (p3bar is None or b3 != p3bar):
                leg = abs(q2 - q1)
                hold = q3 - q1 if candidate == 1 else q1 - q3
                retrace = (q2 - q3) / max(leg, 1e-12) if candidate == 1 else (q3 - q2) / max(leg, 1e-12)
                valid = (leg >= a * float(p["min_leg_atr"])
                         and hold >= a * float(p["min_p3_hold_atr"])
                         and float(p["min_retrace"]) <= retrace <= float(p["max_retrace"]))
                if valid:
                    direction = candidate
                    p1bar, p2bar, p3bar = b1, b2, b3
                    p1, p2, p3, neck = q1, q2, q3, q2
                    broken = retested = False
                    breakbar = None
                    pocket_top = pocket_bottom = np.nan

        if direction and not broken:
            expired = i - p3bar > int(p["max_structure_age"])
            invalid = lo[i] < p1 - a * float(p["invalid_buffer_atr"]) if direction == 1 else hi[i] > p1 + a * float(p["invalid_buffer_atr"])
            if expired or invalid:
                direction = 0
            else:
                crossed = cl[i] > neck + a * float(p["break_buffer_atr"]) if direction == 1 else cl[i] < neck - a * float(p["break_buffer_atr"])
                if crossed:
                    broken, breakbar = True, i
                    pocket_top = neck + a * float(p["pocket_width_atr"])
                    pocket_bottom = neck - a * float(p["pocket_width_atr"])
                    if entry_mode in ("breakout", "both"):
                        stop = p3 - a * float(p["stop_buffer_atr"]) if direction == 1 else p3 + a * float(p["stop_buffer_atr"])
                        risk = cl[i] - stop if direction == 1 else stop - cl[i]
                        if risk > 0:
                            sig[i] = direction
                            stops[i] = stop
                            targets[i] = cl[i] + direction * risk * float(p["reward_risk"])

        # Pine evaluates retest state after the break block; a breakout bar
        # can therefore qualify as a same-bar pocket touch/hold.
        if direction and broken and not retested:
            in_window = i - breakbar <= int(p["retest_window"])
            invalid = cl[i] < p3 - a * float(p["invalid_buffer_atr"]) if direction == 1 else cl[i] > p3 + a * float(p["invalid_buffer_atr"])
            touched = lo[i] <= pocket_top if direction == 1 else hi[i] >= pocket_bottom
            held = (cl[i] >= pocket_bottom and cl[i] > neck) if direction == 1 else (cl[i] <= pocket_top and cl[i] < neck)
            if invalid:
                direction = 0
            elif in_window and touched and held:
                retested = True
                if entry_mode in ("retest", "both") and sig[i] == 0:
                    stop = p3 - a * float(p["stop_buffer_atr"]) if direction == 1 else p3 + a * float(p["stop_buffer_atr"])
                    risk = cl[i] - stop if direction == 1 else stop - cl[i]
                    if risk > 0:
                        sig[i] = direction
                        stops[i] = stop
                        targets[i] = cl[i] + direction * risk * float(p["reward_risk"])

    d["atr"] = av
    d["signal"] = sig
    d["custom_stop_price"] = stops
    d["custom_target_price"] = targets
    return d


def _register_variant(sid, name, desc, mode):
    register({
        "id": sid, "name": name, "description": desc,
        "default_params": {**DEFAULTS, "entry_mode": mode},
        "param_schema": {
            "left_bars": {"type": "int", "min": 1, "max": 15, "step": 1},
            "right_bars": {"type": "int", "min": 1, "max": 15, "step": 1},
            "atr_len": {"type": "int", "min": 2, "max": 100, "step": 1},
            "min_leg_atr": {"type": "float", "min": 0.1, "max": 10.0, "step": 0.05},
            "min_p3_hold_atr": {"type": "float", "min": 0.0, "max": 5.0, "step": 0.05},
            "min_retrace": {"type": "float", "min": 0.05, "max": 0.9, "step": 0.05},
            "max_retrace": {"type": "float", "min": 0.1, "max": 0.98, "step": 0.01},
            "max_structure_age": {"type": "int", "min": 20, "max": 500, "step": 5},
            "break_buffer_atr": {"type": "float", "min": 0.0, "max": 2.0, "step": 0.01},
            "pocket_width_atr": {"type": "float", "min": 0.05, "max": 3.0, "step": 0.05},
            "retest_window": {"type": "int", "min": 5, "max": 200, "step": 1},
            "invalid_buffer_atr": {"type": "float", "min": 0.0, "max": 3.0, "step": 0.05},
            "stop_buffer_atr": {"type": "float", "min": 0.0, "max": 3.0, "step": 0.05},
            "reward_risk": {"type": "float", "min": 0.25, "max": 10.0, "step": 0.25},
        },
        "generate_signals": lambda df, params, m=mode: _generate_signals(df, params, m),
    })


_register_variant("agpro_123_breakout", "AGPro 1-2-3 Reversal — Breakout", "Confirmed 1-2-3 structure; enter on the close-confirmed neckline break.", "breakout")
_register_variant("agpro_123_retest", "AGPro 1-2-3 Reversal — Retest", "Enter after the confirmed neckline break retests and holds the ATR pocket.", "retest")
_register_variant("agpro_123_both", "AGPro 1-2-3 Reversal — Breakout + Retest", "Accept either the neckline breakout or a subsequent held retest.", "both")
