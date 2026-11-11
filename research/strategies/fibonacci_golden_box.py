"""Automatic, causal Fibonacci Golden Box strategy for XAUUSD and other OHLC markets.

Four coordinated drawings are emitted for each confirmed A-B-C structure:
1) A-B Fibonacci retracement, 2) A-B-C trend-based extension,
3) the 0.618-0.786 entry pocket, and 4) projected profit-taking zone/levels.

Pivots are only known after right_bars confirmation. Signals are generated later,
after a candle trades into the pocket and closes back out in the trend direction.
"""
import numpy as np
from .base import register
from .indicators import atr

DEFAULTS = {
    "left_bars": 3, "right_bars": 3, "atr_len": 14,
    "min_leg_atr": 1.0, "golden_shallow": 0.618, "golden_deep": 0.786,
    "zone_buffer_atr": 0.10, "stop_buffer_atr": 0.20,
    "reward_extension": 1.272, "max_structure_age": 180,
    "allow_shorts": True, "confirmation": "directional_rejection",
    "target_zone_start": 1.272, "target_zone_end": 1.414,
}


def generate_signals(df, params=None):
    p = {**DEFAULTS, **(params or {})}
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing OHLC columns: " + ", ".join(sorted(missing)))
    left, right = max(1, int(p["left_bars"])), max(1, int(p["right_bars"]))
    shallow, deep = float(p["golden_shallow"]), float(p["golden_deep"])
    if not (0 < shallow < deep < 1):
        raise ValueError("Golden levels must satisfy 0 < shallow < deep < 1.")
    ext = float(p["reward_extension"])
    tz0, tz1 = float(p["target_zone_start"]), float(p["target_zone_end"])
    if ext < 1 or not (1 <= tz0 < tz1 <= 4):
        raise ValueError("Extension target must be >=1 and target zone must be ordered between 1 and 4.")

    d = df.copy()
    op, hi, lo, cl = (d[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    av = atr(d, max(2, int(p["atr_len"]))).to_numpy(dtype=float)
    n = len(d)
    sig = np.zeros(n, dtype=int)
    names = ("custom_stop_price", "custom_target_price", "golden_box_low", "golden_box_high",
             "swing_a", "swing_b", "swing_c", "fib_a_bar", "fib_b_bar", "fib_c_bar",
             "fib_05", "fib_0618", "fib_0786", "fib_zone_shallow", "fib_zone_deep", "fib_1272", "fib_1414", "fib_1618",
             "fib_2000", "entry_box_low", "entry_box_high", "target_box_low", "target_box_high",
             "target_1272", "target_1414", "target_1618", "target_2000", "setup_direction")
    out = {k: np.full(n, np.nan) for k in names}
    piv_t, piv_i, piv_p = [], [], []
    active = None
    used_c = set()

    def add_pivot(kind, bar, price):
        if piv_t and piv_t[-1] == kind:
            better = price > piv_p[-1] if kind == 1 else price < piv_p[-1]
            if better:
                piv_i[-1], piv_p[-1] = bar, price
                return True
            return False
        piv_t.append(kind); piv_i.append(bar); piv_p.append(price)
        if len(piv_t) > 8:
            piv_t.pop(0); piv_i.pop(0); piv_p.pop(0)
        return True

    def projected(c, leg, direction, ratio):
        return c + direction * leg * ratio

    for i in range(n):
        a_atr = av[i]
        if not np.isfinite(a_atr) or a_atr <= 0:
            continue
        # A pivot at pb is only used after right-side candles have closed.
        pb = i - right
        changed = False
        if pb >= left:
            start = pb - left
            if hi[pb] >= np.max(hi[start:i+1]):
                changed = add_pivot(1, pb, hi[pb]) or changed
            if lo[pb] <= np.min(lo[start:i+1]):
                changed = add_pivot(-1, pb, lo[pb]) or changed

        if changed and len(piv_t) >= 3:
            t1, t2, t3 = piv_t[-3:]
            i1, i2, i3 = piv_i[-3:]
            q1, q2, q3 = piv_p[-3:]
            direction = 1 if (t1, t2, t3) == (-1, 1, -1) and q3 > q1 else (
                -1 if (t1, t2, t3) == (1, -1, 1) and q3 < q1 and p["allow_shorts"] else 0)
            leg = abs(q2 - q1)
            retrace = ((q2-q3)/leg if direction == 1 else (q3-q2)/leg) if leg else np.nan
            valid = (direction != 0 and leg >= a_atr * float(p["min_leg_atr"])
                     and shallow <= retrace <= deep
                     and i-i3 <= int(p["max_structure_age"]))
            if valid and i3 not in used_c:
                # Retracement levels are measured from B back toward A.
                fib = {r: q2 - direction * leg * r for r in (0.5, 0.618, 0.786)}
                z1 = q2 - direction * leg * deep
                z2 = q2 - direction * leg * shallow
                # Entry pocket boundaries are ordered numerically.
                entry_low, entry_high = min(z1, z2), max(z1, z2)
                # Stop beyond C, buffered by ATR known at confirmation.
                stop = q3 - direction * a_atr * float(p["stop_buffer_atr"])
                targets = {r: projected(q3, leg, direction, r) for r in (1.272, 1.414, 1.618, 2.0)}
                target = projected(q3, leg, direction, ext)
                target_zone = [projected(q3, leg, direction, tz0), projected(q3, leg, direction, tz1)]
                active = {
                    "dir": direction, "a": q1, "b": q2, "c": q3,
                    "ai": i1, "bi": i2, "ci": i3, "detected": i,
                    "low": entry_low, "high": entry_high, "stop": stop,
                    "target": target, "fib": fib, "targets": targets,
                    "target_low": min(target_zone), "target_high": max(target_zone), "entered": False,
                }
                used_c.add(i3)

        if not active:
            continue
        age = i - active["detected"]
        if age > int(p["max_structure_age"]):
            active = None
            continue
        direction = active["dir"]
        # A structure is invalidated only by a close through its C/stop boundary.
        invalid = lo[i] <= active["stop"] if direction == 1 else hi[i] >= active["stop"]
        if invalid:
            active = None
            continue

        vals = {
            "swing_a": active["a"], "swing_b": active["b"], "swing_c": active["c"],
            "fib_a_bar": active["ai"], "fib_b_bar": active["bi"], "fib_c_bar": active["ci"],
            "fib_05": active["fib"][0.5], "fib_0618": active["fib"][0.618],
            "fib_0786": active["fib"][0.786], "fib_zone_shallow": shallow, "fib_zone_deep": deep, "fib_1272": active["targets"][1.272],
            "fib_1414": active["targets"][1.414], "fib_1618": active["targets"][1.618],
            "fib_2000": active["targets"][2.0], "golden_box_low": active["low"],
            "golden_box_high": active["high"], "entry_box_low": active["low"],
            "entry_box_high": active["high"], "target_box_low": active["target_low"],
            "target_box_high": active["target_high"], "target_1272": active["targets"][1.272],
            "target_1414": active["targets"][1.414], "target_1618": active["targets"][1.618],
            "target_2000": active["targets"][2.0], "setup_direction": direction,
        }
        for key, value in vals.items():
            out[key][i] = value

        touched = lo[i] <= active["high"] and hi[i] >= active["low"]
        if direction == 1:
            rejection = cl[i] > active["high"] and cl[i] > op[i]
            if p["confirmation"] == "close_above_previous_high":
                rejection = rejection and i > 0 and cl[i] > hi[i-1]
        else:
            rejection = cl[i] < active["low"] and cl[i] < op[i]
            if p["confirmation"] == "close_above_previous_high":
                rejection = rejection and i > 0 and cl[i] < lo[i-1]
        if touched and rejection and not active["entered"]:
            sig[i] = direction
            out["custom_stop_price"][i] = active["stop"]
            out["custom_target_price"][i] = active["target"]
            active["entered"] = True

    d["atr"] = av
    d["signal"] = sig
    for key, values in out.items():
        d[key] = values
    return d


register({
    "id": "fibonacci_golden_box", "name": "Fibonacci Golden Box (4-stage A-B-C)",
    "description": "Automatic two-way A-B retracement, A-B-C extension, golden-pocket rejection entry and extension profit zones.",
    "default_params": dict(DEFAULTS),
    "param_schema": {
        "left_bars": {"type":"int","min":1,"max":15,"step":1},
        "right_bars": {"type":"int","min":1,"max":15,"step":1},
        "atr_len": {"type":"int","min":2,"max":100,"step":1},
        "min_leg_atr": {"type":"float","min":0.1,"max":10,"step":0.1},
        "golden_shallow": {"type":"float","min":0.5,"max":0.75,"step":0.001},
        "golden_deep": {"type":"float","min":0.5,"max":0.9,"step":0.001},
        "zone_buffer_atr": {"type":"float","min":0,"max":2,"step":0.05},
        "stop_buffer_atr": {"type":"float","min":0,"max":3,"step":0.05},
        "reward_extension": {"type":"float","min":1,"max":3,"step":0.001},
        "max_structure_age": {"type":"int","min":20,"max":500,"step":5},
        "allow_shorts": {"type":"bool"},
        "confirmation": {"type":"str","options":["directional_rejection","close_above_previous_high"]},
        "target_zone_start": {"type":"float","min":1,"max":3.9,"step":0.001},
        "target_zone_end": {"type":"float","min":1.001,"max":4,"step":0.001},
    },
    "generate_signals": generate_signals,
})
