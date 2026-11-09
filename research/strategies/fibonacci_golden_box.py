"""Fibonacci Golden Box continuation strategy.

Causal A-B-C swing detection: pivots become available only after right_bars
confirmation candles have closed. Entries are generated only after a confirmed
C retracement into the 0.618-0.786 pocket and a close-confirmed rejection.
The generic engine executes the signal at the following bar open.
"""
import numpy as np
from .base import register
from .indicators import atr

DEFAULTS = {
    "left_bars": 3, "right_bars": 3, "atr_len": 14,
    "min_leg_atr": 1.0, "golden_shallow": 0.618, "golden_deep": 0.786,
    "zone_buffer_atr": 0.10, "stop_buffer_atr": 0.20,
    "reward_extension": 1.272, "max_structure_age": 180,
    "confirmation": "close_above_previous_high",
    "allow_shorts": True,
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
    d = df.copy()
    hi, lo, cl = (d[k].to_numpy(dtype=float) for k in ("High", "Low", "Close"))
    av = atr(d, max(2, int(p["atr_len"]))).to_numpy(dtype=float)
    n = len(d)
    sig = np.zeros(n, dtype=int)
    stops = np.full(n, np.nan); targets = np.full(n, np.nan)
    box_low = np.full(n, np.nan); box_high = np.full(n, np.nan)
    a_line = np.full(n, np.nan); b_line = np.full(n, np.nan); c_line = np.full(n, np.nan)
    piv_t, piv_i, piv_p = [], [], []
    used_c = set()
    active = None

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

    for i in range(n):
        a = av[i]
        if not np.isfinite(a) or a <= 0:
            continue
        pb = i - right
        changed = False
        if pb >= left:
            start = pb - left
            if hi[pb] >= np.max(hi[start:i+1]):
                changed = add_pivot(1, pb, hi[pb]) or changed
            if lo[pb] <= np.min(lo[start:i+1]):
                changed = add_pivot(-1, pb, lo[pb]) or changed

        if changed and len(piv_t) >= 3:
            t1,t2,t3 = piv_t[-3:]; i1,i2,i3 = piv_i[-3:]; q1,q2,q3 = piv_p[-3:]
            direction = 1 if (t1,t2,t3)==(-1,1,-1) and q3>q1 else (-1 if (t1,t2,t3)==(1,-1,1) and q3<q1 and p["allow_shorts"] else 0)
            leg = abs(q2-q1)
            retrace = ((q2-q3)/leg if direction==1 else (q3-q2)/leg) if leg else np.nan
            valid = (direction and leg >= a*float(p["min_leg_atr"])
                     and shallow <= retrace <= deep
                     and i-i3 <= int(p["max_structure_age"]))
            if valid and i3 not in used_c:
                if direction == 1:
                    z1=q2-leg*deep; z2=q2-leg*shallow
                    target=q3+leg*float(p["reward_extension"])
                    stop=q3-a*float(p["stop_buffer_atr"])
                else:
                    z1=q2+leg*shallow; z2=q2+leg*deep
                    target=q3-leg*float(p["reward_extension"])
                    stop=q3+a*float(p["stop_buffer_atr"])
                active={"dir":direction,"a":q1,"b":q2,"c":q3,"cbar":i3,
                        "low":min(z1,z2),"high":max(z1,z2),"stop":stop,"target":target}
                used_c.add(i3)

        if active:
            age=i-active["cbar"]
            box_low[i]=active["low"]; box_high[i]=active["high"]
            a_line[i]=active["a"]; b_line[i]=active["b"]; c_line[i]=active["c"]
            if age > int(p["max_structure_age"]) or (active["dir"]==1 and lo[i] <= active["stop"]) or (active["dir"]==-1 and hi[i] >= active["stop"]):
                active=None
                continue
            touched = lo[i] <= active["high"] and hi[i] >= active["low"]
            if active["dir"]==1:
                confirm = cl[i] > active["high"] and (p["confirmation"]!="close_above_previous_high" or (i>0 and cl[i]>hi[i-1]))
            else:
                confirm = cl[i] < active["low"] and (p["confirmation"]!="close_above_previous_high" or (i>0 and cl[i]<lo[i-1]))
            if touched and confirm:
                sig[i]=active["dir"]; stops[i]=active["stop"]; targets[i]=active["target"]
                active=None

    d["atr"]=av; d["signal"]=sig
    d["custom_stop_price"]=stops; d["custom_target_price"]=targets
    d["golden_box_low"]=box_low; d["golden_box_high"]=box_high
    d["swing_a"]=a_line; d["swing_b"]=b_line; d["swing_c"]=c_line
    return d

register({
    "id":"fibonacci_golden_box", "name":"Fibonacci Golden Box (A-B-C)",
    "description":"Confirmed A-B-C swing, 0.618-0.786 retracement pocket, rejection confirmation and Fibonacci extension target.",
    "default_params":dict(DEFAULTS), "param_schema":{
        "left_bars":{"type":"int","min":1,"max":15,"step":1},
        "right_bars":{"type":"int","min":1,"max":15,"step":1},
        "atr_len":{"type":"int","min":2,"max":100,"step":1},
        "min_leg_atr":{"type":"float","min":0.1,"max":10,"step":0.1},
        "golden_shallow":{"type":"float","min":0.5,"max":0.75,"step":0.001},
        "golden_deep":{"type":"float","min":0.65,"max":0.9,"step":0.001},
        "zone_buffer_atr":{"type":"float","min":0,"max":2,"step":0.05},
        "stop_buffer_atr":{"type":"float","min":0,"max":3,"step":0.05},
        "reward_extension":{"type":"float","min":1,"max":3,"step":0.001},
        "max_structure_age":{"type":"int","min":20,"max":500,"step":5},
        "allow_shorts":{"type":"bool"},
    }, "generate_signals":generate_signals,
})
