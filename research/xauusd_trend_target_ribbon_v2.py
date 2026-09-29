"""Trend Target Ribbon V2.

Python research implementation of the supplied BOSWaves Trend Target Ribbon
Pine indicator.  The trend engine is ALMA + deviation confirmation + ATR
normalised slope.  A new position starts on a confirmed trend flip, uses
structure-based ATR-clamped risk, and displays 1R..N targets.  Targets are
diagnostic levels; the source indicator does not close at a target. Positions
close on the opposite trend flip or stop.

The backtest uses the signal-bar close as the entry, matching the source
indicator's entryPrice := close. This is intentionally explicit so it is easy
to change later to next-bar execution for a stricter execution model.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TrendTargetRibbonConfig:
    alma_len: int = 34
    alma_offset: float = 0.85
    alma_sigma: float = 6.0
    dev_len: int = 34
    dev_mult: float = 0.65
    slope_len: int = 3
    slope_min: float = 0.08
    atr_len: int = 14
    stop_lookback: int = 12
    min_stop_atr: float = 0.75
    max_stop_atr: float = 3.0
    target_count: int = 4
    risk_pct: float = 0.5
    initial_equity: float = 10000.0
    max_hold_bars: int = 0  # 0 = source-style: hold until flip/stop
    cooldown_bars: int = 0
    # Candidate filter based on recent GLD trade evidence: avoid oversized signal candles.
    max_entry_body_atr: float = 1.0


def _atr(df, n):
    prev = df.Close.shift(1)
    tr = pd.concat([
        df.High - df.Low,
        (df.High - prev).abs(),
        (df.Low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _alma(series, length, offset, sigma):
    # Pine ta.alma: Gaussian weights centred by offset.
    m = offset * (length - 1)
    s = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
    weights = weights / weights.sum()
    return series.rolling(length, min_periods=length).apply(
        lambda x: float(np.dot(x, weights)), raw=True
    )


def build_signals(df, cfg=TrendTargetRibbonConfig()):
    x = df[["Open", "High", "Low", "Close", "Volume"]].copy().sort_index()
    x["alma"] = _alma(x.Close, cfg.alma_len, cfg.alma_offset, cfg.alma_sigma)
    x["dev"] = x.Close.rolling(cfg.dev_len, min_periods=cfg.dev_len).std(ddof=0)
    x["atr"] = _atr(x, cfg.atr_len)
    x["slope_score"] = (
        (x.alma - x.alma.shift(cfg.slope_len)) / x.atr.replace(0, np.nan)
    )
    x["upper_confirm"] = x.alma + x.dev * cfg.dev_mult
    x["lower_confirm"] = x.alma - x.dev * cfg.dev_mult

    bull_setup = (
        (x.slope_score > cfg.slope_min)
        & (x.Close > x.upper_confirm)
    )
    bear_setup = (
        (x.slope_score < -cfg.slope_min)
        & (x.Close < x.lower_confirm)
    )

    trend = np.zeros(len(x), dtype=int)
    bull_flip = np.zeros(len(x), dtype=bool)
    bear_flip = np.zeros(len(x), dtype=bool)

    current = 0
    for i in range(len(x)):
        if bool(bull_setup.iloc[i]) and current != 1:
            current = 1
            bull_flip[i] = True
        elif bool(bear_setup.iloc[i]) and current != -1:
            current = -1
            bear_flip[i] = True
        trend[i] = current

    x["trend"] = trend
    x["bull_flip"] = bull_flip
    x["bear_flip"] = bear_flip
    x["signal"] = np.where(bull_flip, 1, np.where(bear_flip, -1, 0))
    x["sl"] = np.nan
    x["tp1"] = np.nan
    x["tp2"] = np.nan
    x["tp3"] = np.nan
    x["tp4"] = np.nan
    x["risk_dist"] = np.nan
    x["reason"] = ""

    start = max(cfg.alma_len, cfg.dev_len, cfg.atr_len, cfg.stop_lookback) + cfg.slope_len + 2
    for i in range(start, len(x)):
        side = int(x.signal.iloc[i])
        a = float(x.atr.iloc[i])
        entry = float(x.Close.iloc[i])
        if side == 0 or not np.isfinite(a) or a <= 0 or not np.isfinite(entry):
            continue

        body_atr = abs(float(x.Close.iloc[i]) - float(x.Open.iloc[i])) / a
        if np.isfinite(body_atr) and body_atr > cfg.max_entry_body_atr:
            continue

        if side == 1:
            structure_stop = float(x.Low.iloc[i - cfg.stop_lookback:i].min())
            raw_risk = entry - structure_stop
        else:
            structure_stop = float(x.High.iloc[i - cfg.stop_lookback:i].max())
            raw_risk = structure_stop - entry

        if not np.isfinite(raw_risk) or raw_risk <= 0:
            continue

        risk_dist = min(max(raw_risk, a * cfg.min_stop_atr), a * cfg.max_stop_atr)
        stop = entry - side * risk_dist
        x.iat[i, x.columns.get_loc("sl")] = stop
        x.iat[i, x.columns.get_loc("risk_dist")] = risk_dist
        for n in range(1, min(max(cfg.target_count, 2), 4) + 1):
            x.iat[i, x.columns.get_loc(f"tp{n}")] = entry + side * risk_dist * n
        x.iat[i, x.columns.get_loc("reason")] = (
            "ALMA trend flip | slope + deviation confirmation | "
            "structure/ATR stop"
        )

    # Remove invalid flips rather than opening positions with an undefined stop.
    valid = x.sl.notna()
    invalid_flip = (x.signal != 0) & ~valid
    x.loc[invalid_flip, "signal"] = 0
    return x


def _metrics(trades, initial_equity, curve):
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_R": 0.0,
            "total_R": 0.0,
            "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0,
            "final_equity": round(float(initial_equity), 2),
        }
    t = pd.DataFrame(trades)
    r = pd.to_numeric(t["R"], errors="coerce").fillna(0.0)
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    e = pd.Series(curve, dtype=float)
    dd = float((e / e.cummax() - 1.0).min() * 100.0) if len(e) else 0.0
    final = float(e.iloc[-1]) if len(e) else float(initial_equity)
    return {
        "num_trades": int(len(t)),
        "win_rate": round(float((r > 0).mean() * 100), 2),
        "profit_factor": round(gains / losses, 2) if losses else float("inf"),
        "expectancy_R": round(float(r.mean()), 3),
        "total_R": round(float(r.sum()), 2),
        "max_drawdown_pct": round(dd, 2),
        "total_return_pct": round((final / initial_equity - 1.0) * 100, 2),
        "final_equity": round(final, 2),
    }


def backtest(df, cfg=TrendTargetRibbonConfig()):
    s = build_signals(df, cfg)
    equity = float(cfg.initial_equity)
    pos = None
    trades = []
    curve = []
    last_exit_i = -10**9

    for i in range(len(s)):
        ts = s.index[i]
        row = s.iloc[i]

        # The source freezes an existing position when the trend flips.
        if pos is not None and int(row.signal) == -pos["side"]:
            exit_price = float(row.Close)
            r_price = (exit_price - pos["entry"]) * pos["side"]
            r = r_price / pos["risk_dist"] if pos["risk_dist"] > 0 else 0.0
            equity *= 1.0 + (cfg.risk_pct / 100.0) * r
            trades.append({
                "entry_time": str(pos["entry_time"]),
                "exit_time": str(ts),
                "side": "LONG" if pos["side"] == 1 else "SHORT",
                "entry": pos["entry"],
                "exit_price": exit_price,
                "sl": pos["sl"],
                "tp1": pos["tp1"],
                "tp2": pos["tp2"],
                "tp3": pos["tp3"],
                "tp4": pos["tp4"],
                "R": r,
                "equity": equity,
                "bars_held": i - pos["entry_i"],
                "max_target_R": pos["max_target_R"],
                "exit_reason": "Trend Flip",
                "reason": pos["reason"],
            })
            pos = None
            last_exit_i = i

        if pos is not None:
            hi, lo = float(row.High), float(row.Low)
            if pos["side"] == 1:
                stop_hit = lo <= pos["sl"]
                reached = [n for n, tp in enumerate(pos["targets"], 1)
                           if hi >= tp]
            else:
                stop_hit = hi >= pos["sl"]
                reached = [n for n, tp in enumerate(pos["targets"], 1)
                           if lo <= tp]
            if reached:
                pos["max_target_R"] = max(pos["max_target_R"], max(reached))

            timed_out = (
                cfg.max_hold_bars > 0
                and i - pos["entry_i"] >= cfg.max_hold_bars
            )
            if stop_hit or timed_out:
                exit_price = float(pos["sl"] if stop_hit else row.Close)
                r_price = (exit_price - pos["entry"]) * pos["side"]
                r = r_price / pos["risk_dist"] if pos["risk_dist"] > 0 else 0.0
                equity *= 1.0 + (cfg.risk_pct / 100.0) * r
                trades.append({
                    "entry_time": str(pos["entry_time"]),
                    "exit_time": str(ts),
                    "side": "LONG" if pos["side"] == 1 else "SHORT",
                    "entry": pos["entry"],
                    "exit_price": exit_price,
                    "sl": pos["sl"],
                    "tp1": pos["tp1"],
                    "tp2": pos["tp2"],
                    "tp3": pos["tp3"],
                    "tp4": pos["tp4"],
                    "R": r,
                    "equity": equity,
                    "bars_held": i - pos["entry_i"],
                    "max_target_R": pos["max_target_R"],
                    "exit_reason": "Stop Loss" if stop_hit else "Max Hold",
                    "reason": pos["reason"],
                })
                pos = None
                last_exit_i = i

        # Signal-bar close entry, matching the supplied Pine indicator.
        if (
            pos is None
            and int(row.signal) != 0
            and i - last_exit_i >= cfg.cooldown_bars
        ):
            side = int(row.signal)
            entry = float(row.Close)
            stop = float(row.sl)
            risk_dist = float(row.risk_dist)
            if (
                np.isfinite(stop)
                and np.isfinite(risk_dist)
                and risk_dist > 0
                and ((side == 1 and stop < entry) or (side == -1 and stop > entry))
            ):
                targets = [
                    entry + side * risk_dist * n
                    for n in range(1, min(max(cfg.target_count, 2), 4) + 1)
                ]
                pos = {
                    "side": side,
                    "entry_i": i,
                    "entry_time": ts,
                    "entry": entry,
                    "sl": stop,
                    "risk_dist": risk_dist,
                    "targets": targets,
                    "tp1": targets[0],
                    "tp2": targets[1] if len(targets) > 1 else np.nan,
                    "tp3": targets[2] if len(targets) > 2 else np.nan,
                    "tp4": targets[3] if len(targets) > 3 else np.nan,
                    "max_target_R": 0,
                    "reason": str(row.reason),
                }

        curve.append(equity)

    # Mark an open final position to market for the PnL curve, but do not count
    # it as a closed trade metric. This avoids inventing an exit after the data.
    if pos is not None and len(s):
        mark = float(s.Close.iloc[-1])
        r_price = (mark - pos["entry"]) * pos["side"]
        marked_r = r_price / pos["risk_dist"] if pos["risk_dist"] > 0 else 0.0
        curve[-1] = equity * (1.0 + (cfg.risk_pct / 100.0) * marked_r)

    return {
        "signals": s,
        "trades": trades,
        "equity_curve": pd.Series(curve, index=s.index, dtype=float),
        "metrics": _metrics(trades, cfg.initial_equity, curve),
    }
