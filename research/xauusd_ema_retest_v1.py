"""Gold EMA 20/50 Third-Retest Strategy V1.

Rules implemented from the user's specification:
- 15m execution chart with EMA20 and EMA50 only.
- Direction is armed by a confirmed EMA20/EMA50 cross.
- Longs only when EMA20 > EMA50; shorts only when EMA20 < EMA50.
- Wait for the THIRD clean retest after the cross.
- "Clean retest" is made objective:
    long: candle trades down to EMA20, stays above EMA50, and closes back above EMA20.
    short: candle trades up to EMA20, stays below EMA50, and closes back below EMA20.
- 1H EMA20/EMA50 direction must agree, using the last completed 1H bar.
- Entries only during the configured London/New York overlap window.
- Maximum one entry per UTC day.
- Entry is next 15m bar open (no same-bar lookahead).
- Stop is beyond EMA50 by a small ATR buffer and is fixed at entry.
- Target is fixed at 2.5R and is never moved.
- Risk is 1% of current equity per trade by default.

This is research/backtest code, not a performance guarantee.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class EMARetestConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    rr: float = 2.5
    risk_pct: float = 1.0
    retest_number: int = 3
    atr_len: int = 14
    sl_buffer_atr: float = 0.10
    overlap_start_utc_hour: int = 13
    overlap_start_utc_minute: int = 30
    overlap_end_utc_hour: int = 16
    overlap_end_utc_minute: int = 0
    one_trade_per_day: bool = True


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df[["Open", "High", "Low", "Close"]].resample(
        rule, label="right", closed="right"
    ).agg({"Open":"first","High":"max","Low":"min","Close":"last"})
    if "Volume" in df.columns:
        out["Volume"] = df["Volume"].resample(rule, label="right", closed="right").sum()
    return out.dropna(subset=["Open","High","Low","Close"])


def _in_overlap(ts, cfg: EMARetestConfig) -> bool:
    mins = ts.hour * 60 + ts.minute
    start = cfg.overlap_start_utc_hour * 60 + cfg.overlap_start_utc_minute
    end = cfg.overlap_end_utc_hour * 60 + cfg.overlap_end_utc_minute
    return start <= mins < end


def build_signals(m5: pd.DataFrame, cfg: EMARetestConfig | None = None) -> pd.DataFrame:
    cfg = cfg or EMARetestConfig()
    m15 = _resample_ohlc(m5, "15min")
    h1 = _resample_ohlc(m5, "1h")

    m15["ema20"] = m15["Close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    m15["ema50"] = m15["Close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    m15["atr"] = _atr(m15, cfg.atr_len)

    h1["ema20"] = h1["Close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    h1["ema50"] = h1["Close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    h1["bias"] = np.where(h1["ema20"] > h1["ema50"], 1,
                   np.where(h1["ema20"] < h1["ema50"], -1, 0))

    # Shift one completed 1H bar before mapping to 15m to avoid HTF lookahead.
    h1_bias = h1["bias"].shift(1).reindex(m15.index, method="ffill").fillna(0).astype(int)

    fast = m15["ema20"]
    slow = m15["ema50"]
    bull_cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    bear_cross = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    m15["signal"] = 0
    m15["retest_count"] = 0
    m15["h1_bias"] = h1_bias
    m15["sl_ref"] = np.nan
    m15["reason"] = ""

    direction = 0
    count = 0

    warmup = max(cfg.ema_slow, cfg.atr_len) + 2
    for i in range(warmup, len(m15)):
        if bool(bull_cross.iloc[i]):
            direction, count = 1, 0
            continue
        if bool(bear_cross.iloc[i]):
            direction, count = -1, 0
            continue

        # If ordering flips without a new clean cross event, reset state.
        if direction == 1 and fast.iloc[i] <= slow.iloc[i]:
            direction, count = 0, 0
            continue
        if direction == -1 and fast.iloc[i] >= slow.iloc[i]:
            direction, count = 0, 0
            continue
        if direction == 0:
            continue

        row = m15.iloc[i]
        ef, es = float(row.ema20), float(row.ema50)
        clean = False
        if direction == 1:
            clean = (
                float(row.Low) <= ef
                and float(row.Low) > es
                and float(row.Close) > ef
            )
        else:
            clean = (
                float(row.High) >= ef
                and float(row.High) < es
                and float(row.Close) < ef
            )

        if not clean:
            continue

        count += 1
        m15.iat[i, m15.columns.get_loc("retest_count")] = count

        if count != cfg.retest_number:
            continue
        if int(h1_bias.iloc[i]) != direction:
            continue
        if not _in_overlap(m15.index[i], cfg):
            continue

        a = float(row.atr)
        if not np.isfinite(a) or a <= 0:
            continue

        buffer = cfg.sl_buffer_atr * a
        sl_ref = es - buffer if direction == 1 else es + buffer
        m15.iat[i, m15.columns.get_loc("signal")] = direction
        m15.iat[i, m15.columns.get_loc("sl_ref")] = sl_ref
        m15.iat[i, m15.columns.get_loc("reason")] = (
            f"EMA20/50 armed | retest #{count} | 1H aligned | overlap"
        )

    return m15


def backtest(m5: pd.DataFrame, cfg: EMARetestConfig | None = None, initial_equity: float = 10000.0):
    cfg = cfg or EMARetestConfig()
    s = build_signals(m5, cfg)
    equity = float(initial_equity)
    pos = None
    trades = []
    curve = []
    traded_days = set()

    for i in range(len(s)):
        row = s.iloc[i]
        ts = s.index[i]

        # Existing positions are managed using fixed SL/TP only.
        if pos is not None:
            hi, lo = float(row.High), float(row.Low)
            if pos["side"] == 1:
                hit_sl, hit_tp = lo <= pos["sl"], hi >= pos["tp"]
            else:
                hit_sl, hit_tp = hi >= pos["sl"], lo <= pos["tp"]

            r_mult = None
            exit_reason = None
            # Conservative assumption if both levels trade in one bar.
            if hit_sl:
                r_mult, exit_reason = -1.0, "SL"
            elif hit_tp:
                r_mult, exit_reason = cfg.rr, "TP"

            if r_mult is not None:
                pnl = pos["risk_cash"] * r_mult
                equity += pnl
                trades.append({
                    "entry_time": str(pos["entry_time"]),
                    "exit_time": str(ts),
                    "side": "LONG" if pos["side"] == 1 else "SHORT",
                    "entry": pos["entry"],
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "risk_distance": pos["risk_distance"],
                    "risk_cash": pos["risk_cash"],
                    "qty": pos["qty"],
                    "R": r_mult,
                    "pnl": pnl,
                    "equity": equity,
                    "reason": pos["reason"] + " | " + exit_reason,
                })
                pos = None

        # Signal is confirmed at candle close; enter next 15m open.
        if pos is None and i + 1 < len(s) and int(row.signal) != 0:
            day_key = ts.date().isoformat()
            if cfg.one_trade_per_day and day_key in traded_days:
                curve.append(equity)
                continue

            nxt = s.iloc[i + 1]
            entry = float(nxt.Open)
            side = int(row.signal)
            stop = float(row.sl_ref)
            valid = (side == 1 and stop < entry) or (side == -1 and stop > entry)
            if valid:
                risk_distance = abs(entry - stop)
                if np.isfinite(risk_distance) and risk_distance > 0:
                    risk_cash = equity * (cfg.risk_pct / 100.0)
                    qty = risk_cash / risk_distance
                    tp = entry + side * cfg.rr * risk_distance
                    pos = {
                        "side": side,
                        "entry_time": s.index[i + 1],
                        "entry": entry,
                        "sl": stop,
                        "tp": tp,
                        "risk_distance": risk_distance,
                        "risk_cash": risk_cash,
                        "qty": qty,
                        "reason": str(row.reason),
                    }
                    traded_days.add(day_key)

        curve.append(equity)

    # Mark-to-market is deliberately not used: open trades remain open and are excluded.
    t = pd.DataFrame(trades)
    e = pd.Series(curve, index=s.index[:len(curve)], dtype="float64")

    if t.empty:
        metrics = {
            "num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy_R": 0.0, "total_R": 0.0, "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0, "final_equity": round(equity, 2),
        }
    else:
        gross_win = float(t.loc[t.R > 0, "R"].sum())
        gross_loss = float(-t.loc[t.R < 0, "R"].sum())
        dd = float((e / e.cummax() - 1).min() * 100)
        metrics = {
            "num_trades": int(len(t)),
            "win_rate": round(float((t.R > 0).mean() * 100), 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
            "expectancy_R": round(float(t.R.mean()), 3),
            "total_R": round(float(t.R.sum()), 2),
            "max_drawdown_pct": round(dd, 2),
            "total_return_pct": round((equity / initial_equity - 1) * 100, 2),
            "final_equity": round(equity, 2),
        }

    diagnostics = {
        "armed_bull_crosses": int(((s.ema20 > s.ema50) & (s.ema20.shift(1) <= s.ema50.shift(1))).sum()),
        "armed_bear_crosses": int(((s.ema20 < s.ema50) & (s.ema20.shift(1) >= s.ema50.shift(1))).sum()),
        "third_retests": int((s.retest_count == cfg.retest_number).sum()),
        "final_signals": int((s.signal != 0).sum()),
    }
    return {"signals": s, "trades": t, "equity_curve": e, "metrics": metrics, "diagnostics": diagnostics}
