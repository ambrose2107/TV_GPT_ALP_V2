"""Triple-RSI mean-reversion strategy from the supplied reference rules.

Rules:
1. RSI(5) is below 30.
2. RSI(5) has fallen for the third consecutive day.
3. RSI(5) was below 60 three sessions ago.
4. Close is above the 200-day simple moving average.
5. Enter long at that session close.
6. Exit at the close when RSI(5) crosses above 50.

This is a daily strategy. The implementation is intentionally literal and uses
only information available at the current completed daily close.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TripleRSIConfig:
    rsi_len: int = 5
    rsi_entry_max: float = 30.0
    rsi_three_days_ago_max: float = 60.0
    rsi_exit_level: float = 50.0
    ma_len: int = 200
    risk_pct: float = 0.5
    initial_equity: float = 10000.0


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss.ne(0), 100.0)


def _metrics(trades, equity_curve, initial_equity, risk_pct):
    if not trades:
        return {
            "num_trades": 0, "win_rate": 0.0, "profit_factor": np.nan,
            "expectancy_R": 0.0, "total_R": 0.0, "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0, "final_equity": float(initial_equity),
        }
    rets = np.asarray([float(t["return_pct"]) / 100.0 for t in trades])
    wins, losses = rets[rets > 0], rets[rets < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else (
        np.inf if len(wins) else np.nan
    )
    rs = rets / max(float(risk_pct) / 100.0, 1e-9)
    eq = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0) * 100.0
    return {
        "num_trades": int(len(trades)),
        "win_rate": float((rets > 0).mean() * 100.0),
        "profit_factor": pf,
        "expectancy_R": float(rs.mean()),
        "total_R": float(rs.sum()),
        "total_return_pct": float((eq[-1] / initial_equity - 1.0) * 100.0),
        "max_drawdown_pct": float(dd.min()) if len(dd) else 0.0,
        "final_equity": float(eq[-1]),
    }


def backtest(df: pd.DataFrame, cfg: TripleRSIConfig | None = None):
    cfg = cfg or TripleRSIConfig()
    x = df.copy()
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_convert("UTC").tz_localize(None)
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(x.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    d = x[["Open", "High", "Low", "Close"]].resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()
    d["rsi"] = _rsi(d["Close"], cfg.rsi_len)
    d["ma200"] = d["Close"].rolling(cfg.ma_len, min_periods=cfg.ma_len).mean()
    d["rsi_falling_3"] = (
        (d["rsi"] < d["rsi"].shift(1)) &
        (d["rsi"].shift(1) < d["rsi"].shift(2))
    )
    d["rsi_3d_ago_below_60"] = d["rsi"].shift(3) < cfg.rsi_three_days_ago_max
    d["rsi_cross_above_50"] = (
        (d["rsi"] > cfg.rsi_exit_level) &
        (d["rsi"].shift(1) <= cfg.rsi_exit_level)
    )

    trades, signals = [], []
    equity = float(cfg.initial_equity)
    equity_curve = [equity]
    position = None

    # Buy/sell at the completed daily close exactly as specified by the reference.
    for i in range(3, len(d)):
        row = d.iloc[i]

        if position is not None and bool(row["rsi_cross_above_50"]):
            exit_price = float(row["Close"])
            entry_price = position["entry"]
            ret = exit_price / entry_price - 1.0
            equity *= (1.0 + ret)
            trades.append({
                "entry_time": position["entry_time"].isoformat(),
                "exit_time": row.name.isoformat(),
                "side": "LONG",
                "entry": entry_price,
                "sl": np.nan,
                "tp": np.nan,
                "R": ret / max(cfg.risk_pct / 100.0, 1e-9),
                "return_pct": ret * 100.0,
                "equity": equity,
                "reason": "rsi5_cross_above_50",
                "risk_pct": cfg.risk_pct,
            })
            equity_curve.append(equity)
            position = None

        valid = (
            np.isfinite(row["rsi"]) and np.isfinite(row["ma200"]) and
            bool(row["rsi_falling_3"]) and bool(row["rsi_3d_ago_below_60"]) and
            float(row["rsi"]) < cfg.rsi_entry_max and
            float(row["Close"]) > float(row["ma200"])
        )
        signals.append({
            "signal_time": row.name.isoformat(),
            "rsi5": float(row["rsi"]),
            "rsi5_three_days_ago": float(d.iloc[i - 3]["rsi"]) if np.isfinite(d.iloc[i - 3]["rsi"]) else np.nan,
            "ma200": float(row["ma200"]) if np.isfinite(row["ma200"]) else np.nan,
            "rsi_falling_3": bool(row["rsi_falling_3"]),
            "rsi_3d_ago_below_60": bool(row["rsi_3d_ago_below_60"]),
            "signal": bool(valid),
        })

        if position is None and valid:
            position = {
                "entry_time": row.name,
                "entry": float(row["Close"]),
            }

    metrics = _metrics(trades, equity_curve, cfg.initial_equity, cfg.risk_pct)
    return {
        "data": d,
        "signals": pd.DataFrame(signals),
        "trades": trades,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "diagnostics": {
            "candidate_signals": int(sum(bool(s["signal"]) for s in signals)),
            "closed_trades": int(len(trades)),
            "open_position": position is not None,
        },
    }
