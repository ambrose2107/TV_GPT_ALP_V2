"""Gold Quad-RSI mean-reversion strategy.

Research implementation of the rule set shown in the supplied reference:
- RSI(3) below an oversold threshold
- at least N RSI declines in the last M sessions
- Internal Bar Strength (IBS) below a threshold
- close above a long-term moving average
- enter at the next session open
- exit after a close above the prior session high, at the following open
- one position at a time

The default is deliberately close to the published rule set. No optimization is
performed inside this module.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class QuadRSIConfig:
    rsi_len: int = 3
    rsi_oversold: float = 25.0
    decline_window: int = 10
    min_rsi_declines: int = 4
    ibs_max: float = 10.0
    ma_len: int = 150
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
    out = out.where(avg_loss.ne(0), 100.0)
    return out


def _metrics(trades, equity_curve, initial_equity):
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": np.nan,
            "expectancy_R": 0.0,
            "total_R": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "final_equity": float(initial_equity),
        }

    rets = np.asarray([float(t["return_pct"]) / 100.0 for t in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (np.inf if len(wins) else np.nan)

    # R is a normalized research statistic: trade return divided by configured
    # risk percentage. It is not a broker position-size calculation.
    risk = max(float(trades[0].get("risk_pct", 0.5)), 1e-9)
    rs = rets / (risk / 100.0)
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


def backtest(df: pd.DataFrame, cfg: QuadRSIConfig | None = None):
    cfg = cfg or QuadRSIConfig()
    x = df.copy()
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_convert("UTC").tz_localize(None)

    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(x.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Work on completed daily sessions. For GLD this follows the available
    # exchange-session bars; for a true 24h XAUUSD feed, use its sessionized data.
    d = x[["Open", "High", "Low", "Close"]].resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()
    d["rsi"] = _rsi(d["Close"], cfg.rsi_len)
    d["ma"] = d["Close"].rolling(cfg.ma_len, min_periods=cfg.ma_len).mean()
    rng = (d["High"] - d["Low"]).replace(0.0, np.nan)
    d["ibs"] = ((d["Close"] - d["Low"]) / rng) * 100.0
    d["rsi_decline"] = d["rsi"] < d["rsi"].shift(1)
    d["declines"] = d["rsi_decline"].rolling(cfg.decline_window).sum()

    trades = []
    equity = float(cfg.initial_equity)
    equity_curve = [equity]
    position = None
    signals = []

    # Signal is evaluated at a completed session close. Entry is next session open.
    for i in range(1, len(d) - 1):
        row = d.iloc[i]
        next_row = d.iloc[i + 1]

        # Exit is deliberately checked from a completed session and executed
        # at the following session open.
        if position is not None and float(row["Close"]) > float(d.iloc[i - 1]["High"]):
            exit_price = float(next_row["Open"])
            entry_price = position["entry"]
            ret = exit_price / entry_price - 1.0
            pnl = equity * ret
            equity += pnl
            trades.append({
                "entry_time": position["entry_time"].isoformat(),
                "exit_time": next_row.name.isoformat(),
                "side": "LONG",
                "entry": entry_price,
                "sl": np.nan,
                "tp": np.nan,
                "R": ret / (cfg.risk_pct / 100.0),
                "return_pct": ret * 100.0,
                "equity": equity,
                "reason": "close_above_yesterday_high",
                "risk_pct": cfg.risk_pct,
            })
            equity_curve.append(equity)
            position = None

        if position is not None:
            continue

        valid = (
            np.isfinite(row["rsi"]) and
            np.isfinite(row["ma"]) and
            np.isfinite(row["ibs"]) and
            np.isfinite(row["declines"]) and
            float(row["rsi"]) < cfg.rsi_oversold and
            float(row["declines"]) >= cfg.min_rsi_declines and
            float(row["ibs"]) < cfg.ibs_max and
            float(row["Close"]) > float(row["ma"])
        )
        signals.append({
            "signal_time": row.name.isoformat(),
            "rsi": float(row["rsi"]) if np.isfinite(row["rsi"]) else np.nan,
            "rsi_declines": float(row["declines"]) if np.isfinite(row["declines"]) else np.nan,
            "ibs": float(row["ibs"]) if np.isfinite(row["ibs"]) else np.nan,
            "ma150": float(row["ma"]) if np.isfinite(row["ma"]) else np.nan,
            "signal": bool(valid),
        })

        if valid:
            position = {
                "entry_time": next_row.name,
                "entry": float(next_row["Open"]),
            }

    # An open position is excluded from realized metrics rather than inventing
    # an exit at the end of the dataset.
    metrics = _metrics(trades, equity_curve, cfg.initial_equity)
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
