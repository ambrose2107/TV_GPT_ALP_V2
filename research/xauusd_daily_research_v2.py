"""Daily research strategies transcribed from the supplied strategy screenshots.

Rules are implemented as written, with signals confirmed on the daily close and
executed at the next session open. These are research implementations; the
source screenshots report SPX results, while this lab can test the selected
ticker.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def _rsi(close, length):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _williams_r(df, length=2):
    hh = df.High.rolling(length, min_periods=length).max()
    ll = df.Low.rolling(length, min_periods=length).min()
    span = (hh - ll).replace(0, np.nan)
    return -100 * (hh - df.Close) / span


def _cci(df, length=16):
    tp = (df.High + df.Low + df.Close) / 3.0
    ma = tp.rolling(length, min_periods=length).mean()
    md = tp.rolling(length, min_periods=length).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _metrics(trades, initial_equity, risk_pct):
    if not trades:
        return {
            "num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy_R": 0.0, "total_R": 0.0, "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0, "final_equity": round(initial_equity, 2)
        }
    t = pd.DataFrame(trades)
    returns = t["return_pct"].astype(float)
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    r = returns / max(float(risk_pct), 1e-9)
    equity = t["equity"].astype(float)
    dd = float((equity / equity.cummax() - 1).min() * 100)
    return {
        "num_trades": int(len(t)),
        "win_rate": round(float((returns > 0).mean() * 100), 2),
        "profit_factor": round(gains / losses, 2) if losses else float("inf"),
        "expectancy_R": round(float(r.mean()), 3),
        "total_R": round(float(r.sum()), 2),
        "max_drawdown_pct": round(dd, 2),
        "total_return_pct": round((float(equity.iloc[-1]) / initial_equity - 1) * 100, 2),
        "final_equity": round(float(equity.iloc[-1]), 2)
    }


def _run_long_only(d, entry_signal, exit_signal, cfg, name):
    trades, signals = [], []
    equity = float(cfg.initial_equity)
    pos = None
    curve = []

    for i in range(len(d)):
        row = d.iloc[i]
        ts = d.index[i]

        # Exit on the confirmed daily close.
        if pos is not None and bool(exit_signal.iloc[i]):
            exit_price = float(row.Close)
            ret = exit_price / pos["entry"] - 1.0
            equity *= 1.0 + ret
            trades.append({
                "entry_time": pos["entry_time"].isoformat(),
                "exit_time": ts.isoformat(),
                "side": "LONG",
                "entry": pos["entry"],
                "exit_price": exit_price,
                "R": ret / (cfg.risk_pct / 100.0),
                "return_pct": ret * 100.0,
                "equity": equity,
                "reason": pos["reason"] + " | exit"
            })
            pos = None

        # Confirm signal at today's close; enter next session open.
        valid = bool(entry_signal.iloc[i])
        signals.append({
            "signal_time": ts.isoformat(),
            "signal": valid,
            "close": float(row.Close),
        })
        if pos is None and valid and i + 1 < len(d):
            nxt = d.iloc[i + 1]
            pos = {
                "entry_time": d.index[i + 1],
                "entry": float(nxt.Open),
                "reason": name
            }

        curve.append(equity)

    return {
        "data": d,
        "signals": pd.DataFrame(signals),
        "trades": trades,
        "equity_curve": curve,
        "metrics": _metrics(trades, cfg.initial_equity, cfg.risk_pct)
    }


@dataclass
class WilliamsRConfig:
    length: int = 2
    entry_level: float = -98.0
    ma_len: int = 175
    exit_level: float = -50.0
    risk_pct: float = 0.5
    initial_equity: float = 10000.0


def backtest_williams_r(df, cfg=WilliamsRConfig()):
    d = df[["Open", "High", "Low", "Close"]].copy().sort_index()
    d["williams_r"] = _williams_r(d, cfg.length)
    d["ma175"] = d.Close.rolling(cfg.ma_len, min_periods=cfg.ma_len).mean()
    entry = (d.williams_r < cfg.entry_level) & (d.Close > d.ma175)
    exit_ = d.williams_r > cfg.exit_level
    return _run_long_only(d, entry, exit_, cfg, "Williams %R")


@dataclass
class CCIConfig:
    length: int = 16
    entry_level: float = -180.0
    exit_level: float = 150.0
    risk_pct: float = 0.5
    initial_equity: float = 10000.0


def backtest_cci(df, cfg=CCIConfig()):
    d = df[["Open", "High", "Low", "Close"]].copy().sort_index()
    d["cci"] = _cci(d, cfg.length)
    entry = (d.cci > cfg.entry_level) & (d.cci.shift(1) <= cfg.entry_level)
    exit_ = d.cci > cfg.exit_level
    return _run_long_only(d, entry, exit_, cfg, "CCI")


@dataclass
class MultiHorizonRSIConfig:
    rsi_fast: int = 5
    rsi_mid: int = 14
    rsi_slow: int = 50
    entry_fast_max: float = 45.0
    entry_mid_max: float = 65.0
    entry_slow_max: float = 55.0
    exit_fast: float = 90.0
    exit_mid: float = 65.0
    risk_pct: float = 0.5
    initial_equity: float = 10000.0


def backtest_multi_rsi(df, cfg=MultiHorizonRSIConfig()):
    d = df[["Open", "High", "Low", "Close"]].copy().sort_index()
    d["rsi5"] = _rsi(d.Close, cfg.rsi_fast)
    d["rsi14"] = _rsi(d.Close, cfg.rsi_mid)
    d["rsi50"] = _rsi(d.Close, cfg.rsi_slow)
    entry = (
        (d.rsi5 < cfg.entry_fast_max)
        & (d.rsi14 < cfg.entry_mid_max)
        & (d.rsi50 < cfg.entry_slow_max)
    )
    exit_ = (d.rsi5 > cfg.exit_fast) | (d.rsi14 > cfg.exit_mid)
    return _run_long_only(d, entry, exit_, cfg, "Triple RSI Multi-Horizon")
