"""
research/backtest_engine.py
Generic bar-by-bar backtest engine used by the dashboard's Backtest tab.

- Pulls OHLCV via yfinance directly (works for stocks, crypto e.g. BTC-USD,
  gold futures GC=F, forex pairs like EURUSD=X, etc.) - independent of the
  Alpaca-first core/market_data.py path, since that one is tuned for
  Alpaca-tradable US equities.
- Runs ONE registered strategy (see research/strategies/) bar by bar with
  no lookahead: a signal computed at bar i's close is executed at bar i+1's
  open.
- Risk-% position sizing, ATR-based stop, R-multiple take-profit, optional
  breakeven + ATR trailing.
- Returns JSON-serializable metrics + a downsampled equity curve + a capped
  trade list, so responses stay small enough for a synchronous web request.
"""
import numpy as np
import pandas as pd

from research.strategies import get_strategy

DEFAULT_PERIOD_BY_INTERVAL = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d",
    "30m": "60d", "1h": "730d", "1d": "5y",
}


def fetch_yf_data(symbol: str, interval: str = "5m", period: str = None) -> pd.DataFrame:
    import yfinance as yf

    period = period or DEFAULT_PERIOD_BY_INTERVAL.get(interval, "60d")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} @ {interval} (period={period}).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def run_backtest(
    df: pd.DataFrame,
    strategy_id: str,
    params: dict = None,
    initial_equity: float = 10000.0,
    risk_pct: float = 0.5,
    spread_points: float = 0.0,
    commission_pct: float = 0.0005,
    slippage_points: float = 0.0,
) -> dict:
    strat = get_strategy(strategy_id)
    if strat is None:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")

    merged_params = {**strat["default_params"], **(params or {})}
    d = strat["generate_signals"](df, merged_params)

    # need an ATR column for stop distance even if the strategy doesn't
    # already provide one under this name
    if "atr" not in d.columns:
        from research.strategies.indicators import atr as _atr
        d["atr"] = _atr(d, 14)

    d = d.dropna(subset=["atr"]).copy()
    if len(d) < 5:
        raise ValueError("Not enough bars after indicator warmup to backtest.")

    o, h, l, c = d["Open"].values, d["High"].values, d["Low"].values, d["Close"].values
    atr_vals = d["atr"].values
    signals = d["signal"].values
    idx = d.index

    equity = initial_equity
    equity_curve_t, equity_curve_v = [], []
    trades = []
    position = None

    atr_mult_sl = float(merged_params.get("atr_mult_sl", 1.5))
    r_multiple = float(merged_params.get("r_multiple", 2.0))
    use_trailing = bool(merged_params.get("use_trailing", False))
    breakeven_at_R = float(merged_params.get("breakeven_at_R", 1.0))
    trail_atr_mult = float(merged_params.get("trail_atr_mult", 1.2))

    for i in range(1, len(d)):
        if position is not None:
            side = position["side"]
            sl, tp = position["sl"], position["tp"]
            exit_price = exit_reason = None

            if side == 1:
                if l[i] <= sl:
                    exit_price, exit_reason = sl, "SL"
                elif h[i] >= tp:
                    exit_price, exit_reason = tp, "TP"
            else:
                if h[i] >= sl:
                    exit_price, exit_reason = sl, "SL"
                elif l[i] <= tp:
                    exit_price, exit_reason = tp, "TP"

            if exit_price is not None:
                gross = (exit_price - position["entry_price"]) * position["size"] * side
                cost = commission_pct * (exit_price * position["size"] + position["entry_price"] * position["size"])
                pnl = gross - cost
                equity += pnl
                trades.append({
                    "entry_time": str(position["entry_time"]), "exit_time": str(idx[i]),
                    "side": "LONG" if side == 1 else "SHORT",
                    "entry_price": round(position["entry_price"], 5),
                    "exit_price": round(exit_price, 5),
                    "size": round(position["size"], 6),
                    "pnl": round(pnl, 2), "reason": exit_reason,
                    "R": round(pnl / position["risk_amount"], 3) if position["risk_amount"] > 0 else None,
                })
                position = None
            elif use_trailing:
                move = (c[i] - position["entry_price"]) * side
                r_dist = position["r_dist"]
                if not position["breakeven_done"] and move >= breakeven_at_R * r_dist:
                    position["sl"] = position["entry_price"]
                    position["breakeven_done"] = True
                if position["breakeven_done"]:
                    trail_dist = trail_atr_mult * atr_vals[i]
                    new_sl = c[i] - side * trail_dist
                    position["sl"] = max(position["sl"], new_sl) if side == 1 else min(position["sl"], new_sl)

        if position is None and signals[i - 1] != 0:
            side = int(signals[i - 1])
            entry_price = o[i] + (spread_points / 2 + slippage_points) * side
            stop_dist = atr_mult_sl * atr_vals[i - 1]
            if stop_dist > 0 and not np.isnan(stop_dist):
                sl = entry_price - side * stop_dist
                tp = entry_price + side * stop_dist * r_multiple
                risk_amount = equity * (risk_pct / 100.0)
                size = risk_amount / stop_dist
                position = {
                    "side": side, "entry_price": entry_price, "sl": sl, "tp": tp,
                    "size": size, "entry_time": idx[i], "risk_amount": risk_amount,
                    "r_dist": stop_dist, "breakeven_done": False,
                }

        equity_curve_t.append(str(idx[i]))
        equity_curve_v.append(round(equity, 2))

    metrics = _compute_metrics(trades, equity_curve_v, initial_equity)

    # downsample equity curve to keep payload small
    max_points = 300
    n = len(equity_curve_v)
    if n > max_points:
        step = n // max_points
        ds_t = equity_curve_t[::step]
        ds_v = equity_curve_v[::step]
    else:
        ds_t, ds_v = equity_curve_t, equity_curve_v

    return {
        "strategy_id": strategy_id,
        "strategy_name": strat["name"],
        "params_used": merged_params,
        "metrics": metrics,
        "equity_curve": {"t": ds_t, "v": ds_v},
        "trades": trades[-200:],  # cap payload
        "num_bars": len(d),
        "data_range": {"start": str(idx[0]), "end": str(idx[-1])},
    }


def _compute_metrics(trades: list, equity_curve: list, initial_equity: float) -> dict:
    if not trades:
        return {
            "num_trades": 0, "win_rate": None, "profit_factor": None,
            "expectancy_R": None, "max_drawdown_pct": None,
            "total_return_pct": 0.0, "final_equity": round(initial_equity, 2),
        }

    pnls = [t["pnl"] for t in trades]
    rs = [t["R"] for t in trades if t["R"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else None)

    eq = np.array(equity_curve) if equity_curve else np.array([initial_equity])
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    max_dd = float(drawdown.min() * 100) if len(drawdown) else 0.0

    final_equity = eq[-1] if len(eq) else initial_equity
    total_return_pct = (final_equity / initial_equity - 1) * 100

    return {
        "num_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "profit_factor": (round(float(profit_factor), 2) if isinstance(profit_factor, (float, np.floating)) and np.isfinite(profit_factor)
                          else ("inf" if profit_factor == float("inf") else None)),
        "expectancy_R": round(float(np.mean(rs)), 3) if rs else None,
        "max_drawdown_pct": round(max_dd, 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "final_equity": round(float(final_equity), 2),
    }
