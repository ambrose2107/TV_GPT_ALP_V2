"""
engine.py
Trade simulation engine: entry on signal-bar close, ATR stop, up to 4
progressive partial targets, breakeven-after-first-partial, risk-%
position sizing, R-multiple accounting.

Performance: the core bar-by-bar loop is JIT-compiled with numba when
available (falls back to a pure-Python/NumPy loop otherwise - same logic,
just slower). All state lives in flat NumPy arrays (no Python objects in
the hot loop) so the numba version compiles in nopython mode.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrap


@dataclass
class EngineConfig:
    sl_atr_mult: float = 1.5
    rr: float = 2.0
    n_targets: int = 4
    move_sl_to_breakeven_after_first_tp: bool = True
    risk_pct: float = 0.5
    initial_equity: float = 10_000.0
    commission_pct: float = 0.0005
    spread_points: float = 0.20
    slippage_points: float = 0.05


def _core_loop_impl(open_, high, low, close, atr, signal,
                     sl_atr_mult, rr, n_targets, risk_pct, initial_equity,
                     commission_pct, spread_points, slippage_points, move_sl_to_be):
    n = len(open_)
    max_trades = n

    t_entry_idx = np.full(max_trades, -1, dtype=np.int64)
    t_exit_idx = np.full(max_trades, -1, dtype=np.int64)
    t_side = np.zeros(max_trades, dtype=np.int64)
    t_entry_price = np.zeros(max_trades)
    t_exit_price = np.full(max_trades, np.nan)
    t_sl_initial = np.zeros(max_trades)
    t_sl_final = np.zeros(max_trades)
    t_risk_distance = np.zeros(max_trades)
    t_risk_money = np.zeros(max_trades)
    t_tp = np.zeros((max_trades, 4))
    t_be_triggered = np.zeros(max_trades, dtype=np.int64)
    t_be_idx = np.full(max_trades, -1, dtype=np.int64)
    t_pnl = np.zeros(max_trades)
    t_targets_hit = np.zeros(max_trades, dtype=np.int64)

    equity_curve = np.zeros(n)

    equity = initial_equity
    n_trades = 0

    in_position = False
    pos_side = 0
    pos_entry_price = 0.0
    pos_sl = 0.0
    pos_risk_distance = 0.0
    pos_size_total = 0.0
    pos_remaining_frac = 1.0
    pos_next_target = 0
    pos_targets = np.zeros(4)
    pos_n_targets = 0
    pos_trade_idx = -1
    pos_be_done = False

    for i in range(n):
        if in_position:
            side = pos_side
            sl = pos_sl

            sl_touched = (low[i] <= sl) if side == 1 else (high[i] >= sl)
            tp_touched = False
            if pos_next_target < pos_n_targets:
                tgt = pos_targets[pos_next_target]
                tp_touched = (high[i] >= tgt) if side == 1 else (low[i] <= tgt)

            if sl_touched and tp_touched:
                # conservative sl_first precedence
                exit_price = sl
                notional = pos_size_total * pos_remaining_frac
                gross = (exit_price - pos_entry_price) * notional * side
                cost = commission_pct * exit_price * notional
                pnl = gross - cost
                t_pnl[pos_trade_idx] += pnl
                equity += pnl
                t_exit_idx[pos_trade_idx] = i
                t_exit_price[pos_trade_idx] = exit_price
                t_sl_final[pos_trade_idx] = pos_sl
                t_be_triggered[pos_trade_idx] = 1 if pos_be_done else 0
                in_position = False
            elif sl_touched:
                exit_price = sl
                notional = pos_size_total * pos_remaining_frac
                gross = (exit_price - pos_entry_price) * notional * side
                cost = commission_pct * exit_price * notional
                pnl = gross - cost
                t_pnl[pos_trade_idx] += pnl
                equity += pnl
                t_exit_idx[pos_trade_idx] = i
                t_exit_price[pos_trade_idx] = exit_price
                t_sl_final[pos_trade_idx] = pos_sl
                t_be_triggered[pos_trade_idx] = 1 if pos_be_done else 0
                in_position = False
            elif tp_touched:
                keep_filling = True
                while keep_filling:
                    if pos_next_target >= pos_n_targets or pos_remaining_frac <= 1e-9:
                        keep_filling = False
                        continue
                    tgt = pos_targets[pos_next_target]
                    touched_now = (high[i] >= tgt) if side == 1 else (low[i] <= tgt)
                    if not touched_now:
                        keep_filling = False
                        continue
                    frac = 1.0 / pos_n_targets
                    if frac > pos_remaining_frac:
                        frac = pos_remaining_frac
                    notional = pos_size_total * frac
                    gross = (tgt - pos_entry_price) * notional * side
                    cost = commission_pct * tgt * notional
                    pnl = gross - cost
                    t_pnl[pos_trade_idx] += pnl
                    equity += pnl
                    pos_remaining_frac -= frac
                    pos_next_target += 1
                    t_targets_hit[pos_trade_idx] = pos_next_target
                    if move_sl_to_be and (not pos_be_done) and pos_next_target == 1:
                        pos_sl = pos_entry_price
                        pos_be_done = True
                        t_be_triggered[pos_trade_idx] = 1
                        t_be_idx[pos_trade_idx] = i

                if pos_remaining_frac <= 1e-9 or pos_next_target >= pos_n_targets:
                    if pos_remaining_frac > 1e-9:
                        tgt = pos_targets[pos_n_targets - 1]
                        notional = pos_size_total * pos_remaining_frac
                        gross = (tgt - pos_entry_price) * notional * side
                        cost = commission_pct * tgt * notional
                        pnl = gross - cost
                        t_pnl[pos_trade_idx] += pnl
                        equity += pnl
                        pos_remaining_frac = 0.0
                    t_exit_idx[pos_trade_idx] = i
                    t_exit_price[pos_trade_idx] = exit_price
                    t_sl_final[pos_trade_idx] = pos_sl
                    in_position = False

        if (not in_position) and signal[i] != 0:
            side = signal[i]
            entry_price = close[i] + (spread_points / 2.0 + slippage_points) * side
            risk_distance = sl_atr_mult * atr[i]
            if risk_distance > 0.0 and not np.isnan(risk_distance):
                sl0 = entry_price - side * risk_distance
                total_target_dist = rr * risk_distance
                n_t = n_targets
                if n_t < 1:
                    n_t = 1
                if n_t > 4:
                    n_t = 4

                risk_money = equity * (risk_pct / 100.0)
                size_total = risk_money / risk_distance
                entry_cost = commission_pct * entry_price * size_total
                equity -= entry_cost

                idx = n_trades
                t_entry_idx[idx] = i
                t_side[idx] = side
                t_entry_price[idx] = entry_price
                t_sl_initial[idx] = sl0
                t_sl_final[idx] = sl0
                t_risk_distance[idx] = risk_distance
                t_risk_money[idx] = risk_money
                t_pnl[idx] = -entry_cost
                for k in range(4):
                    if k < n_t:
                        t_tp[idx, k] = entry_price + side * ((k + 1) / n_t) * total_target_dist
                    else:
                        t_tp[idx, k] = np.nan

                pos_side = side
                pos_entry_price = entry_price
                pos_sl = sl0
                pos_risk_distance = risk_distance
                pos_size_total = size_total
                pos_remaining_frac = 1.0
                pos_next_target = 0
                for k in range(4):
                    pos_targets[k] = t_tp[idx, k]
                pos_n_targets = n_t
                pos_trade_idx = idx
                pos_be_done = False
                in_position = True
                n_trades += 1

        equity_curve[i] = equity

    if in_position:
        exit_price = close[n - 1]
        notional = pos_size_total * pos_remaining_frac
        gross = (exit_price - pos_entry_price) * notional * pos_side
        cost = commission_pct * exit_price * notional
        pnl = gross - cost
        t_pnl[pos_trade_idx] += pnl
        equity += pnl
        t_exit_idx[pos_trade_idx] = n - 1
        t_exit_price[pos_trade_idx] = exit_price
        t_sl_final[pos_trade_idx] = pos_sl
        equity_curve[n - 1] = equity

    return (t_entry_idx[:n_trades], t_exit_idx[:n_trades], t_side[:n_trades], t_entry_price[:n_trades],
            t_exit_price[:n_trades], t_sl_initial[:n_trades], t_sl_final[:n_trades], t_risk_distance[:n_trades], t_risk_money[:n_trades],
            t_tp[:n_trades], t_be_triggered[:n_trades], t_be_idx[:n_trades], t_pnl[:n_trades],
            t_targets_hit[:n_trades], equity_curve)


_core_loop_numba = njit(cache=True)(_core_loop_impl) if NUMBA_AVAILABLE else _core_loop_impl


def simulate_trades(df: pd.DataFrame, cfg: EngineConfig, use_numba: bool = True) -> dict:
    """
    df must have: Open, High, Low, Close, atr, signal.
    Returns a dict with a trades DataFrame, equity curve (pd.Series), and metrics.
    """
    d = df.dropna(subset=["atr"]).copy()
    open_ = d["Open"].values.astype(np.float64)
    high = d["High"].values.astype(np.float64)
    low = d["Low"].values.astype(np.float64)
    close = d["Close"].values.astype(np.float64)
    atr_v = d["atr"].values.astype(np.float64)
    signal = d["signal"].values.astype(np.int64)
    idx = d.index

    fn = _core_loop_numba if (use_numba and NUMBA_AVAILABLE) else _core_loop_impl
    (entry_idx, exit_idx, side, entry_price, exit_price, sl_initial, sl_final, risk_distance, risk_money,
     tp, be_triggered, be_idx, pnl, targets_hit, equity_curve) = fn(
        open_, high, low, close, atr_v, signal,
        cfg.sl_atr_mult, cfg.rr, cfg.n_targets, cfg.risk_pct, cfg.initial_equity,
        cfg.commission_pct, cfg.spread_points, cfg.slippage_points, cfg.move_sl_to_breakeven_after_first_tp,
    )

    n_trades = len(entry_idx)
    trades = pd.DataFrame({
        "entry_time": [idx[i] for i in entry_idx],
        "exit_time": [idx[i] if i >= 0 else None for i in exit_idx],
        "side": np.where(side == 1, "LONG", "SHORT"),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "sl_initial": sl_initial,
        "sl_final": sl_final,
        "tp1": tp[:, 0] if n_trades else [],
        "tp2": tp[:, 1] if n_trades else [],
        "tp3": tp[:, 2] if n_trades else [],
        "tp4": tp[:, 3] if n_trades else [],
        "breakeven_triggered": be_triggered.astype(bool),
        "breakeven_time": [idx[i] if i >= 0 else None for i in be_idx],
        "risk_distance": risk_distance,
        "risk_money": risk_money,
        "pnl": pnl,
        "R": np.where(risk_money > 0, pnl / np.where(risk_money > 0, risk_money, 1), np.nan),
        "targets_hit": targets_hit,
    })

    equity_series = pd.Series(equity_curve, index=idx)
    metrics = compute_metrics(trades, equity_series, cfg.initial_equity)
    return {"trades": trades, "equity_curve": equity_series, "metrics": metrics, "df": d, "used_numba": fn is _core_loop_numba}


def compute_metrics(trades: pd.DataFrame, equity_curve: pd.Series, initial_equity: float) -> dict:
    if trades.empty:
        return {
            "num_trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
            "expectancy_R": np.nan, "total_R": 0.0, "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0, "final_equity": initial_equity,
        }

    pnls = trades["pnl"].values
    rs = trades["R"].dropna().values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(trades) * 100
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else np.nan)

    eq_vals = equity_curve.values
    running_max = np.maximum.accumulate(eq_vals) if len(eq_vals) else np.array([initial_equity])
    drawdown = (eq_vals - running_max) / running_max
    max_dd = drawdown.min() * 100 if len(drawdown) else 0.0
    final_equity = eq_vals[-1] if len(eq_vals) else initial_equity
    total_return_pct = (final_equity / initial_equity - 1) * 100

    return {
        "num_trades": int(len(trades)),
        "win_rate": round(float(win_rate), 2),
        "profit_factor": round(float(profit_factor), 3) if np.isfinite(profit_factor) else (
            "inf" if profit_factor == np.inf else None),
        "expectancy_R": round(float(np.mean(rs)), 3) if len(rs) else np.nan,
        "total_R": round(float(np.sum(rs)), 2) if len(rs) else 0.0,
        "max_drawdown_pct": round(float(max_dd), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "final_equity": round(float(final_equity), 2),
    }
