"""
optimizer.py
Same statistically-disciplined approach as the earlier framework: joint
grid search over the strategy's tunable knobs, an objective that penalizes
too-few-trades and excessive drawdown, and walk-forward validation (train
window selects the config, an untouched test window scores it) so you're
not just curve-fitting to one historical stretch.

Given how selective this strategy is by design (several independent rare
conditions must align), the grid here focuses on the knobs that most
directly control selectivity vs. frequency: impulse quality thresholds,
confluence count required, and the 5m sweep timing window.
"""
import itertools
from dataclasses import replace
import numpy as np
import pandas as pd

from .mtf_signal import MTFConfig, build_mtf_signals
from .engine import EngineConfig, simulate_trades

MIN_TRADES = 20  # realistic bar for trusting results with proper historical data

PARAM_GRID = {
    "min_confluences": [2, 3],
    "min_impulse_efficiency": [0.3, 0.4],
    "min_impulse_atr_mult": [2.0, 3.0],
    "sweep_window_5m": [6, 10, 15],
    "zone_tolerance_atr": [0.3, 0.5],
}


def _grid_combos(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def score_config(metrics: dict, min_trades: int = MIN_TRADES) -> float:
    if metrics["num_trades"] < min_trades:
        return -1e9
    pf = metrics["profit_factor"] if isinstance(metrics["profit_factor"], (int, float)) and np.isfinite(metrics["profit_factor"]) else 3.0
    dd_penalty = abs(metrics["max_drawdown_pct"]) / 100.0
    expectancy = metrics["expectancy_R"] if pd.notna(metrics["expectancy_R"]) else -1.0
    return pf * 1.0 + expectancy * 2.0 - dd_penalty * 3.0


def evaluate_config(data: dict, params: dict, base_cfg: MTFConfig, engine_cfg: EngineConfig) -> dict:
    cfg = replace(base_cfg, **params)
    sig_result = build_mtf_signals(data, cfg)
    engine_result = simulate_trades(sig_result["df5"], engine_cfg)
    return {"metrics": engine_result["metrics"], "funnel": sig_result["funnel"]}


def grid_search(data: dict, grid_combos: list, base_cfg: MTFConfig, engine_cfg: EngineConfig,
                 min_trades: int = MIN_TRADES) -> pd.DataFrame:
    rows = []
    for params in grid_combos:
        try:
            result = evaluate_config(data, params, base_cfg, engine_cfg)
        except Exception:
            continue
        row = {**params, **result["metrics"], "score": score_config(result["metrics"], min_trades)}
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def walk_forward(full_5m_df: pd.DataFrame, n_folds: int, base_cfg: MTFConfig, engine_cfg: EngineConfig,
                  grid_combos: list = None, min_trades: int = MIN_TRADES):
    """
    Splits the 5m series into n_folds consecutive windows (70% train / 30%
    test each), resamples 15m fresh within each slice (so no cross-window
    leakage), grid-searches on train, scores the frozen winner on test.
    """
    from data_loader import resample_to_15m

    grid_combos = grid_combos or list(_grid_combos(PARAM_GRID))
    n = len(full_5m_df)
    fold_size = n // n_folds
    fold_results = []

    for f in range(n_folds):
        start = f * fold_size
        end = n if f == n_folds - 1 else (f + 1) * fold_size
        fold_5m = full_5m_df.iloc[start:end]
        split = int(len(fold_5m) * 0.7)
        train_5m, test_5m = fold_5m.iloc[:split], fold_5m.iloc[split:]
        if len(train_5m) < 2000 or len(test_5m) < 800:
            continue

        train_data = {"m5": train_5m, "m15": resample_to_15m(train_5m)}
        test_data = {"m5": test_5m, "m15": resample_to_15m(test_5m)}

        train_results = grid_search(train_data, grid_combos, base_cfg, engine_cfg, min_trades)
        if train_results.empty or train_results.iloc[0]["score"] <= -1e8:
            fold_results.append({"fold": f, "status": "no_valid_config_on_train"})
            continue

        best_row = train_results.iloc[0]
        best_params = {k: best_row[k] for k in PARAM_GRID.keys()}
        test_eval = evaluate_config(test_data, best_params, base_cfg, engine_cfg)
        test_score = score_config(test_eval["metrics"], min_trades)

        fold_results.append({
            "fold": f, "status": "ok", "best_params": best_params,
            "train_score": best_row["score"], "test_metrics": test_eval["metrics"], "test_score": test_score,
        })

    valid_folds = [fr for fr in fold_results if fr.get("status") == "ok" and fr["test_score"] > -1e8]
    if valid_folds:
        oos_returns = [fr["test_metrics"]["total_return_pct"] for fr in valid_folds]
        summary = {
            "n_folds_run": len(fold_results), "n_folds_valid": len(valid_folds),
            "avg_oos_return_pct": round(float(np.mean(oos_returns)), 2),
            "pct_folds_profitable": round(100 * float(np.mean([r > 0 for r in oos_returns])), 1),
        }
    else:
        summary = {"n_folds_run": len(fold_results), "n_folds_valid": 0,
                   "note": "No fold produced a config with enough trades to trust. "
                           "This strategy is highly selective by design - try more historical "
                           "data (this needs real months of 5m bars, not a short synthetic sample) "
                           "or widen PARAM_GRID."}
    return {"folds": fold_results, "summary": summary}
