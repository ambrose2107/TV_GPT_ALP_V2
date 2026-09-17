"""
main.py
Full pipeline: funnel diagnostic -> auto-optimizer (grid search + walk-
forward) -> final report on the winning config.

Run directly: python3 main.py
This sandbox has no network access to Yahoo Finance, so it runs on
synthetic data (clearly labeled, not real results) to validate the whole
pipeline end to end. Set use_live=True on your machine for real data.

IMPORTANT: this strategy is intentionally very selective (multiple
independent confluences must align across two timeframes). Yahoo's free
5-minute history is capped at ~60 days, which is NOT enough data for this
strategy's expected trade frequency to produce a statistically trustworthy
sample. For real use, source months of 5-minute XAUUSD history from your
broker (MT4/5 export, Dukascopy, etc.) - see README for details.
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import asdict, replace

from .data_loader import get_mtf_data
from .mtf_signal import MTFConfig
from .engine import EngineConfig, simulate_trades
from .backtest import run_backtest, print_funnel_report
from optimizer import walk_forward, grid_search, PARAM_GRID, _grid_combos
from .visualize import plot_overview, plot_fib_and_trade_detail, plot_funnel

OUT_DIR = "/mnt/user-data/outputs/xauusd_fib_mtf_report"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def main(use_live: bool = True, n_bars: int = 60000, n_folds: int = 2, demo_min_trades: int = 3):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("XAUUSD 5m/15m FIBONACCI + LIQUIDITY + MARKET STRUCTURE STRATEGY")
    print("=" * 78)

    base_cfg = MTFConfig()
    engine_cfg = EngineConfig()

    print("\n[1/3] Loading multi-timeframe data and running default-config diagnostic...")
    data = get_mtf_data(use_live=use_live, n_bars=n_bars)
    baseline = run_backtest(mtf_cfg=base_cfg, engine_cfg=engine_cfg, data=data)
    print_funnel_report(baseline["funnel"])
    print(f"Default config result: {baseline['metrics']['num_trades']} trades.")

    print("\n[2/3] Running auto-optimizer (grid search + walk-forward)...")
    print(f"NOTE: using demo_min_trades={demo_min_trades} for this run (the sample here is small "
          f"and/or synthetic). For real use, raise this back toward optimizer.MIN_TRADES (20+) once "
          f"you have months of real data - see README.")
    combos = list(_grid_combos(PARAM_GRID))
    full_5m = data["m5"]
    wf = walk_forward(full_5m, n_folds=n_folds, base_cfg=base_cfg, engine_cfg=engine_cfg,
                       grid_combos=combos, min_trades=demo_min_trades)
    print("Walk-forward summary:", wf["summary"])
    for fr in wf["folds"]:
        print(" ", fr)

    full_grid = grid_search(data, combos, base_cfg, engine_cfg, min_trades=demo_min_trades)

    if full_grid.empty or full_grid.iloc[0]["score"] <= -1e8:
        print("\nNo configuration reached even the relaxed demo trade-count threshold on this "
              "sample. Reporting on the DEFAULT config instead.")
        best_cfg = base_cfg
        best_result = baseline
    else:
        best_row = full_grid.iloc[0]
        best_params = {k: best_row[k] for k in PARAM_GRID.keys()}
        best_cfg = replace(base_cfg, **best_params)
        print(f"\nBest config found: {best_params}")
        best_result = run_backtest(mtf_cfg=best_cfg, engine_cfg=engine_cfg, data=data)

    print("\n[3/3] Generating final report...")
    print("Best/default config metrics:", best_result["metrics"])

    plot_overview(best_result["df5"], best_result["trades"], best_result["equity_curve"],
                  f"{OUT_DIR}/overview.png")
    if not best_result["trades"].empty:
        plot_fib_and_trade_detail(best_result["df5"], best_result["df15"], best_result["trades"],
                                   f"{OUT_DIR}/trade_detail.png")
        best_result["trades"].to_csv(f"{OUT_DIR}/trades.csv", index=False)
    plot_funnel(best_result["funnel"], f"{OUT_DIR}/funnel.png")

    report = {
        "used_numba": best_result["used_numba"],
        "default_config_funnel": baseline["funnel"],
        "default_config_metrics": baseline["metrics"],
        "walk_forward_summary": wf["summary"],
        "best_config": _jsonable(asdict(best_cfg)),
        "best_config_metrics": best_result["metrics"],
        "best_config_funnel": best_result["funnel"],
    }
    with open(f"{OUT_DIR}/full_report.json", "w") as f:
        json.dump(_jsonable(report), f, indent=2, default=str)

    print(f"\nReport saved to {OUT_DIR}/")
    return report


if __name__ == "__main__":
    main()
