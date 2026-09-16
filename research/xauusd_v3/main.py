"""
main.py
Full pipeline:
  1. Run the "everything on" config first and print its funnel - this
     reproduces and diagnoses the "no trades initiated" issue directly.
  2. Compare a set of sensible configs (base signal, each confluence alone,
     a couple of light combinations) so you can see which actually
     preserve enough trades to be worth considering.
  3. Generate the full visual report for the best-performing config that
     still has a statistically meaningful trade count.
  4. Save everything (JSON report, trade CSV, all charts) for you to
     review and later wire into your website's backtest tab.

Run directly: python3 main.py
This sandbox can't reach yfinance, so it runs on synthetic data (clearly
labeled) to validate the whole pipeline end-to-end. Set use_live=True on
your own machine to run on real GC=F data.
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from .backtest import run_backtest, print_funnel_report
from .signal_layer import SignalConfig
from .engine import EngineConfig
from .visualize import plot_overview, plot_trade_detail, plot_fibonacci_levels, plot_zones, plot_funnel

OUT_DIR = "/mnt/user-data/outputs/xauusd_v3_report"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def main(interval: str = "5m", use_live: bool = True, n_bars: int = 30000):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("XAUUSD v3 PYTHON PIPELINE - INSTITUTIONAL CONFLUENCES")
    print("=" * 78)

    ecfg = EngineConfig()

    # ---- Step 1: reproduce & diagnose the "no trades" issue ----
    print("\n[1/3] Running the 'everything on' config (reproduces the no-trades issue)...")
    heavy_cfg = SignalConfig(
        use_fib=True, use_liquidity=True, use_sd=True, use_fvg=True, use_poi=True,
        use_accuracy_score=True, min_confluence_score=3, use_dxy=True, use_silver=True,
    )
    heavy_result = run_backtest(interval, use_live, n_bars, heavy_cfg, ecfg)
    print_funnel_report(heavy_result["funnel"])
    print(f"Result: {heavy_result['metrics']['num_trades']} trades with every filter stacked.")

    # ---- Step 2: compare sensible configs ----
    print("\n[2/3] Comparing base signal vs. individual confluences vs. light combos...")
    configs_to_test = {
        "base_no_filters": SignalConfig(),
        "fib_only": SignalConfig(use_fib=True),
        "liquidity_only": SignalConfig(use_liquidity=True),
        "supply_demand_only": SignalConfig(use_sd=True),
        "fvg_only": SignalConfig(use_fvg=True),
        "volume_poi_only": SignalConfig(use_poi=True),
        "confluence_score_2of5": SignalConfig(use_accuracy_score=True, min_confluence_score=2),
        "confluence_score_3of5": SignalConfig(use_accuracy_score=True, min_confluence_score=3),
        "dxy_only": SignalConfig(use_dxy=True),
        "silver_only": SignalConfig(use_silver=True),
    }

    comparison_rows = []
    results_by_name = {}
    data_cache = None
    for name, cfg in configs_to_test.items():
        result = run_backtest(interval, use_live, n_bars, cfg, ecfg)
        results_by_name[name] = result
        m = result["metrics"]
        comparison_rows.append({
            "config": name, "num_trades": m["num_trades"], "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"], "total_return_pct": m["total_return_pct"],
            "max_drawdown_pct": m["max_drawdown_pct"], "expectancy_R": m["expectancy_R"],
        })

    comp_df = pd.DataFrame(comparison_rows)
    print("\n" + comp_df.to_string(index=False))
    comp_df.to_csv(f"{OUT_DIR}/config_comparison.csv", index=False)

    # ---- Step 3: pick the best config with a meaningful trade count, generate full report ----
    MIN_TRADES_FOR_TRUST = 30
    eligible = comp_df[comp_df["num_trades"] >= MIN_TRADES_FOR_TRUST]
    if eligible.empty:
        print(f"\nNo config reached {MIN_TRADES_FOR_TRUST}+ trades on this data window. "
              f"Reporting on 'base_no_filters' instead so you have something to inspect.")
        best_name = "base_no_filters"
    else:
        best_name = eligible.loc[eligible["profit_factor"].astype(float).idxmax(), "config"] \
            if eligible["profit_factor"].apply(lambda x: isinstance(x, (int, float))).any() else eligible.iloc[0]["config"]
        best_name = eligible.sort_values("total_return_pct", ascending=False).iloc[0]["config"]

    print(f"\n[3/3] Generating full visual report for: '{best_name}'")
    best_result = results_by_name[best_name]
    best_cfg = configs_to_test[best_name]

    plot_overview(best_result["df"], best_result["trades"], best_result["equity_curve"],
                  f"{OUT_DIR}/overview.png", title=f"XAUUSD ({interval}) - {best_name}")
    plot_trade_detail(best_result["df"], best_result["trades"], f"{OUT_DIR}/trade_detail.png")
    plot_fibonacci_levels(best_result["df"], best_cfg.fib_swing_lookback, f"{OUT_DIR}/fibonacci_levels.png")
    plot_zones(best_result["df"], best_result["confluence_data"], f"{OUT_DIR}/institutional_zones.png")
    plot_funnel(heavy_result["funnel"], f"{OUT_DIR}/funnel_diagnostic.png")

    if not best_result["trades"].empty:
        best_result["trades"].to_csv(f"{OUT_DIR}/trades.csv", index=False)

    report = {
        "interval": interval,
        "used_numba": best_result["used_numba"],
        "heavy_config_funnel": heavy_result["funnel"],
        "config_comparison": comparison_rows,
        "best_config_name": best_name,
        "best_config": _jsonable(configs_to_test[best_name].__dict__),
        "best_config_metrics": best_result["metrics"],
    }
    with open(f"{OUT_DIR}/full_report.json", "w") as f:
        json.dump(_jsonable(report), f, indent=2, default=str)

    print(f"\nReport saved to {OUT_DIR}/")
    print("  - full_report.json, config_comparison.csv, trades.csv")
    print("  - overview.png, trade_detail.png, fibonacci_levels.png,")
    print("    institutional_zones.png, funnel_diagnostic.png")
    return report


if __name__ == "__main__":
    main()
