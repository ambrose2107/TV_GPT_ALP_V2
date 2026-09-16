"""
backtest.py
Orchestrates: load data -> build signals (with confluence funnel) -> run
engine -> return everything needed for reporting/visualization.
"""
from dataclasses import asdict
import pandas as pd

from .data_loader import get_data
from .signal_layer import SignalConfig, build_signals
from .engine import EngineConfig, simulate_trades


def run_backtest(symbol_interval: str = "5m", use_live: bool = True, n_bars: int = 20000,
                  signal_cfg: SignalConfig = None, engine_cfg: EngineConfig = None,
                  use_numba: bool = True) -> dict:
    signal_cfg = signal_cfg or SignalConfig()
    engine_cfg = engine_cfg or EngineConfig()

    data = get_data(interval=symbol_interval, use_live=use_live, n_bars=n_bars)
    sig_result = build_signals(data, signal_cfg)
    engine_result = simulate_trades(sig_result["df"], engine_cfg, use_numba=use_numba)

    return {
        "data": data,
        "df": engine_result["df"],
        "funnel": sig_result["funnel"],
        "confluence_data": sig_result["confluence_data"],
        "trades": engine_result["trades"],
        "equity_curve": engine_result["equity_curve"],
        "metrics": engine_result["metrics"],
        "used_numba": engine_result["used_numba"],
        "signal_cfg": asdict(signal_cfg),
        "engine_cfg": asdict(engine_cfg),
    }


def print_funnel_report(funnel: dict):
    print("\n" + "=" * 60)
    print("SIGNAL FUNNEL (diagnoses \"why did no trades fire\")")
    print("=" * 60)
    prev = None
    for stage, count in funnel.items():
        drop = "" if prev is None else f"  (dropped {prev - count})"
        print(f"  {stage:35s} {count:>8d}{drop}")
        prev = count
    print("=" * 60)
    if prev == 0:
        print("!! ZERO final signals. Check the funnel above to see which")
        print("   filter stage dropped to zero and relax that filter first.")
