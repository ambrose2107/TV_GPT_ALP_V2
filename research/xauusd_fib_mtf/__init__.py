"""
research/xauusd_fib_mtf/ - 5m/15m Fibonacci + liquidity + market-structure
XAUUSD strategy package.

Public API:
    from research.xauusd_fib_mtf import run_backtest, MTFConfig, EngineConfig
"""
from .backtest import run_backtest, print_funnel_report
from .mtf_signal import MTFConfig
from .engine import EngineConfig, NUMBA_AVAILABLE
from .optimizer import walk_forward, grid_search, PARAM_GRID, MIN_TRADES

__all__ = [
    "run_backtest", "print_funnel_report", "MTFConfig", "EngineConfig", "NUMBA_AVAILABLE",
    "walk_forward", "grid_search", "PARAM_GRID", "MIN_TRADES",
]
