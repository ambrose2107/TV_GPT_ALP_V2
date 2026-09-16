"""
research/xauusd_v3/ - Institutional confluences XAUUSD strategy package.

Public API:
    from research.xauusd_v3 import run_backtest, SignalConfig, EngineConfig

See backtest.py for run_backtest()'s signature and README_INTEGRATION.md
(shipped alongside this package) for how to wire this into the dashboard.
"""
from .backtest import run_backtest, print_funnel_report
from .signal_layer import SignalConfig
from .engine import EngineConfig, NUMBA_AVAILABLE

__all__ = ["run_backtest", "print_funnel_report", "SignalConfig", "EngineConfig", "NUMBA_AVAILABLE"]
