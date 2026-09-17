"""
backtest.py
Orchestrates: load MTF data -> build MTF signals (with funnel) -> run
engine -> return everything for reporting.
"""
from dataclasses import asdict

from .data_loader import get_mtf_data
from .mtf_signal import MTFConfig, build_mtf_signals
from .engine import EngineConfig, simulate_trades


def run_backtest(use_live: bool = True, n_bars: int = 20000,
                  mtf_cfg: MTFConfig = None, engine_cfg: EngineConfig = None,
                  use_numba: bool = True, data: dict = None,
                  data_source: str = "yfinance", lookback_days: int = None,
                  alpaca_symbol: str = "GLD") -> dict:
    mtf_cfg = mtf_cfg or MTFConfig()
    engine_cfg = engine_cfg or EngineConfig()

    data = data if data is not None else get_mtf_data(
        use_live=use_live, n_bars=n_bars, data_source=data_source,
        lookback_days=lookback_days, alpaca_symbol=alpaca_symbol,
    )
    sig_result = build_mtf_signals(data, mtf_cfg)
    engine_result = simulate_trades(sig_result["df5"], engine_cfg, use_numba=use_numba)

    return {
        "df5": engine_result["df"], "df15": sig_result["df15"], "data": data,
        "funnel": sig_result["funnel"],
        "trades": engine_result["trades"], "equity_curve": engine_result["equity_curve"],
        "metrics": engine_result["metrics"], "used_numba": engine_result["used_numba"],
        "mtf_cfg": asdict(mtf_cfg), "engine_cfg": asdict(engine_cfg),
    }


def print_funnel_report(funnel: dict):
    print("\n" + "=" * 68)
    print("SIGNAL FUNNEL (15m bias/impulse/fib/confluence -> 5m confirmation)")
    print("=" * 68)
    for stage, count in funnel.items():
        print(f"  {stage:40s} {count:>8d}")
    print("=" * 68)
