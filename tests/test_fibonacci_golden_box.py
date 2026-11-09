import numpy as np
import pandas as pd

from research.strategies.fibonacci_golden_box import generate_signals


def _sample_bars(n=420):
    # Smooth, deterministic oscillation with valid OHLC ranges.
    x = np.arange(n, dtype=float)
    close = 100 + 4*np.sin(x/7.0) + 1.5*np.sin(x/2.7) + 0.012*x
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.35
    low = np.minimum(open_, close) - 0.35
    return pd.DataFrame({"Open":open_, "High":high, "Low":low,
                         "Close":close, "Volume":1000.0},
                        index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))


def test_golden_box_returns_causal_columns():
    bars = _sample_bars()
    out = generate_signals(bars)
    for col in ("atr", "signal", "custom_stop_price", "custom_target_price",
                "golden_box_low", "golden_box_high", "swing_a", "swing_b", "swing_c"):
        assert col in out.columns
        assert len(out[col]) == len(bars)
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_golden_box_has_no_future_bar_dependency():
    bars = _sample_bars(180)
    full = generate_signals(bars)
    for end in (80, 110, 145, 179):
        prefix = generate_signals(bars.iloc[:end+1])
        pd.testing.assert_series_equal(
            full["signal"].iloc[:end+1].reset_index(drop=True),
            prefix["signal"].reset_index(drop=True),
        )
