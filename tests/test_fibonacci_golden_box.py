import numpy as np
import pandas as pd
import pytest

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
                "golden_box_low", "golden_box_high", "swing_a", "swing_b", "swing_c",
                "fib_a_bar", "fib_b_bar", "fib_c_bar", "fib_05", "fib_0618", "fib_0786",
                "fib_1272", "fib_1414", "fib_1618", "fib_2000", "entry_box_low",
                "entry_box_high", "target_box_low", "target_box_high", "setup_direction"):
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


def test_long_and_short_setups_are_supported_and_drawings_are_ordered():
    bars = _sample_bars(900)
    out = generate_signals(bars, {"allow_shorts": True})
    assert set(out["signal"].unique()).issubset({-1, 0, 1})
    active = out.dropna(subset=["entry_box_low", "entry_box_high"])
    if not active.empty:
        assert (active["entry_box_low"] <= active["entry_box_high"]).all()
        assert (active["target_box_low"] <= active["target_box_high"]).all()
        assert active["fib_a_bar"].notna().all()
        assert active["fib_b_bar"].notna().all()
        assert active["fib_c_bar"].notna().all()


def test_shorts_can_be_disabled():
    bars = _sample_bars(420)
    out = generate_signals(bars, {"allow_shorts": False})
    assert not (out["signal"] < 0).any()


def test_shallow_golden_zone_preset_accepts_half_to_618():
    bars = _sample_bars(900)
    out = generate_signals(bars, {"golden_shallow": 0.5, "golden_deep": 0.618})
    assert "fib_zone_shallow" in out.columns
    assert "fib_zone_deep" in out.columns
    active = out.dropna(subset=["entry_box_low", "entry_box_high"])
    if not active.empty:
        assert (active["fib_zone_shallow"] == 0.5).all()
        assert (active["fib_zone_deep"] == 0.618).all()
        assert (active["entry_box_low"] <= active["entry_box_high"]).all()


def test_golden_zone_rejects_reversed_levels():
    bars = _sample_bars(120)
    import pytest
    with pytest.raises(ValueError, match="Golden levels"):
        generate_signals(bars, {"golden_shallow": 0.618, "golden_deep": 0.5})


def test_strict_confirmation_does_not_break_short_side():
    bars = _sample_bars(900)
    out = generate_signals(bars, {"strict_confirmation": True, "allow_shorts": True})
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_optional_15m_trend_filter_is_causal_and_valid():
    bars = _sample_bars(900)
    out = generate_signals(bars, {"trend_filter": "shorts_only"})
    assert set(out["signal"].unique()).issubset({-1, 0, 1})
    with pytest.raises(ValueError, match="trend_filter"):
        generate_signals(bars, {"trend_filter": "future_data"})


def test_last_confirmed_abc_remains_visible_after_invalidation():
    bars = _sample_bars(900)
    out = generate_signals(bars)
    assert "setup_status" in out.columns
    invalidated = out["setup_status"] == -1
    assert invalidated.any()
    assert out.loc[invalidated, ["fib_a_bar", "fib_b_bar", "fib_c_bar"]].notna().all().all()
    assert out.loc[invalidated, ["swing_a", "swing_b", "swing_c"]].notna().all().all()


def test_15m_trend_filter_does_not_use_incomplete_future_15m_bar():
    bars = _sample_bars(900)
    full = generate_signals(bars, {"trend_filter": "both"})
    for end in (80, 110, 145, 179, 320):
        prefix = generate_signals(bars.iloc[:end+1], {"trend_filter": "both"})
        pd.testing.assert_series_equal(
            full["signal"].iloc[:end+1].reset_index(drop=True),
            prefix["signal"].reset_index(drop=True),
        )
