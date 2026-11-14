"""Deterministic tests for portfolio terminal daily-return calculations."""
from research.portfolio_terminal_routes import _price_metrics, STOCK_UNIVERSE, SECTOR_PROXIES


def test_five_session_return_uses_six_closes():
    closes = [100, 101, 102, 103, 104, 110]
    bars = [
        {"t": f"2026-11-{i:02d}T21:00:00Z", "c": value, "v": 1000}
        for i, value in enumerate(closes, start=1)
    ]
    result = _price_metrics("TEST", bars)
    assert result["day_pct"] == 5.77
    assert result["five_day_pct"] == 10.0
    assert result["session"] == "2026-11-06"


def test_five_day_return_is_null_without_six_sessions():
    bars = [
        {"t": f"2026-11-{i:02d}T21:00:00Z", "c": 100 + i, "v": 500}
        for i in range(1, 5)
    ]
    result = _price_metrics("TEST", bars)
    assert result["five_day_pct"] is None
    assert result["day_pct"] > 0


def test_invalid_prices_are_ignored_and_rows_sorted():
    bars = [
        {"t": "2026-11-03T21:00:00Z", "c": 102, "v": 10},
        {"t": "2026-11-01T21:00:00Z", "c": 100, "v": 10},
        {"t": "2026-11-02T21:00:00Z", "c": 0, "v": 10},
    ]
    result = _price_metrics("TEST", bars)
    assert result["price"] == 102
    assert result["day_pct"] == 2.0


def test_tracked_universe_has_no_duplicate_symbols():
    assert len(STOCK_UNIVERSE) == len(set(STOCK_UNIVERSE))
    assert "Semiconductors" in SECTOR_PROXIES
