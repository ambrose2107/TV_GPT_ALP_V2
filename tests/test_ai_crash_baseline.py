"""Deterministic checks for the checked-in AI-crash historical baseline.

These tests use only the Python standard library so they run before the live
Playwright test and do not depend on Yahoo/Alpaca availability.
"""
import csv
from datetime import date
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[1] / "research" / "data" / "ai_crash_validation_history.csv"


def load_series():
    with BASELINE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "Historical validation CSV is empty"
    dates = [date.fromisoformat(row["date"]) for row in rows]
    assert dates == sorted(dates), "Historical baseline dates must be sorted"
    assert len(dates) == len(set(dates)), "Historical baseline contains duplicate dates"
    return rows, dates


def series_for(rows, symbol):
    column = {"SPY": "SPY_close", "QQQ": "QQQ_adj_close", "^VIX": "VIX_close"}[symbol]
    return [(date.fromisoformat(row["date"]), float(row[column]))
            for row in rows if row.get(column, "").strip()]


def test_checked_in_baseline_has_long_history_for_all_required_series():
    rows, dates = load_series()
    assert len(rows) >= 9000, f"Expected multi-decade daily baseline, found {len(rows)} rows"
    assert dates[0] <= date(1990, 1, 5), f"VIX baseline starts too late: {dates[0]}"
    expected = {"SPY": date(2000, 6, 1), "QQQ": date(2000, 6, 1), "^VIX": date(1990, 6, 1)}
    for symbol, latest_allowed_start in expected.items():
        values = series_for(rows, symbol)
        minimum = 9000 if symbol == "^VIX" else 5000
        assert len(values) >= minimum, (
            f"{symbol} baseline has too few non-empty observations: {len(values)}"
        )
        assert values[0][0] <= latest_allowed_start, (
            f"{symbol} baseline starts too late: {values[0][0]}"
        )
        # This committed baseline intentionally retains a durable historical core.
        # SPY/QQQ tails are refreshed from live providers at runtime; the browser
        # smoke test separately asserts that the merged replay is current.
        minimum_tail = {
            "SPY": date(2025, 8, 1),
            "QQQ": date(2024, 1, 1),
            "^VIX": date(2026, 9, 1),
        }[symbol]
        assert values[-1][0] >= minimum_tail, (
            f"{symbol} committed baseline tail is unexpectedly old: {values[-1][0]}"
        )


def test_all_four_validation_episodes_have_a_20_percent_breach_in_one_year():
    rows, _ = load_series()
    cases = [
        ("Dot-com bust", "QQQ", date(2000, 3, 10)),
        ("Global financial crisis", "SPY", date(2007, 10, 9)),
        ("COVID shock", "SPY", date(2020, 2, 19)),
        ("2022 bear market", "SPY", date(2022, 1, 3)),
    ]
    for name, symbol, peak_date in cases:
        values = series_for(rows, symbol)
        nearest = min(range(len(values)), key=lambda i: abs((values[i][0] - peak_date).days))
        observed_peak_date, peak_price = values[nearest]
        assert abs((observed_peak_date - peak_date).days) <= 4, (
            f"{name}: could not locate peak date near {peak_date}"
        )
        forward = values[nearest + 1: nearest + 253]
        breach = next((d for d, price in forward if price <= peak_price * 0.80), None)
        assert breach is not None, f"{name}: no 20% drawdown found within 252 trading observations"
