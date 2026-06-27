"""Unit tests for core.metrics. Run: pytest -q"""
from core import metrics


def approx(a, b, tol=1e-4):
    return a is not None and abs(a - b) <= tol


def test_mean_and_stdev():
    assert approx(metrics.mean([1, 2, 3, 4]), 2.5)
    assert approx(metrics.stdev([2, 4, 4, 4, 5, 5, 7, 9]), 2.13809, tol=1e-4)
    assert metrics.stdev([5]) is None
    assert metrics.mean([]) is None


def test_mean_ignores_none_and_nan():
    assert approx(metrics.mean([1, None, 3, float("nan")]), 2.0)


def test_pct_returns():
    r = metrics.pct_returns([100, 110, 99])
    assert approx(r[0], 0.10)
    assert approx(r[1], -0.10)
    assert metrics.pct_returns([100]) == []


def test_sharpe_basic():
    s = metrics.sharpe_ratio([0.01, 0.02, -0.01, 0.03, 0.0])
    assert approx(s, 0.6325, tol=1e-3)


def test_sharpe_zero_volatility_is_none():
    assert metrics.sharpe_ratio([0.01, 0.01, 0.01]) is None
    assert metrics.sharpe_ratio([0.01]) is None


def test_sortino_only_penalises_downside():
    s = metrics.sortino_ratio([0.02, 0.03, -0.01, 0.04])
    assert s is not None and s > 0


def test_sortino_no_downside_is_none():
    assert metrics.sortino_ratio([0.01, 0.02, 0.03]) is None


def test_equity_curve_from_pnl():
    assert metrics.equity_curve_from_pnl([10, -5, 20], start_equity=100) == [110, 105, 125]


def test_max_drawdown():
    eq = [100, 110, 125, 115, 105, 130]
    dd = metrics.max_drawdown(eq)
    assert approx(dd["abs"], -20.0, tol=1e-6)
    assert approx(dd["pct"], -16.0, tol=1e-6)


def test_max_drawdown_monotonic_up_is_zero():
    dd = metrics.max_drawdown([100, 101, 102, 103])
    assert dd["abs"] == 0.0 and dd["pct"] == 0.0


def test_current_drawdown():
    cur = metrics.current_drawdown([100, 125, 110, 105])
    assert approx(cur["abs"], -20.0, tol=1e-6)
    assert approx(cur["pct"], -16.0, tol=1e-6)
    cur2 = metrics.current_drawdown([100, 110, 130])
    assert cur2["abs"] == 0.0


def test_exposure_long_and_short():
    pos = [{"market_value": 10000}, {"market_value": 5000}, {"market_value": -4000}]
    e = metrics.exposure(pos)
    assert e["long"] == 15000.0
    assert e["short"] == 4000.0
    assert e["gross"] == 19000.0
    assert e["net"] == 11000.0


def test_leverage():
    assert approx(metrics.leverage(19000, 10000), 1.9)
    assert metrics.leverage(19000, 0) is None


def test_hhi_equal_weighted():
    assert approx(metrics.concentration_hhi([25, 25, 25, 25]), 0.25)


def test_hhi_single_name():
    assert approx(metrics.concentration_hhi([100]), 1.0)
    assert metrics.concentration_hhi([0, 0]) is None


def test_normalized_concentration():
    assert approx(metrics.concentration_normalized([25, 25, 25, 25]), 0.0)
    assert approx(metrics.concentration_normalized([100]), 1.0)


def test_beta_perfectly_correlated_2x():
    bench = [0.01, -0.02, 0.03, -0.01, 0.02]
    asset = [2 * x for x in bench]
    assert approx(metrics.beta(asset, bench), 2.0, tol=1e-6)


def test_beta_zero_benchmark_variance_is_none():
    assert metrics.beta([0.01, 0.02, 0.03], [0.01, 0.01, 0.01]) is None


def test_alpha_when_asset_equals_benchmark_is_zero():
    bench = [0.01, -0.02, 0.03, -0.01, 0.02]
    assert approx(metrics.alpha(bench, bench), 0.0, tol=1e-9)
