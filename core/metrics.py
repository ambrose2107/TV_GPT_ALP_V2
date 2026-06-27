"""
core/metrics.py - Institutional-grade quantitative metrics.

Pure functions only (stdlib math), so they are trivial to unit test and have
no I/O or framework dependencies. All functions are defensive against empty
inputs, zero variance, and divide-by-zero, returning None where a metric is
undefined rather than raising.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

TRADING_DAYS = 252


def _clean(xs: Sequence[Optional[float]]) -> list[float]:
    out: list[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = _clean(xs)
    return sum(vals) / len(vals) if vals else None


def stdev(xs: Sequence[Optional[float]], sample: bool = True) -> Optional[float]:
    vals = _clean(xs)
    n = len(vals)
    if n < 2:
        return None
    m = sum(vals) / n
    denom = (n - 1) if sample else n
    var = sum((v - m) ** 2 for v in vals) / denom
    return math.sqrt(var)


def pct_returns(prices: Sequence[Optional[float]]) -> list[float]:
    vals = _clean(prices)
    rets: list[float] = []
    for i in range(1, len(vals)):
        prev = vals[i - 1]
        if prev:
            rets.append((vals[i] - prev) / prev)
    return rets


def sharpe_ratio(returns: Sequence[Optional[float]], risk_free: float = 0.0,
                 annualise: bool = False) -> Optional[float]:
    vals = _clean(returns)
    if len(vals) < 2:
        return None
    excess = [v - risk_free for v in vals]
    m = sum(excess) / len(excess)
    sd = stdev(excess)
    if not sd:
        return None
    sharpe = m / sd
    if annualise:
        sharpe *= math.sqrt(TRADING_DAYS)
    return round(sharpe, 4)


def sortino_ratio(returns: Sequence[Optional[float]], risk_free: float = 0.0,
                  annualise: bool = False) -> Optional[float]:
    vals = _clean(returns)
    if len(vals) < 2:
        return None
    excess = [v - risk_free for v in vals]
    m = sum(excess) / len(excess)
    downside = [e for e in excess if e < 0]
    if not downside:
        return None
    dd = math.sqrt(sum(e ** 2 for e in downside) / len(excess))
    if not dd:
        return None
    sortino = m / dd
    if annualise:
        sortino *= math.sqrt(TRADING_DAYS)
    return round(sortino, 4)


def equity_curve_from_pnl(pnls: Sequence[Optional[float]], start_equity: float = 0.0) -> list[float]:
    eq = start_equity
    curve: list[float] = []
    for p in pnls:
        try:
            eq += float(p or 0)
        except (TypeError, ValueError):
            pass
        curve.append(round(eq, 4))
    return curve


def max_drawdown(equity: Sequence[Optional[float]]) -> dict:
    vals = _clean(equity)
    if not vals:
        return {"abs": 0.0, "pct": 0.0}
    peak = vals[0]
    worst_abs = 0.0
    worst_pct = 0.0
    for v in vals:
        if v > peak:
            peak = v
        drop = v - peak
        if drop < worst_abs:
            worst_abs = drop
        if peak != 0:
            pct = drop / abs(peak) * 100
            if pct < worst_pct:
                worst_pct = pct
    return {"abs": round(worst_abs, 2), "pct": round(worst_pct, 2)}


def current_drawdown(equity: Sequence[Optional[float]]) -> dict:
    vals = _clean(equity)
    if not vals:
        return {"abs": 0.0, "pct": 0.0}
    running_peak = vals[0]
    for v in vals:
        if v > running_peak:
            running_peak = v
    last = vals[-1]
    abs_dd = last - running_peak
    pct_dd = (abs_dd / abs(running_peak) * 100) if running_peak else 0.0
    return {"abs": round(abs_dd, 2), "pct": round(pct_dd, 2)}


def exposure(positions: Sequence[dict]) -> dict:
    long_mv = 0.0
    short_mv = 0.0
    for p in positions:
        try:
            mv = float(p.get("market_value") or 0)
        except (TypeError, ValueError):
            continue
        if mv >= 0:
            long_mv += mv
        else:
            short_mv += abs(mv)
    return {
        "long": round(long_mv, 2),
        "short": round(short_mv, 2),
        "gross": round(long_mv + short_mv, 2),
        "net": round(long_mv - short_mv, 2),
    }


def leverage(gross_exposure: float, equity: float) -> Optional[float]:
    if not equity:
        return None
    return round(gross_exposure / equity, 4)


def concentration_hhi(weights_or_values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [abs(v) for v in _clean(weights_or_values) if v is not None]
    total = sum(vals)
    if total <= 0:
        return None
    weights = [v / total for v in vals]
    return round(sum(w ** 2 for w in weights), 4)


def concentration_normalized(weights_or_values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [abs(v) for v in _clean(weights_or_values) if v is not None]
    n = len(vals)
    if n <= 1:
        return 1.0 if n == 1 else None
    hhi = concentration_hhi(vals)
    if hhi is None:
        return None
    floor = 1.0 / n
    return round((hhi - floor) / (1 - floor), 4)


def beta(asset_returns: Sequence[float], benchmark_returns: Sequence[float]) -> Optional[float]:
    a = _clean(asset_returns)
    b = _clean(benchmark_returns)
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[-n:], b[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
    var_b = sum((x - mb) ** 2 for x in b) / (n - 1)
    if not var_b:
        return None
    return round(cov / var_b, 4)


def alpha(asset_returns: Sequence[float], benchmark_returns: Sequence[float],
          risk_free: float = 0.0) -> Optional[float]:
    a = _clean(asset_returns)
    b = _clean(benchmark_returns)
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[-n:], b[-n:]
    be = beta(a, b)
    if be is None:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    return round(ma - (risk_free + be * (mb - risk_free)), 6)
