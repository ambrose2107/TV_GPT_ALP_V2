"""
data_loader.py
Loads XAUUSD (gold) OHLCV plus DXY and Silver for the intermarket
correlation confluences, with a synthetic fallback for offline development.
"""
import numpy as np
import pandas as pd

DEFAULT_PERIOD_BY_INTERVAL = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d",
    "30m": "60d", "1h": "730d", "1d": "10y",
}

DEFAULT_SYMBOLS = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "silver": "SI=F",
}


def _fetch_one(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} @ {interval} (period={period}).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def get_live_data(interval: str = "5m", period: str = None,
                   symbols: dict = None) -> dict:
    """Returns {'gold': df, 'dxy': df, 'silver': df} aligned to gold's index."""
    symbols = symbols or DEFAULT_SYMBOLS
    period = period or DEFAULT_PERIOD_BY_INTERVAL.get(interval, "60d")

    gold = _fetch_one(symbols["gold"], interval, period)
    out = {"gold": gold}
    for key in ("dxy", "silver"):
        try:
            df = _fetch_one(symbols[key], interval, period)
            out[key] = df.reindex(gold.index, method="ffill")
        except Exception as e:
            print(f"[data_loader] Could not fetch {key} ({symbols[key]}): {e}. "
                  f"Correlation confluence for {key} will be disabled.")
            out[key] = None
    return out


def get_synthetic_data(n_bars: int = 20000, bar_seconds: int = 300,
                        start_price: float = 2350.0, seed: int = 42) -> dict:
    """
    Synthetic gold + correlated DXY (inverse) + correlated Silver (same
    direction, noisier) series, for offline pipeline validation only.
    NOT real market data.
    """
    rng = np.random.default_rng(seed)
    n_regimes = max(6, n_bars // 400)
    regime_len = n_bars // n_regimes
    drift = np.zeros(n_bars)
    idx = 0
    for r in range(n_regimes):
        length = regime_len if r < n_regimes - 1 else n_bars - idx
        if rng.random() < 0.4:
            strength = rng.choice([-1, 1]) * rng.uniform(0.00003, 0.00013)
        else:
            strength = rng.uniform(-0.00002, 0.00002)
        drift[idx: idx + length] = strength
        idx += length

    vol = np.zeros(n_bars)
    vol[0] = 0.0006
    omega, alpha, beta = 1e-7, 0.12, 0.85
    shocks = rng.standard_normal(n_bars)
    for t in range(1, n_bars):
        vol[t] = np.sqrt(max(omega + alpha * (shocks[t-1]*vol[t-1])**2 + beta*vol[t-1]**2, 1e-8))

    bars_per_day = max(1, int(86400 / bar_seconds))
    session = 0.6 + 0.8 * (0.5 + 0.5*np.sin(2*np.pi*(np.arange(n_bars) % bars_per_day)/bars_per_day - np.pi/2))

    gold_log_ret = drift + vol * session * shocks
    gold_close = np.exp(np.log(start_price) + np.cumsum(gold_log_ret))

    def _build_ohlcv(close, vol_local, session_local, base_volume=(500, 5000)):
        n = len(close)
        open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
        intrabar = np.abs(rng.standard_normal(n)) * vol_local * close * 3.0 + 0.02
        high = np.maximum(open_, close) + intrabar * rng.uniform(0.2, 0.6, n)
        low = np.minimum(open_, close) - intrabar * rng.uniform(0.2, 0.6, n)
        volume = (rng.uniform(*base_volume, n) * (1 + session_local)).astype(int)
        return open_, high, low, close, volume

    start_time = pd.Timestamp.now().floor("D") - pd.Timedelta(days=60)
    idx_time = pd.date_range(start=start_time, periods=n_bars, freq=f"{bar_seconds}s")

    o, h, l, c, v = _build_ohlcv(gold_close, vol, session)
    gold_df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx_time)

    # DXY: inversely correlated to gold with its own noise
    dxy_noise = rng.standard_normal(n_bars) * 0.0003
    dxy_log_ret = -0.6 * gold_log_ret + dxy_noise
    dxy_close = np.exp(np.log(100.0) + np.cumsum(dxy_log_ret))
    o, h, l, c, v = _build_ohlcv(dxy_close, vol * 0.5, session, base_volume=(100, 1000))
    dxy_df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx_time)

    # Silver: positively correlated to gold, noisier
    silver_noise = rng.standard_normal(n_bars) * 0.0009
    silver_log_ret = 0.7 * gold_log_ret + silver_noise
    silver_close = np.exp(np.log(28.0) + np.cumsum(silver_log_ret))
    o, h, l, c, v = _build_ohlcv(silver_close, vol * 1.2, session, base_volume=(300, 3000))
    silver_df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx_time)

    return {"gold": gold_df, "dxy": dxy_df, "silver": silver_df}


def get_data(interval: str = "5m", use_live: bool = True, period: str = None,
             n_bars: int = 20000, symbols: dict = None, primary_symbol: str = None) -> dict:
    if primary_symbol:
        symbols = dict(symbols or DEFAULT_SYMBOLS)
        symbols["gold"] = str(primary_symbol).upper()
    if use_live:
        try:
            return get_live_data(interval=interval, period=period, symbols=symbols)
        except Exception as e:
            print(f"[data_loader] Live fetch failed ({e}); using synthetic data instead "
                  f"(clearly not real market data - for pipeline validation only).")
    bar_seconds = {"1m": 60, "2m": 120, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}.get(interval, 300)
    return get_synthetic_data(n_bars=n_bars, bar_seconds=bar_seconds)
