"""
data_loader.py
Loads 5-minute XAUUSD data and derives the 15-minute timeframe by
RESAMPLING the 5m bars (3 bars -> 1), so the two timeframes are always
perfectly consistent (the 15m bar's High/Low genuinely contains its three
5m bars) - critical for multi-timeframe logic to be meaningful at all.
"""
import numpy as np
import pandas as pd

DEFAULT_PERIOD_5M = "60d"


def get_live_5m(symbol: str = "GC=F", period: str = DEFAULT_PERIOD_5M) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval="5m", progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No 5m data for {symbol} (period={period}).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def resample_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    df_15m = df_5m.resample("15min").agg(agg).dropna()
    return df_15m


def get_synthetic_5m(n_bars: int = 20000, start_price: float = 2350.0, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic gold-like 5-minute series with regime-switching drift and
    GARCH-like volatility clustering, structured to actually produce clean
    impulsive legs periodically (not pure noise) so multi-timeframe
    structure/impulse detection has something real to find. NOT real
    market data - for pipeline validation only.
    """
    rng = np.random.default_rng(seed)
    n_regimes = max(8, n_bars // 250)
    regime_len = n_bars // n_regimes
    drift = np.zeros(n_bars)
    regime_type = []
    idx = 0
    for r in range(n_regimes):
        length = regime_len if r < n_regimes - 1 else n_bars - idx
        roll = rng.random()
        if roll < 0.45:
            # impulsive trending regime - stronger, more persistent drift
            strength = rng.choice([-1, 1]) * rng.uniform(0.00008, 0.00022)
            regime_type.append("impulse")
        else:
            strength = rng.uniform(-0.00002, 0.00002)
            regime_type.append("chop")
        drift[idx: idx + length] = strength
        idx += length

    vol = np.zeros(n_bars)
    vol[0] = 0.0005
    omega, alpha, beta = 1e-7, 0.1, 0.87
    shocks = rng.standard_normal(n_bars)
    for t in range(1, n_bars):
        vol[t] = np.sqrt(max(omega + alpha * (shocks[t-1]*vol[t-1])**2 + beta*vol[t-1]**2, 1e-8))

    bars_per_day = 288  # 24h at 5min
    session = 0.6 + 0.8 * (0.5 + 0.5*np.sin(2*np.pi*(np.arange(n_bars) % bars_per_day)/bars_per_day - np.pi/2))

    log_ret = drift + vol * session * shocks * 0.7  # dampen pure noise a bit vs drift
    close = np.exp(np.log(start_price) + np.cumsum(log_ret))

    open_ = np.empty(n_bars); open_[0] = start_price; open_[1:] = close[:-1]
    intrabar = np.abs(rng.standard_normal(n_bars)) * vol * close * 2.5 + 0.02
    high = np.maximum(open_, close) + intrabar * rng.uniform(0.15, 0.5, n_bars)
    low = np.minimum(open_, close) - intrabar * rng.uniform(0.15, 0.5, n_bars)
    volume = (rng.uniform(500, 5000, n_bars) * (1 + session)).astype(int)

    start_time = pd.Timestamp.now().floor("D") - pd.Timedelta(days=90)
    idx_time = pd.date_range(start=start_time, periods=n_bars, freq="5min")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx_time)


def get_mtf_data(use_live: bool = True, symbol: str = "GC=F", n_bars: int = 20000) -> dict:
    """Returns {'m5': df_5m, 'm15': df_15m} with m15 derived by resampling m5."""
    if use_live:
        try:
            df_5m = get_live_5m(symbol)
            return {"m5": df_5m, "m15": resample_to_15m(df_5m)}
        except Exception as e:
            print(f"[data_loader] Live fetch failed ({e}); using synthetic data "
                  f"(clearly not real market data - for pipeline validation only).")
    df_5m = get_synthetic_5m(n_bars=n_bars)
    return {"m5": df_5m, "m15": resample_to_15m(df_5m)}
