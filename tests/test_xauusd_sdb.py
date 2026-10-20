import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.xauusd_sdb import _resample, build_sdb_signals, load_gold_data


def sample_bars(periods=1152):
    idx = pd.date_range("2026-01-05 00:00:00", periods=periods, freq="5min")
    x = np.arange(periods, dtype=float)
    close = 2350 + 0.012 * x + 2.2 * np.sin(x / 13.0) + 0.7 * np.sin(x / 3.7)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.25 + 0.15 * (1 + np.sin(x / 5))
    low = np.minimum(open_, close) - 0.25 - 0.15 * (1 + np.cos(x / 7))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": np.full(periods, 100.0)}, index=idx)


class SDBTests(unittest.TestCase):
    def test_resampled_bar_is_timestamped_at_close(self):
        bars = sample_bars(12)
        m15 = _resample(bars, "15min")
        self.assertEqual(m15.index[0], bars.index[0] + pd.Timedelta(minutes=15))
        self.assertEqual(m15.index[-1], bars.index[-1] + pd.Timedelta(minutes=5))
        self.assertAlmostEqual(float(m15.iloc[0]["Close"]), float(bars.iloc[2]["Close"]))

    def test_signal_engine_returns_expected_columns_without_lookahead_timestamps(self):
        bars = sample_bars()
        result = build_sdb_signals(bars)
        required = {"signal", "stop", "sd1", "sd2", "sd25", "sd4", "bias", "setup", "sweep_time"}
        self.assertTrue(required.issubset(set(result.columns)))
        self.assertTrue(set(result["signal"].dropna().astype(int).unique()).issubset({-1, 0, 1}))
        signals = result[result["signal"] != 0]
        for timestamp, row in signals.iterrows():
            if row["sweep_time"]:
                self.assertGreaterEqual(timestamp, pd.Timestamp(row["sweep_time"]))

    def test_insufficient_higher_timeframe_history_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            build_sdb_signals(sample_bars(120))

    def test_live_data_failure_never_falls_back_to_synthetic(self):
        fake_yf = types.SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame())
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            with self.assertRaisesRegex(RuntimeError, "No synthetic fallback"):
                load_gold_data()


if __name__ == "__main__":
    unittest.main()
