import sys
import types
import unittest
from unittest.mock import patch

from flask import Flask
from research.xauusd_sdb_routes import xauusd_sdb_bp
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

    def test_sdb_api_rejects_unauthenticated_requests(self):
        app = Flask(__name__)
        app.secret_key = "unit-test-only"
        app.register_blueprint(xauusd_sdb_bp)
        response = app.test_client().post("/api/xauusd-sdb/run", json={})
        self.assertEqual(response.status_code, 401)

    def test_missing_alpaca_credentials_fails_clearly(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Alpaca credentials are missing"):
                load_gold_data()

    def test_live_data_failure_never_falls_back_to_synthetic(self):
        response = types.SimpleNamespace(
            status_code=200,
            json=lambda: {"bars": [], "next_page_token": None},
            raise_for_status=lambda: None,
        )
        with patch.dict("os.environ", {"APCA_API_KEY_ID": "test", "APCA_API_SECRET_KEY": "test"}, clear=True):
            with patch("requests.get", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "No synthetic fallback"):
                    load_gold_data()

    def test_alpaca_gld_bars_are_normalized(self):
        times = pd.date_range("2026-01-05", periods=600, freq="5min", tz="UTC")
        bars = [{
            "t": ts.isoformat().replace("+00:00", "Z"),
            "o": 400.0, "h": 401.0, "l": 399.0, "c": 400.5, "v": 1000
        } for ts in times]
        response = types.SimpleNamespace(
            status_code=200,
            json=lambda: {"bars": bars, "next_page_token": None},
            raise_for_status=lambda: None,
        )
        with patch.dict("os.environ", {"APCA_API_KEY_ID": "test", "APCA_API_SECRET_KEY": "test"}, clear=True):
            with patch("requests.get", return_value=response) as get:
                result = load_gold_data()
        self.assertEqual(len(result), 600)
        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertEqual(float(result.iloc[0]["Close"]), 400.5)
        self.assertIn("/v2/stocks/GLD/bars", get.call_args.args[0])
        self.assertEqual(get.call_args.kwargs["params"]["timeframe"], "5Min")


if __name__ == "__main__":
    unittest.main()
