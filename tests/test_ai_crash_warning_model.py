"""Unit tests for coverage-aware AI crash-warning scoring."""
import unittest

import numpy as np
import pandas as pd

from research.ai_crash_warning import SCORE_WEIGHTS, _score


class CrashWarningScoringTests(unittest.TestCase):
    def setUp(self):
        self.market_dates = pd.bdate_range("2023-01-02", periods=320)
        trend = np.linspace(100.0, 145.0, len(self.market_dates))
        self.px = pd.DataFrame({
            "SPY": trend,
            "QQQ": trend * np.linspace(1.0, 1.08, len(trend)),
            "RSP": trend * np.linspace(0.99, 1.01, len(trend)),
            "IWM": trend * np.linspace(0.98, 1.00, len(trend)),
            "SOXX": trend * np.linspace(1.01, 1.14, len(trend)),
            "^VIX": 18.0 + 4.0 * np.sin(np.linspace(0.0, 15.0, len(trend))),
        }, index=self.market_dates)

        weekly = pd.date_range("2020-01-03", periods=170, freq="W-FRI")
        monthly = pd.date_range("2018-01-01", periods=70, freq="MS")
        self.fred = {
            "hy_oas": pd.Series(np.linspace(3.0, 5.0, len(weekly)), index=weekly),
            "nfci": pd.Series(0.1 * np.sin(np.linspace(0.0, 12.0, len(weekly))), index=weekly),
            "dfii10": pd.Series(np.linspace(0.5, 2.4, len(self.market_dates)), index=self.market_dates),
            "unrate": pd.Series(4.0 + 0.4 * np.sin(np.linspace(0.0, 8.0, len(monthly))), index=monthly),
        }
        self.fund = {
            "capex_growth_gap": 12.0,
            "debt_growth_gap": 8.0,
            "capex_revenue": 18.0,
        }

    def test_configured_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(SCORE_WEIGHTS.values()), 1.0, places=8)

    def test_complete_inputs_cover_all_configured_weight(self):
        result = _score(self.px, self.fred, self.fund)
        self.assertAlmostEqual(result["score_coverage_pct"], 100.0, places=6)
        self.assertEqual(result["missing_components"], [])
        self.assertTrue(np.isfinite(result["score"]))
        self.assertEqual(set(result["available_components"]), set(SCORE_WEIGHTS))

    def test_missing_inputs_are_excluded_not_counted_as_neutral(self):
        partial_fred = {
            "dfii10": self.fred["dfii10"],
            "unrate": self.fred["unrate"],
        }
        result = _score(self.px, partial_fred, {})
        self.assertAlmostEqual(result["score_coverage_pct"], 38.0, places=6)
        self.assertIn("Credit", result["missing_components"])
        self.assertIn("Liquidity", result["missing_components"])
        self.assertIn("AI Fundamental", result["missing_components"])
        self.assertIn("AI Financing", result["missing_components"])
        self.assertNotIn("AI Fundamental", result["available_components"])
        self.assertTrue(np.isfinite(result["score"]))

    def test_score_renormalizes_over_available_components(self):
        result = _score(self.px, {"dfii10": self.fred["dfii10"]}, {})
        available = result["available_components"]
        denominator = sum(SCORE_WEIGHTS[key] for key in available)
        expected = sum(result["components"][key] * SCORE_WEIGHTS[key] for key in available) / denominator
        self.assertAlmostEqual(result["score"], expected, places=1)


if __name__ == "__main__":
    unittest.main()
