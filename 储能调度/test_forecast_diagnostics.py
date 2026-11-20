from __future__ import annotations

import unittest

import pandas as pd

from forecast_diagnostics import forecast_diagnostics


class ForecastDiagnosticsTest(unittest.TestCase):
    def test_bias_and_mae(self) -> None:
        frame = pd.DataFrame(
            {
                "hour": [0, 0, 1],
                "actual": [1.0, 2.0, 3.0],
                "lightgbm": [0.5, 2.0, 4.0],
                "seasonal_naive_daily": [1.5, 2.0, 2.0],
                "actual_pv_kwh": [0.0, 0.0, 0.5],
                "forecast_pv_kwh": [0.0, 0.0, 0.3],
            }
        )
        hourly, summary = forecast_diagnostics(frame)
        self.assertAlmostEqual(summary["load"]["mean_bias_kwh"], 0.1666666667)
        self.assertAlmostEqual(summary["load"]["mae_kwh"], 0.5)
        self.assertEqual(len(hourly), 2)
        self.assertAlmostEqual(summary["pv"]["mean_bias_kwh"], -0.0666666667)


if __name__ == "__main__":
    unittest.main()
