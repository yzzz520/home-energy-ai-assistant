from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from seasonal_naive import (
    aggregate_forecast_metrics,
    evaluate_forecasts,
    make_dayahead_forecasts,
)


class SeasonalNaiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        timestamps = pd.date_range(
            "2024-01-01 00:00",
            periods=20 * 48,
            freq="30min",
        )
        slot = np.arange(len(timestamps)) % 48
        weekday = timestamps.dayofweek
        values = 1.0 + slot / 48 + weekday * 0.1
        cls.series = pd.DataFrame(
            {
                "interval_start": timestamps,
                "total_load_kwh": values,
            }
        )

    def test_dayahead_forecasts_have_48_steps(self) -> None:
        forecasts = make_dayahead_forecasts(
            self.series,
            test_start="2024-01-15",
        )
        self.assertEqual(
            forecasts.groupby("origin").size().unique().tolist(),
            [48],
        )
        self.assertEqual(len(forecasts), 6 * 48)

    def test_weekly_seasonal_naive_uses_previous_week(self) -> None:
        forecasts = make_dayahead_forecasts(
            self.series,
            test_start="2024-01-15",
        )
        first = forecasts.iloc[0]
        source_time = (
            first["forecast_timestamp"] - pd.Timedelta(days=7)
        )
        expected = self.series.set_index("interval_start").loc[
            source_time, "total_load_kwh"
        ]
        self.assertAlmostEqual(
            first["seasonal_naive_weekly"],
            expected,
        )

    def test_metrics_are_zero_for_exact_weekly_pattern(self) -> None:
        forecasts = make_dayahead_forecasts(
            self.series,
            test_start="2024-01-15",
        )
        forecasts["seasonal_naive_weekly"] = forecasts["actual"]
        metrics = evaluate_forecasts(
            forecasts,
            forecast_columns=("seasonal_naive_weekly",),
        )
        self.assertAlmostEqual(
            metrics["seasonal_naive_weekly"]["mae_kwh"],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["seasonal_naive_weekly"]["wape_pct"],
            0.0,
        )

    def test_global_aggregation_counts_every_point(self) -> None:
        forecasts = make_dayahead_forecasts(
            self.series,
            test_start="2024-01-15",
        )
        forecasts["seasonal_naive_daily"] = forecasts["actual"] + 0.1
        aggregated = aggregate_forecast_metrics(
            forecasts,
            forecast_columns=("seasonal_naive_daily",),
        )["seasonal_naive_daily"]

        self.assertEqual(aggregated["rows"], len(forecasts))
        self.assertAlmostEqual(
            aggregated["sum_abs_error"],
            0.1 * len(forecasts),
        )


if __name__ == "__main__":
    unittest.main()
