from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lightgbm_walk_forward import (
    FEATURE_COLUMNS,
    build_feature_frame,
    walk_forward_folds,
)


class LightGBMWalkForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        timestamps = pd.date_range(
            "2024-01-01 00:00",
            periods=60 * 48,
            freq="30min",
        )
        slot = np.arange(len(timestamps)) % 48
        weekday = timestamps.dayofweek
        cls.clean = pd.DataFrame(
            {
                "Customer": 1,
                "source_year": "2024",
                "source_date": timestamps.normalize(),
                "interval_start": timestamps,
                "slot_index": slot,
                "total_load_kwh": (
                    0.3
                    + 0.02 * weekday
                    + 0.4 * (slot / 48)
                ),
            }
        )

    def test_feature_frame_has_no_short_horizon_lags(self) -> None:
        features = build_feature_frame(self.clean)
        for forbidden in ("lag_1", "lag_2", "lag_47"):
            self.assertNotIn(forbidden, features.columns)
        for column in FEATURE_COLUMNS:
            self.assertIn(column, features.columns)
        self.assertTrue(features["lag_48"].notna().sum() > 0)

    def test_walk_forward_has_three_non_overlapping_twenty_day_folds(self) -> None:
        features = build_feature_frame(self.clean)
        folds = walk_forward_folds(features)
        self.assertEqual(len(folds), 3)
        self.assertEqual(
            (folds[0]["test_end"].normalize() - folds[0]["test_start"]).days,
            19,
        )
        self.assertEqual(
            folds[1]["test_start"],
            folds[0]["test_end"].normalize() + pd.Timedelta(days=1),
        )


if __name__ == "__main__":
    unittest.main()
