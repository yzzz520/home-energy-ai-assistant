from __future__ import annotations

import unittest

import pandas as pd

from reader import AusgridReader
from time_axis import build_sample


class TimeAxisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.long = build_sample(cls.reader)

    def test_one_customer_day_is_expanded_to_48_rows_per_category(self) -> None:
        counts = self.long.groupby(
            "Consumption Category", observed=True
        ).size()
        self.assertEqual(len(self.long), 144)
        self.assertEqual(counts.to_dict(), {"CL": 48, "GC": 48, "GG": 48})

    def test_interval_start_covers_full_day(self) -> None:
        expected = pd.date_range(
            "2010-07-01 00:00:00", periods=48, freq="30min"
        )
        for _, group in self.long.groupby("Consumption Category", observed=True):
            actual = pd.DatetimeIndex(group["interval_start"]).sort_values()
            actual = actual.rename(None)
            pd.testing.assert_index_equal(actual, expected)

    def test_last_zero_label_maps_to_end_of_source_day(self) -> None:
        gc = self.long[
            (self.long["Consumption Category"] == "GC")
            & (self.long["interval_end_label"] == "0:00")
        ].iloc[0]

        self.assertEqual(gc["interval_start"], pd.Timestamp("2010-07-01 23:30"))
        self.assertEqual(gc["interval_end"], pd.Timestamp("2010-07-02 00:00"))
        self.assertEqual(gc["slot_index"], 47)
        self.assertAlmostEqual(gc["energy_kwh"], 0.125)

    def test_first_interval_value_is_preserved(self) -> None:
        gc = self.long[
            (self.long["Consumption Category"] == "GC")
            & (self.long["interval_end_label"] == "0:30")
        ].iloc[0]

        self.assertEqual(gc["interval_start"], pd.Timestamp("2010-07-01 00:00"))
        self.assertEqual(gc["interval_end"], pd.Timestamp("2010-07-01 00:30"))
        self.assertEqual(gc["slot_index"], 0)
        self.assertAlmostEqual(gc["energy_kwh"], 0.303, places=6)


if __name__ == "__main__":
    unittest.main()
