from __future__ import annotations

import unittest

import pandas as pd

from pipeline import process_customer_year
from reader import AusgridReader


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.clean = process_customer_year(
            cls.reader,
            customer_id=1,
            source_file="Solar home 2010-2011.csv",
        )
        cls.customer_2 = process_customer_year(
            cls.reader,
            customer_id=2,
            source_file="Solar home 2012-2013.csv",
        )

    def test_customer_year_has_expected_rows_and_dates(self) -> None:
        self.assertEqual(len(self.clean), 17_520)
        self.assertEqual(self.clean["source_date"].nunique(), 365)
        self.assertEqual(self.clean["interval_start"].nunique(), 17_520)

    def test_every_day_has_48_intervals(self) -> None:
        daily_counts = self.clean.groupby("source_date").size()
        self.assertTrue((daily_counts == 48).all())

    def test_expected_columns_are_present(self) -> None:
        expected = {
            "Customer",
            "source_date",
            "interval_start",
            "interval_end",
            "slot_index",
            "gc_kwh",
            "cl_kwh",
            "gg_kwh",
            "total_load_kwh",
            "net_grid_kwh",
            "import_kwh",
            "export_kwh",
        }
        self.assertTrue(expected.issubset(self.clean.columns))

    def test_energy_balance_identities_hold(self) -> None:
        reconstructed_load = self.clean["gc_kwh"] + self.clean["cl_kwh"]
        reconstructed_net = (
            self.clean["import_kwh"] - self.clean["export_kwh"]
        )
        self.assertTrue(
            ((self.clean["total_load_kwh"] - reconstructed_load).abs() < 1e-8).all()
        )
        self.assertTrue(
            ((self.clean["net_grid_kwh"] - reconstructed_net).abs() < 1e-8).all()
        )

    def test_no_missing_energy_values(self) -> None:
        energy_columns = [
            "gc_kwh",
            "cl_kwh",
            "gg_kwh",
            "total_load_kwh",
            "net_grid_kwh",
            "import_kwh",
            "export_kwh",
        ]
        self.assertFalse(self.clean[energy_columns].isna().any().any())
        self.assertFalse(self.clean["is_missing_interval"].any())

    def test_time_range_and_customer_are_correct(self) -> None:
        self.assertEqual(self.clean["Customer"].unique().tolist(), [1])
        self.assertEqual(
            self.clean["interval_start"].min(),
            pd.Timestamp("2010-07-01 00:00"),
        )
        self.assertEqual(
            self.clean["interval_start"].max(),
            pd.Timestamp("2011-06-30 23:30"),
        )

    def test_customer_2_missing_gap_is_kept_as_missing_rows(self) -> None:
        self.assertEqual(len(self.customer_2), 17_520)
        self.assertEqual(self.customer_2["source_date"].nunique(), 365)
        self.assertEqual(
            int(self.customer_2.loc[self.customer_2["is_missing_day"], "source_date"].nunique()),
            81,
        )
        self.assertEqual(
            int(self.customer_2["is_missing_interval"].sum()),
            81 * 48,
        )
        self.assertTrue(
            self.customer_2.loc[
                self.customer_2["is_missing_day"], "total_load_kwh"
            ].isna().all()
        )
        self.assertEqual(
            self.customer_2.loc[
                self.customer_2["is_missing_day"], "gc_quality"
            ].unique().tolist(),
            ["MISSING_DAY"],
        )

    def test_dst_metadata_is_present(self) -> None:
        self.assertIn("interval_start_utc", self.clean.columns)
        self.assertIn("dst_transition_day", self.clean.columns)
        self.assertIn("model_usable", self.clean.columns)


if __name__ == "__main__":
    unittest.main()
