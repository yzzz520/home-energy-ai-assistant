from __future__ import annotations

import unittest

import pandas as pd

from energy_balance import build_balance_sample
from reader import AusgridReader


class EnergyBalanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.with_cl = build_balance_sample(cls.reader, customer_id=1)
        cls.without_cl = build_balance_sample(cls.reader, customer_id=11)

    def test_total_load_is_gc_plus_cl(self) -> None:
        expected = self.with_cl["gc_kwh"] + self.with_cl["cl_kwh"]
        pd.testing.assert_series_equal(
            self.with_cl["total_load_kwh"],
            expected,
            check_names=False,
        )

    def test_net_grid_is_total_load_minus_pv(self) -> None:
        expected = self.with_cl["total_load_kwh"] - self.with_cl["gg_kwh"]
        pd.testing.assert_series_equal(
            self.with_cl["net_grid_kwh"],
            expected,
            check_names=False,
        )

    def test_night_period_is_import(self) -> None:
        first = self.with_cl.iloc[0]
        self.assertAlmostEqual(first["total_load_kwh"], 1.553, places=6)
        self.assertAlmostEqual(first["net_grid_kwh"], 1.553, places=6)
        self.assertGreater(first["net_grid_kwh"], 0)

    def test_solar_surplus_period_is_export(self) -> None:
        row = self.with_cl[
            self.with_cl["interval_start"] == pd.Timestamp("2010-07-01 09:30")
        ].iloc[0]
        self.assertAlmostEqual(row["gc_kwh"], 0.208, places=6)
        self.assertAlmostEqual(row["gg_kwh"], 0.456, places=6)
        self.assertAlmostEqual(row["net_grid_kwh"], -0.248, places=6)
        self.assertLess(row["net_grid_kwh"], 0)

    def test_no_cl_customer_balance_is_correct(self) -> None:
        expected_total = self.without_cl["gc_kwh"]
        pd.testing.assert_series_equal(
            self.without_cl["total_load_kwh"],
            expected_total,
            check_names=False,
            check_dtype=False,
        )
        self.assertFalse(self.with_cl[["total_load_kwh", "net_grid_kwh"]].isna().any().any())
        self.assertFalse(self.without_cl[["total_load_kwh", "net_grid_kwh"]].isna().any().any())

    def test_import_and_export_are_non_negative(self) -> None:
        flow_columns = ["import_kwh", "export_kwh"]
        self.assertFalse(self.with_cl[flow_columns].isna().any().any())
        self.assertFalse((self.with_cl[flow_columns] < 0).any().any())

    def test_import_export_reconstruct_net_grid(self) -> None:
        reconstructed = self.with_cl["import_kwh"] - self.with_cl["export_kwh"]
        pd.testing.assert_series_equal(
            reconstructed,
            self.with_cl["net_grid_kwh"],
            check_names=False,
        )

    def test_import_and_export_never_coexist(self) -> None:
        simultaneous = (
            (self.with_cl["import_kwh"] > 0)
            & (self.with_cl["export_kwh"] > 0)
        )
        self.assertFalse(simultaneous.any())

    def test_night_import_and_solar_export_values(self) -> None:
        night = self.with_cl.iloc[0]
        solar = self.with_cl[
            self.with_cl["interval_start"] == pd.Timestamp("2010-07-01 09:30")
        ].iloc[0]

        self.assertAlmostEqual(night["import_kwh"], 1.553, places=6)
        self.assertAlmostEqual(night["export_kwh"], 0.0, places=6)
        self.assertAlmostEqual(solar["import_kwh"], 0.0, places=6)
        self.assertAlmostEqual(solar["export_kwh"], 0.248, places=6)


if __name__ == "__main__":
    unittest.main()
