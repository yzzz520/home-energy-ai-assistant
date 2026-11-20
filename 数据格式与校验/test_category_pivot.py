from __future__ import annotations

import unittest

import pandas as pd

from category_pivot import build_pivot_sample
from reader import AusgridReader


class CategoryPivotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.with_cl = build_pivot_sample(cls.reader, customer_id=1)
        cls.without_cl = build_pivot_sample(cls.reader, customer_id=11)

    def test_with_cl_becomes_48_rows(self) -> None:
        self.assertEqual(len(self.with_cl), 48)
        self.assertEqual(
            self.with_cl["interval_start"].nunique(),
            48,
        )

    def test_category_values_are_side_by_side(self) -> None:
        first = self.with_cl.iloc[0]
        self.assertAlmostEqual(first["gc_kwh"], 0.303, places=6)
        self.assertAlmostEqual(first["cl_kwh"], 1.250, places=6)
        self.assertAlmostEqual(first["gg_kwh"], 0.0, places=6)
        self.assertTrue(first["has_controlled_load"])

    def test_no_cl_customer_gets_zero_instead_of_missing(self) -> None:
        self.assertEqual(len(self.without_cl), 48)
        self.assertFalse(self.without_cl["has_controlled_load"].any())
        self.assertTrue((self.without_cl["cl_kwh"] == 0).all())
        self.assertFalse(self.without_cl[["gc_kwh", "cl_kwh", "gg_kwh"]].isna().any().any())

    def test_one_row_per_customer_interval(self) -> None:
        duplicate = self.with_cl.duplicated(["Customer", "interval_start"])
        self.assertFalse(duplicate.any())

    def test_time_range_is_preserved(self) -> None:
        expected = pd.date_range(
            "2010-07-01 00:00:00", periods=48, freq="30min"
        )
        actual = pd.DatetimeIndex(self.with_cl["interval_start"]).sort_values()
        actual = actual.rename(None)
        pd.testing.assert_index_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
