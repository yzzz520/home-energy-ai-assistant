from __future__ import annotations

import unittest

import pandas as pd

from reader import AusgridReader


class AusgridReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.df = cls.reader.read_all()

    def test_total_rows_and_categories(self) -> None:
        self.assertEqual(len(self.df), 808_596)
        counts = self.df["Consumption Category"].value_counts().to_dict()
        self.assertEqual(counts["GC"], 328_719)
        self.assertEqual(counts["GG"], 328_719)
        self.assertEqual(counts["CL"], 151_158)

    def test_customer_and_postcode_counts(self) -> None:
        self.assertEqual(self.df["Customer"].nunique(), 300)
        self.assertEqual(self.df["Postcode"].nunique(), 100)
        self.assertEqual(
            set(self.df["Customer"].unique()),
            set(range(1, 301)),
        )

    def test_no_duplicate_customer_date_category(self) -> None:
        duplicates = self.df.duplicated(
            ["Customer", "date", "Consumption Category"]
        )
        self.assertFalse(duplicates.any())

    def test_customer_metadata_is_stable(self) -> None:
        metadata = self.df.groupby("Customer")[
            ["Generator Capacity", "Postcode"]
        ].nunique()
        self.assertTrue((metadata == 1).all().all())

    def test_customer_2_2012_2013_gap_is_preserved(self) -> None:
        customer_2 = self.df[
            (self.df["Customer"] == 2)
            & (self.df["source_year"] == "2012-2013")
        ]
        dates = pd.DatetimeIndex(customer_2["date"].unique()).sort_values()

        self.assertEqual(len(dates), 284)
        self.assertEqual(dates.min(), pd.Timestamp("2012-07-01"))
        self.assertEqual(dates.max(), pd.Timestamp("2013-06-30"))
        self.assertNotIn(pd.Timestamp("2012-10-12"), dates)
        self.assertNotIn(pd.Timestamp("2012-12-31"), dates)

    def test_energy_cells_are_complete_and_non_negative(self) -> None:
        intervals = self.reader.interval_columns
        self.assertFalse(self.df[intervals].isna().any().any())
        self.assertFalse((self.df[intervals] < 0).any().any())

    def test_2010_file_gets_uniform_empty_quality_column(self) -> None:
        first_year = self.df[self.df["source_year"] == "2010-2011"]
        self.assertIn("Row Quality", first_year.columns)
        self.assertTrue(first_year["Row Quality"].astype("string").eq("").all())

    def test_quality_report_checks_pass(self) -> None:
        report = self.reader.build_quality_report(self.df)
        self.assertEqual(report["total_rows"], 808_596)
        self.assertEqual(report["row_quality"]["blank"], 808_065)
        self.assertEqual(report["row_quality"]["NA"], 531)
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
