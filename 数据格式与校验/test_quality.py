from __future__ import annotations

import unittest

import pandas as pd

from category_pivot import build_pivot_sample
from energy_balance import build_balance_sample
from quality import (
    add_outlier_flags,
    add_time_metadata,
    screen_completeness,
    summarize_clean_data,
)
from reader import AusgridReader


class CompletenessScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.raw_2012 = cls.reader.read_single_csv("Solar home 2012-2013.csv")
        cls.spec_2012 = cls.reader.file_specs["Solar home 2012-2013.csv"]

    def test_customer_2_gap_is_reported_in_human_readable_form(self) -> None:
        details, summary = screen_completeness(self.raw_2012, self.spec_2012)
        customer_2 = details.set_index("customer_id").loc[2]

        self.assertEqual(summary["customers_with_missing_days"], 1)
        self.assertEqual(customer_2["observed_days"], 284)
        self.assertEqual(customer_2["missing_days"], 81)
        self.assertEqual(
            customer_2["missing_date_ranges"],
            "2012-10-12..2012-12-31",
        )

    def test_row_quality_na_is_counted(self) -> None:
        _, summary = screen_completeness(self.raw_2012, self.spec_2012)
        self.assertEqual(summary["non_actual_source_rows"], 431)

    def test_cl_absence_is_not_treated_as_incomplete(self) -> None:
        raw_2011 = self.reader.read_single_csv("Solar home 2011-2012.csv")
        spec_2011 = self.reader.file_specs["Solar home 2011-2012.csv"]
        details, summary = screen_completeness(raw_2011, spec_2011)

        self.assertEqual(summary["incomplete_customers"], 0)
        self.assertGreater(
            int(details.set_index("customer_id").loc[27, "cl_absent_days"]),
            0,
        )


class TimeAndOutlierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.clean = build_balance_sample(cls.reader, customer_id=1)

    def test_dst_gap_rows_are_not_guessed(self) -> None:
        frame = pd.DataFrame(
            {
                "Customer": [1, 1, 1],
                "source_date": pd.to_datetime(
                    ["2010-10-03", "2010-10-03", "2010-10-04"]
                ),
                "interval_start": pd.to_datetime(
                    [
                        "2010-10-03 02:00",
                        "2010-10-03 03:00",
                        "2010-10-04 02:00",
                    ]
                ),
            }
        )
        result = add_time_metadata(frame)

        self.assertTrue(result.iloc[0]["dst_transition_day"])
        self.assertFalse(bool(result.iloc[0]["utc_resolvable"]))
        self.assertTrue(pd.isna(result.iloc[0]["interval_start_utc"]))
        self.assertTrue(bool(result.iloc[1]["utc_resolvable"]))
        self.assertFalse(bool(result.iloc[2]["dst_transition_day"]))
        self.assertTrue(bool(result.iloc[2]["utc_resolvable"]))

    def test_physically_impossible_pv_is_flagged_and_masked(self) -> None:
        frame = pd.DataFrame(
            {
                "Customer": [1, 1],
                "Generator Capacity": [2.0, 2.0],
                "gg_kwh": [0.4, 2.0],
                "total_load_kwh": [0.5, 0.6],
                "quality_non_actual": [False, False],
            }
        )
        result = add_outlier_flags(frame)

        self.assertFalse(bool(result.iloc[0]["pv_over_capacity"]))
        self.assertTrue(bool(result.iloc[1]["pv_over_capacity"]))
        self.assertAlmostEqual(result.iloc[0]["gg_clean_kwh"], 0.4)
        self.assertTrue(pd.isna(result.iloc[1]["gg_clean_kwh"]))

    def test_load_extreme_is_not_automatically_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "Customer": [1] * 11,
                "Generator Capacity": [2.0] * 11,
                "gg_kwh": [0.0] * 11,
                "total_load_kwh": [0.2, 0.3, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.4, 0.3, 50.0],
                "quality_non_actual": [False] * 11,
            }
        )
        result = add_outlier_flags(frame, load_robust_z_threshold=8.0)

        self.assertTrue(bool(result.iloc[-1]["load_extreme"]))
        self.assertTrue(bool(result.iloc[-1]["model_usable"]))

    def test_clean_sample_has_quality_summary(self) -> None:
        timed = add_time_metadata(self.clean)
        flagged = add_outlier_flags(
            timed,
            pv_max_energy_per_kwp_kwh=self.reader.pv_max_energy_per_kwp_kwh,
            load_robust_z_threshold=self.reader.load_robust_z_threshold,
        )
        summary = summarize_clean_data(flagged)

        self.assertEqual(summary["rows"], 48)
        self.assertEqual(summary["missing_energy_rows"], 0)
        self.assertEqual(summary["utc_unresolved_rows"], 0)


class RowQualityPropagationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.sample = build_pivot_sample(cls.reader, customer_id=1)

    def test_quality_columns_exist_and_blank_is_preserved(self) -> None:
        self.assertIn("gc_quality", self.sample.columns)
        self.assertIn("cl_quality", self.sample.columns)
        self.assertIn("gg_quality", self.sample.columns)
        self.assertTrue(self.sample["gc_quality"].eq("").all())
        self.assertFalse(self.sample["quality_non_actual"].any())


if __name__ == "__main__":
    unittest.main()
