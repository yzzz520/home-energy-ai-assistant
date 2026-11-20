from __future__ import annotations

import unittest

from batch_process import process_file_customers
from reader import AusgridReader


class BatchProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.spec = cls.reader.file_specs["Solar home 2012-2013.csv"]
        cls.details, cls.summary = process_file_customers(
            cls.reader,
            cls.spec,
            customer_ids=[1, 2],
            progress_every=0,
        )

    def test_customer_details_are_one_row_per_customer(self) -> None:
        self.assertEqual(len(self.details), 2)
        self.assertEqual(
            self.details["customer_id"].tolist(),
            [1, 2],
        )

    def test_annual_summary_aggregates_missing_rows(self) -> None:
        self.assertEqual(self.summary["customers"], 2)
        self.assertEqual(self.summary["rows"], 35_040)
        self.assertEqual(self.summary["customers_with_missing_days"], 1)
        self.assertEqual(self.summary["missing_customer_days"], 81)
        self.assertEqual(self.summary["missing_energy_rows"], 3_888)
        self.assertEqual(self.summary["complete_customers"], 1)

    def test_availability_percentage_is_computed(self) -> None:
        customer_2 = self.details.set_index("customer_id").loc[2]
        self.assertAlmostEqual(
            customer_2["availability_pct"],
            77.8082191780822,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
