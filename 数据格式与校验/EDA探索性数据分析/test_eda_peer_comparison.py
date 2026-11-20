from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eda_peer_comparison import (
    collect_customer_profiles,
    compare_to_population,
    plot_peer_comparison,
)
from reader import AusgridReader


class PeerComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.spec = cls.reader.file_specs["Solar home 2010-2011.csv"]
        cls.profiles = collect_customer_profiles(
            cls.reader,
            cls.spec,
            customer_ids=[1, 11, 23],
            progress_every=0,
        )

    def test_profile_collection_has_48_slots_per_customer(self) -> None:
        counts = self.profiles.groupby("customer_id").size().to_dict()
        self.assertEqual(counts, {1: 48, 11: 48, 23: 48})

    def test_comparison_metrics_are_computed(self) -> None:
        comparison, metrics = compare_to_population(
            self.profiles,
            customer_id=1,
        )
        self.assertEqual(len(comparison), 48)
        self.assertEqual(metrics["customers"], 3)
        self.assertGreaterEqual(metrics["correlation_with_population"], -1)
        self.assertLessEqual(metrics["correlation_with_population"], 1)
        self.assertGreater(metrics["mae_vs_population_kw"], 0)

    def test_comparison_figure_can_be_written(self) -> None:
        comparison, metrics = compare_to_population(
            self.profiles,
            customer_id=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "peer_comparison.png"
            plot_peer_comparison(
                comparison,
                metrics,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
