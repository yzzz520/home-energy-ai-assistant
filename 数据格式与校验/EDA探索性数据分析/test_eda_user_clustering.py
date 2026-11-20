from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from eda_user_clustering import (
    cluster_profile_matrix,
    plot_cluster_profiles,
)


class UserClusteringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        slots = np.arange(48)
        morning = np.exp(-0.5 * ((slots - 14) / 3.0) ** 2)
        evening = np.exp(-0.5 * ((slots - 38) / 4.0) ** 2)
        flat = np.ones(48)
        rows = []
        index = []
        for group, base in enumerate((morning, evening, flat)):
            for member in range(12):
                noise = np.random.default_rng(group * 100 + member).normal(
                    0,
                    0.015,
                    size=48,
                )
                rows.append(base + noise)
                index.append(group * 100 + member)
        cls.matrix = pd.DataFrame(rows, index=index)
        cls.assignments, cls.summary, cls.metadata = cluster_profile_matrix(
            cls.matrix,
            k_min=2,
            k_max=4,
        )

    def test_clustering_returns_all_customers(self) -> None:
        self.assertEqual(len(self.assignments), 36)
        self.assertTrue(self.assignments["cluster_id"].notna().all())

    def test_cluster_count_is_interpretable(self) -> None:
        self.assertIn(self.metadata["selected_k"], {2, 3, 4})
        self.assertGreaterEqual(self.metadata["silhouette"], 0)
        self.assertEqual(
            int(self.summary["customers"].sum()),
            len(self.assignments),
        )

    def test_cluster_figure_can_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "clusters.png"
            plot_cluster_profiles(
                self.matrix,
                self.assignments,
                self.summary,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
