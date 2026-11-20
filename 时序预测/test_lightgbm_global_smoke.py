from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm_global import (
    CATEGORICAL_FEATURES,
    CUSTOMER_FEATURES,
    GLOBAL_FEATURES,
    build_global_feature_frame,
    load_cluster_labels,
    train_global_fold,
)
from reader import AusgridReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_PATH = (
    PROJECT_ROOT
    / "数据格式与校验"
    / "EDA探索性数据分析"
    / "reports"
    / "customer_gc_weekday_clusters.csv"
)
RESULTS_DIR = Path(__file__).with_name("results")
EXPECTED_HOUSEHOLD_FEATURES = {
    "Customer",
    "Generator Capacity",
    "Postcode",
    "cluster_id",
}


class GlobalLightGBMSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reader = AusgridReader()
        cls.global_frame = build_global_feature_frame(
            reader,
            cluster_path=CLUSTER_PATH,
            customer_ids=range(1, 9),
            progress_every=0,
        )

    def test_household_features_reach_global_frame(self) -> None:
        self.assertEqual(set(CUSTOMER_FEATURES), EXPECTED_HOUSEHOLD_FEATURES)
        self.assertTrue(EXPECTED_HOUSEHOLD_FEATURES.issubset(GLOBAL_FEATURES))
        self.assertTrue(
            EXPECTED_HOUSEHOLD_FEATURES.issubset(self.global_frame.columns)
        )
        self.assertEqual(self.global_frame["Customer"].nunique(), 8)
        self.assertTrue(
            self.global_frame[list(EXPECTED_HOUSEHOLD_FEATURES)].notna().all().all()
        )
        self.assertTrue(self.global_frame["Generator Capacity"].gt(0).all())
        for column in CATEGORICAL_FEATURES:
            self.assertEqual(
                str(self.global_frame[column].dtype),
                "category",
            )

    def test_ten_round_global_model_can_train_and_predict(self) -> None:
        model, importance, training_rows, _ = train_global_fold(
            self.global_frame,
            test_start=pd.Timestamp("2011-05-16"),
            random_state=2026,
            num_threads=1,
            num_boost_round=10,
        )

        self.assertGreaterEqual(training_rows, 100_000)
        self.assertEqual(model.num_feature(), len(GLOBAL_FEATURES))
        self.assertEqual(
            set(importance["feature"]),
            set(GLOBAL_FEATURES),
        )

        sample = self.global_frame.loc[
            self.global_frame["interval_start"].ge("2011-05-16")
            & self.global_frame["total_load_kwh"].notna()
        ].head(96)
        predictions = model.predict(
            sample[GLOBAL_FEATURES],
            num_iteration=model.best_iteration or None,
        )
        self.assertEqual(len(predictions), len(sample))
        self.assertTrue(np.isfinite(predictions).all())

    def test_final_300_household_artifacts_are_complete(self) -> None:
        clusters = load_cluster_labels(CLUSTER_PATH)
        self.assertEqual(len(clusters), 300)
        self.assertEqual(
            set(clusters["customer_id"]),
            set(range(1, 301)),
        )
        self.assertTrue(clusters[["cluster_id", "cluster_label"]].notna().all().all())

        predictions = pd.read_csv(
            RESULTS_DIR / "validation_300_households_predictions.csv",
            encoding="utf-8-sig",
            usecols=["fold", "Customer", "forecast_timestamp"],
        )
        self.assertEqual(len(predictions), 864_000)
        self.assertEqual(predictions["Customer"].nunique(), 300)
        self.assertEqual(sorted(predictions["fold"].unique().tolist()), [1, 2, 3])
        self.assertEqual(
            predictions.groupby("fold").size().to_dict(),
            {1: 288_000, 2: 288_000, 3: 288_000},
        )

        metrics = json.loads(
            (
                RESULTS_DIR
                / "validation_300_households_lightgbm_metrics.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["customers"], 300)
        self.assertEqual(metrics["evaluation_points"], 864_000)
        self.assertEqual(metrics["n_folds"], 3)

        importance = pd.read_csv(
            RESULTS_DIR / "validation_300_households_importance.csv",
            encoding="utf-8-sig",
        )
        self.assertTrue(
            EXPECTED_HOUSEHOLD_FEATURES.issubset(set(importance["feature"]))
        )


if __name__ == "__main__":
    unittest.main()
