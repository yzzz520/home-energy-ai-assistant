from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ENGINEERING_ROOT = PROJECT_ROOT / "数据格式与校验"
if str(DATA_ENGINEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ENGINEERING_ROOT))

import pandas as pd

from lightgbm_global import (
    GLOBAL_FEATURES,
    build_global_feature_frame,
    train_global_fold,
)
from lightgbm_walk_forward import walk_forward_folds
from reader import AusgridReader
from seasonal_naive import evaluate_forecasts


def main() -> None:
    started = time.perf_counter()
    reader = AusgridReader()
    cluster_path = (
        DATA_ENGINEERING_ROOT
        / "EDA探索性数据分析"
        / "reports"
        / "customer_gc_weekday_clusters.csv"
    )
    global_frame = build_global_feature_frame(
        reader,
        cluster_path=cluster_path,
        customer_ids=range(1, 31),
        progress_every=0,
    )
    folds = walk_forward_folds(global_frame, n_folds=3, fold_days=20)
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        model, _, _, validation_bias = train_global_fold(
            global_frame,
            test_start=fold["test_start"],
            random_state=500 + fold_number,
            objective="regression_l1",
            num_boost_round=1200,
        )
        test = global_frame.loc[
            global_frame["interval_start"].between(
                fold["test_start"],
                fold["test_end"],
                inclusive="both",
            )
            & global_frame["total_load_kwh"].notna()
        ].dropna(subset=GLOBAL_FEATURES)
        predictions = pd.DataFrame(
            {
                "fold": fold_number,
                "forecast_timestamp": test["interval_start"].to_numpy(),
                "Customer": test["Customer"].to_numpy(),
                "actual": test["total_load_kwh"].to_numpy(),
                "l1_1200": model.predict(
                    test[GLOBAL_FEATURES],
                    num_iteration=model.best_iteration or None,
                ),
                "seasonal_naive_daily": test[
                    "seasonal_naive_daily"
                ].to_numpy(),
            }
        )
        prediction_frames.append(predictions)
        metrics = evaluate_forecasts(
            predictions,
            forecast_columns=("l1_1200", "seasonal_naive_daily"),
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "test_start": fold["test_start"].date().isoformat(),
                "test_end": fold["test_end"].date().isoformat(),
                "best_iteration": int(model.best_iteration or 1200),
                "validation_bias_correction_kwh": round(validation_bias, 6),
                "elapsed_seconds": round(time.perf_counter() - fold_started, 3),
                "metrics": metrics,
            }
        )
        print(
            f"[Fold {fold_number}] L1-1200 MAE="
            f"{metrics['l1_1200']['mae_kwh']:.4f}, "
            f"bias={metrics['l1_1200']['mean_bias_kwh']:.4f}, "
            f"best_iter={model.best_iteration}"
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    report = {
        "customers": 30,
        "evaluation_points": len(all_predictions),
        "n_folds": 3,
        "max_boost_round": 1200,
        "folds": fold_rows,
        "overall_metrics": evaluate_forecasts(
            all_predictions,
            forecast_columns=("l1_1200", "seasonal_naive_daily"),
        ),
        "script_elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "experiment_30_households_l1_1200.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 指标：{report_path}")


if __name__ == "__main__":
    main()
