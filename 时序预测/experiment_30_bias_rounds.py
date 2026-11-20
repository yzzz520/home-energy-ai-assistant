from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ENGINEERING_ROOT = PROJECT_ROOT / "数据格式与校验"
if str(DATA_ENGINEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ENGINEERING_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from lightgbm_global import (
    GLOBAL_FEATURES,
    build_global_feature_frame,
    train_global_fold,
)
from lightgbm_walk_forward import walk_forward_folds
from reader import AusgridReader
from seasonal_naive import evaluate_forecasts


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_COLUMNS = (
    "l1_800",
    "l1_800_bias_corrected",
    "l2_800",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
)


def plot_experiment(
    report: dict[str, object],
    output_path: Path,
) -> None:
    folds = report["folds"]
    x = [fold["fold"] for fold in folds]
    model_specs = {
        "L1 800": ("l1_800", "#24708F"),
        "L1 800 + 偏差修正": ("l1_800_bias_corrected", "#2F7657"),
        "L2 800": ("l2_800", "#C4592E"),
    }

    figure, (mae_axis, bias_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (mae_axis, bias_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    width = 0.24
    offsets = (-width, 0.0, width)
    for offset, (label, (column, color)) in zip(
        offsets,
        model_specs.items(),
        strict=True,
    ):
        mae_axis.bar(
            [value + offset for value in x],
            [fold["metrics"][column]["mae_kwh"] for fold in folds],
            width=width,
            color=color,
            label=label,
        )
        bias_axis.bar(
            [value + offset for value in x],
            [fold["metrics"][column]["mean_bias_kwh"] for fold in folds],
            width=width,
            color=color,
            label=label,
        )

    mae_axis.set_title("MAE（越低越好）", loc="left", fontsize=14, fontweight="bold")
    mae_axis.set_ylabel("kWh / 半小时")
    mae_axis.set_xticks(x)
    mae_axis.set_xticklabels([f"Fold {value}" for value in x])
    mae_axis.legend(loc="upper left", frameon=False)

    bias_axis.axhline(0, color="#47545D", linewidth=1)
    bias_axis.set_title("平均偏差（越接近 0 越好）", loc="left", fontsize=14, fontweight="bold")
    bias_axis.set_ylabel("预测 - 实际（kWh）")
    bias_axis.set_xticks(x)
    bias_axis.set_xticklabels([f"Fold {value}" for value in x])

    figure.suptitle(
        "30 户 × 3 折：迭代上限与低估偏差实验",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.15, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    script_started = time.perf_counter()
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
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for fold_number, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        l1_model, _, _, validation_bias = train_global_fold(
            global_frame,
            test_start=fold["test_start"],
            random_state=200 + fold_number,
            objective="regression_l1",
            num_boost_round=800,
        )
        l2_model, _, _, _ = train_global_fold(
            global_frame,
            test_start=fold["test_start"],
            random_state=300 + fold_number,
            objective="regression_l2",
            num_boost_round=800,
        )
        test = global_frame.loc[
            global_frame["interval_start"].between(
                fold["test_start"],
                fold["test_end"],
                inclusive="both",
            )
            & global_frame["total_load_kwh"].notna()
        ].dropna(subset=GLOBAL_FEATURES)

        l1_prediction = l1_model.predict(
            test[GLOBAL_FEATURES],
            num_iteration=l1_model.best_iteration or None,
        )
        l2_prediction = l2_model.predict(
            test[GLOBAL_FEATURES],
            num_iteration=l2_model.best_iteration or None,
        )
        predictions = pd.DataFrame(
            {
                "fold": fold_number,
                "forecast_timestamp": test["interval_start"].to_numpy(),
                "Customer": test["Customer"].to_numpy(),
                "actual": test["total_load_kwh"].to_numpy(),
                "l1_800": l1_prediction,
                "l1_800_bias_corrected": l1_prediction + validation_bias,
                "l2_800": l2_prediction,
                "seasonal_naive_daily": test[
                    "seasonal_naive_daily"
                ].to_numpy(),
                "seasonal_naive_weekly": test[
                    "seasonal_naive_weekly"
                ].to_numpy(),
            }
        )
        prediction_frames.append(predictions)
        metrics = evaluate_forecasts(
            predictions,
            forecast_columns=MODEL_COLUMNS,
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "test_start": fold["test_start"].date().isoformat(),
                "test_end": fold["test_end"].date().isoformat(),
                "validation_bias_correction_kwh": round(validation_bias, 6),
                "l1_best_iteration": int(l1_model.best_iteration or 800),
                "l2_best_iteration": int(l2_model.best_iteration or 800),
                "elapsed_seconds": round(time.perf_counter() - fold_started, 3),
                "metrics": metrics,
            }
        )
        print(
            f"[Fold {fold_number}] "
            f"L1={metrics['l1_800']['mae_kwh']:.4f}, "
            f"L1+修正={metrics['l1_800_bias_corrected']['mae_kwh']:.4f}, "
            f"L2={metrics['l2_800']['mae_kwh']:.4f}, "
            f"bias修正={validation_bias:.4f}"
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    overall_metrics = evaluate_forecasts(
        all_predictions,
        forecast_columns=MODEL_COLUMNS,
    )
    report = {
        "customers": 30,
        "evaluation_points": len(all_predictions),
        "n_folds": 3,
        "fold_days": 20,
        "max_boost_round": 800,
        "folds": fold_rows,
        "overall_metrics": overall_metrics,
        "script_elapsed_seconds": round(
            time.perf_counter() - script_started,
            3,
        ),
    }

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "experiment_30_households_rounds_bias.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_path = output_dir / "experiment_30_households_rounds_bias.png"
    plot_experiment(report, figure_path)
    prediction_path = output_dir / "experiment_30_households_rounds_bias_predictions.csv"
    all_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    print(
        f"[OK] 总耗时 {report['script_elapsed_seconds']:.1f} 秒，"
        f"评估点 {report['evaluation_points']:,}"
    )
    print(f"[OK] 实验指标：{report_path}")
    print(f"[OK] 实验图：{figure_path}")


if __name__ == "__main__":
    main()
