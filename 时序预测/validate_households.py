from __future__ import annotations

import argparse
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
    build_global_feature_frame,
    plot_global_metrics,
    run_global_walk_forward,
)
from reader import AusgridReader


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_fold_mae(
    report: dict[str, object],
    output_path: Path,
    household_count: int,
) -> None:
    folds = report["folds"]
    x = [fold["fold"] for fold in folds]
    models = {
        "LightGBM": ("lightgbm", "#2F7657"),
        "昨日同时刻": ("seasonal_naive_daily", "#24708F"),
        "上周同时刻": ("seasonal_naive_weekly", "#C4592E"),
    }

    figure, axis = plt.subplots(figsize=(11, 6))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    width = 0.24
    offsets = (-width, 0.0, width)
    for offset, (label, (key, color)) in zip(
        offsets,
        models.items(),
        strict=True,
    ):
        axis.bar(
            [value + offset for value in x],
            [fold["metrics"][key]["mae_kwh"] for fold in folds],
            width=width,
            label=label,
            color=color,
        )

    axis.set_title(
        f"{household_count} 户 × 3 折 Walk-forward MAE 对比",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_xlabel("Walk-forward 折")
    axis.set_ylabel("MAE（kWh / 半小时）")
    axis.set_xticks(x)
    axis.set_xticklabels([f"Fold {value}" for value in x])
    axis.legend(loc="upper left", frameon=False)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=30)
    args = parser.parse_args()
    if args.households <= 0:
        raise ValueError("--households 必须为正整数")

    script_started = time.perf_counter()
    reader = AusgridReader()
    cluster_path = (
        DATA_ENGINEERING_ROOT
        / "EDA探索性数据分析"
        / "reports"
        / "customer_gc_weekday_clusters.csv"
    )
    feature_started = time.perf_counter()
    global_frame = build_global_feature_frame(
        reader,
        cluster_path=cluster_path,
        customer_ids=range(1, args.households + 1),
        progress_every=0,
    )
    feature_seconds = time.perf_counter() - feature_started

    training_started = time.perf_counter()
    predictions, per_customer, importance, report = run_global_walk_forward(
        global_frame,
        n_folds=3,
    )
    training_seconds = time.perf_counter() - training_started

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    report["stage"] = f"{args.households} households x 3 folds"
    report["feature_build_seconds"] = round(feature_seconds, 3)
    report["training_and_prediction_seconds"] = round(training_seconds, 3)
    report["script_elapsed_seconds"] = round(
        time.perf_counter() - script_started,
        3,
    )
    prefix = f"validation_{args.households}_households"
    report_path = output_dir / f"{prefix}_lightgbm_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    per_customer_path = output_dir / f"{prefix}_per_customer.csv"
    per_customer.to_csv(per_customer_path, index=False, encoding="utf-8-sig")
    importance_path = output_dir / f"{prefix}_importance.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    prediction_path = output_dir / f"{prefix}_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    overall_path = output_dir / f"{prefix}_overall_metrics.png"
    plot_global_metrics(report, output_path=overall_path)
    folds_path = output_dir / f"{prefix}_fold_mae.png"
    plot_fold_mae(report, folds_path, args.households)

    print(
        f"[OK] {args.households} 户，特征构建 {feature_seconds:.1f} 秒，"
        f"训练预测 {training_seconds:.1f} 秒，"
        f"评估点 {report['evaluation_points']:,}"
    )
    for fold in report["folds"]:
        daily = fold["metrics"]["seasonal_naive_daily"]["mae_kwh"]
        lightgbm = fold["metrics"]["lightgbm"]["mae_kwh"]
        bias = fold["metrics"]["lightgbm"]["mean_bias_kwh"]
        improvement = (daily - lightgbm) / daily * 100
        print(
            f"[折 {fold['fold']}] LightGBM={lightgbm:.4f}，"
            f"昨日={daily:.4f}，提升={improvement:.2f}%，"
            f"偏差={bias:.4f} kWh，"
            f"耗时={fold['elapsed_seconds']:.1f}s"
        )
    print(f"[OK] 指标：{report_path}")
    print(f"[OK] 折对比图：{folds_path}")


if __name__ == "__main__":
    main()
