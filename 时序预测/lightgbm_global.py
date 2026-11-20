from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ENGINEERING_ROOT = PROJECT_ROOT / "数据格式与校验"
if str(DATA_ENGINEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ENGINEERING_ROOT))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import process_customer_year
from reader import AusgridReader
from seasonal_naive import evaluate_forecasts
from lightgbm_walk_forward import (
    FEATURE_COLUMNS,
    build_feature_frame,
    walk_forward_folds,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CUSTOMER_FEATURES = [
    "Customer",
    "Generator Capacity",
    "Postcode",
    "cluster_id",
]
GLOBAL_FEATURES = [*FEATURE_COLUMNS, *CUSTOMER_FEATURES]
CATEGORICAL_FEATURES = ["Customer", "Postcode", "cluster_id"]
PREDICTION_COLUMNS = (
    "lightgbm",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
)


def load_cluster_labels(
    cluster_path: Path,
) -> pd.DataFrame:
    clusters = pd.read_csv(cluster_path, encoding="utf-8-sig")
    required = {"customer_id", "cluster_id", "cluster_label"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"聚类标签缺少字段：{sorted(missing)}")
    return clusters


def build_global_feature_frame(
    reader: AusgridReader,
    *,
    source_file: str = "Solar home 2010-2011.csv",
    cluster_path: Path,
    customer_ids: Iterable[int] | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    spec = reader.file_specs[source_file]
    clusters = load_cluster_labels(cluster_path).set_index("customer_id")
    raw_year = reader.read_single_csv(source_file)
    available = sorted(raw_year[reader.customer_col].unique().tolist())
    selected = (
        available
        if customer_ids is None
        else sorted({int(value) for value in customer_ids})
    )
    invalid = sorted(set(selected) - set(available))
    if invalid:
        raise ValueError(f"{source_file} 中不存在客户：{invalid}")
    missing_clusters = sorted(set(selected) - set(clusters.index))
    if missing_clusters:
        raise ValueError(f"聚类标签缺少客户：{missing_clusters}")

    frames: list[pd.DataFrame] = []
    keep_columns = [
        "interval_start",
        "total_load_kwh",
        "seasonal_naive_daily",
        "seasonal_naive_weekly",
        *GLOBAL_FEATURES,
    ]
    for index, customer_id in enumerate(selected, start=1):
        clean = process_customer_year(
            reader,
            customer_id=customer_id,
            source_file=source_file,
            raw_year=raw_year,
        )
        frame = build_feature_frame(clean)
        frame["seasonal_naive_daily"] = frame["lag_48"]
        frame["seasonal_naive_weekly"] = frame["lag_336"]
        frame["Customer"] = np.int16(customer_id)
        frame["Generator Capacity"] = np.float32(
            clean[reader.capacity_col].iloc[0]
        )
        frame["Postcode"] = np.int16(clean[reader.postcode_col].iloc[0])
        frame["cluster_id"] = np.int8(
            clusters.loc[customer_id, "cluster_id"]
        )
        frames.append(frame[keep_columns].copy())

        if progress_every > 0 and (
            index % progress_every == 0 or index == len(selected)
        ):
            print(f"[进度] 全局特征：{index}/{len(selected)} 户")

    global_frame = pd.concat(frames, ignore_index=True)
    for column in GLOBAL_FEATURES:
        if column in CATEGORICAL_FEATURES:
            global_frame[column] = global_frame[column].astype("category")
        else:
            global_frame[column] = global_frame[column].astype("float32")
    return global_frame.reset_index(drop=True)


def train_global_fold(
    global_frame: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    random_state: int,
    num_threads: int = 8,
    objective: str = "regression_l1",
    num_boost_round: int = 800,
) -> tuple[lgb.Booster, pd.DataFrame, int, float]:
    train = global_frame.loc[
        global_frame["interval_start"].lt(test_start)
        & global_frame["total_load_kwh"].notna()
    ].dropna(subset=GLOBAL_FEATURES)
    if len(train) < 100_000:
        raise ValueError("全局 LightGBM 训练样本不足")

    validation_start = (
        train["interval_start"].max().normalize()
        - pd.Timedelta(days=13)
    )
    fit = train.loc[train["interval_start"].lt(validation_start)]
    validation = train.loc[train["interval_start"].ge(validation_start)]

    train_dataset = lgb.Dataset(
        fit[GLOBAL_FEATURES],
        label=fit["total_load_kwh"],
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=True,
    )
    validation_dataset = lgb.Dataset(
        validation[GLOBAL_FEATURES],
        label=validation["total_load_kwh"],
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_dataset,
        free_raw_data=True,
    )
    params = {
        "objective": objective,
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "feature_fraction": 0.85,
        "lambda_l2": 1.0,
        "max_bin": 63,
        "seed": random_state,
        "num_threads": num_threads,
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[validation_dataset],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    importance = pd.DataFrame(
        {
            "feature": GLOBAL_FEATURES,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    validation_prediction = model.predict(
        validation[GLOBAL_FEATURES],
        num_iteration=model.best_iteration or None,
    )
    validation_bias = float(
        (
            validation["total_load_kwh"].to_numpy()
            - validation_prediction
        ).mean()
    )
    return model, importance, len(train), validation_bias


def run_global_walk_forward(
    global_frame: pd.DataFrame,
    *,
    n_folds: int = 3,
    fold_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    folds = walk_forward_folds(
        global_frame.rename(columns={"interval_start": "interval_start"}),
        n_folds=n_folds,
        fold_days=fold_days,
    )
    fold_predictions: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, object]] = []

    for fold_number, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        model, importance, train_rows, validation_bias = train_global_fold(
            global_frame,
            test_start=fold["test_start"],
            random_state=100 + fold_number,
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
                "lightgbm": model.predict(
                    test[GLOBAL_FEATURES],
                    num_iteration=model.best_iteration or None,
                ),
                "seasonal_naive_daily": test[
                    "seasonal_naive_daily"
                ].to_numpy(),
                "seasonal_naive_weekly": test[
                    "seasonal_naive_weekly"
                ].to_numpy(),
            }
        )
        fold_predictions.append(predictions)

        importance["fold"] = fold_number
        importance_frames.append(importance)
        metrics = evaluate_forecasts(
            predictions,
            forecast_columns=PREDICTION_COLUMNS,
        )
        elapsed_seconds = time.perf_counter() - fold_started
        fold_reports.append(
            {
                "fold": fold_number,
                "test_start": fold["test_start"].date().isoformat(),
                "test_end": fold["test_end"].date().isoformat(),
                "training_rows": train_rows,
                "best_iteration": int(model.best_iteration or num_boost_round),
                "validation_bias_correction_kwh": round(
                    validation_bias,
                    6,
                ),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "metrics": metrics,
            }
        )
        print(
            f"[OK] Fold {fold_number}: {fold['test_start'].date()} 至 "
            f"{fold['test_end'].date()}，"
            f"LightGBM MAE={metrics['lightgbm']['mae_kwh']:.4f}"
        )

    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    overall_metrics = evaluate_forecasts(
        all_predictions,
        forecast_columns=PREDICTION_COLUMNS,
    )
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    customer_rows: list[dict[str, object]] = []
    for customer_id, subset in all_predictions.groupby("Customer"):
        metrics = evaluate_forecasts(
            subset,
            forecast_columns=PREDICTION_COLUMNS,
        )
        customer_rows.append(
            {
                "customer_id": int(customer_id),
                "lightgbm_mae_kwh": metrics["lightgbm"]["mae_kwh"],
                "lightgbm_rmse_kwh": metrics["lightgbm"]["rmse_kwh"],
                "daily_mae_kwh": metrics["seasonal_naive_daily"]["mae_kwh"],
                "daily_rmse_kwh": metrics["seasonal_naive_daily"]["rmse_kwh"],
                "weekly_mae_kwh": metrics["seasonal_naive_weekly"]["mae_kwh"],
                "weekly_rmse_kwh": metrics["seasonal_naive_weekly"]["rmse_kwh"],
            }
        )
    per_customer = pd.DataFrame(customer_rows).sort_values("customer_id")
    for baseline in ("daily", "weekly"):
        per_customer[f"improvement_vs_{baseline}_pct"] = (
            (
                per_customer[f"{baseline}_mae_kwh"]
                - per_customer["lightgbm_mae_kwh"]
            )
            / per_customer[f"{baseline}_mae_kwh"]
            * 100
        )

    report = {
        "customers": int(global_frame["Customer"].nunique()),
        "evaluation_points": len(all_predictions),
        "n_folds": n_folds,
        "fold_days": fold_days,
        "validation": "expanding-window global walk-forward",
        "total_elapsed_seconds": round(
            sum(float(fold["elapsed_seconds"]) for fold in fold_reports),
            3,
        ),
        "folds": fold_reports,
        "overall_metrics": overall_metrics,
        "per_customer_improvement": {
            "vs_daily_mean_pct": round(
                float(per_customer["improvement_vs_daily_pct"].mean()),
                6,
            ),
            "vs_daily_median_pct": round(
                float(per_customer["improvement_vs_daily_pct"].median()),
                6,
            ),
            "vs_weekly_mean_pct": round(
                float(per_customer["improvement_vs_weekly_pct"].mean()),
                6,
            ),
            "vs_weekly_median_pct": round(
                float(per_customer["improvement_vs_weekly_pct"].median()),
                6,
            ),
        },
    }
    return all_predictions, per_customer, importance, report


def plot_global_metrics(
    report: dict[str, object],
    *,
    output_path: Path,
) -> None:
    metrics = report["overall_metrics"]
    labels = {
        "lightgbm": "LightGBM",
        "seasonal_naive_daily": "昨日同时刻",
        "seasonal_naive_weekly": "上周同时刻",
    }
    colors = {
        "lightgbm": "#2F7657",
        "seasonal_naive_daily": "#24708F",
        "seasonal_naive_weekly": "#C4592E",
    }

    figure, (mae_axis, wape_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5.8),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (mae_axis, wape_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    names = list(labels)
    mae_axis.bar(
        [labels[name] for name in names],
        [metrics[name]["mae_kwh"] for name in names],
        color=[colors[name] for name in names],
        width=0.6,
    )
    mae_axis.set_title("MAE（越低越好）", loc="left", fontsize=14, fontweight="bold")
    mae_axis.set_ylabel("kWh / 半小时")

    wape_axis.bar(
        [labels[name] for name in names],
        [metrics[name]["wape_pct"] for name in names],
        color=[colors[name] for name in names],
        width=0.6,
    )
    wape_axis.set_title("WAPE（越低越好）", loc="left", fontsize=14, fontweight="bold")
    wape_axis.set_ylabel("%")
    wape_axis.tick_params(axis="x", rotation=12)

    figure.suptitle(
        f"{report['customers']} 户全局 LightGBM Walk-forward 结果",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.17, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
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
    )
    predictions, per_customer, importance, report = run_global_walk_forward(
        global_frame
    )

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    prediction_path = output_dir / "global_300_lightgbm_walk_forward_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    per_customer_path = output_dir / "global_300_lightgbm_per_customer.csv"
    per_customer.to_csv(per_customer_path, index=False, encoding="utf-8-sig")
    importance_path = output_dir / "global_300_lightgbm_importance.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    report_path = output_dir / "global_300_lightgbm_walk_forward_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_path = output_dir / "global_300_lightgbm_walk_forward_metrics.png"
    plot_global_metrics(report, output_path=figure_path)

    overall = report["overall_metrics"]["lightgbm"]
    print(
        f"[OK] 全局 LightGBM：MAE={overall['mae_kwh']:.4f}，"
        f"RMSE={overall['rmse_kwh']:.4f}，"
        f"WAPE={overall['wape_pct']:.2f}%"
    )
    print(f"[OK] 指标：{report_path}")
    print(f"[OK] 客户级结果：{per_customer_path}")
    print(f"[OK] 预测明细：{prediction_path}")
    print(f"[OK] 特征重要性：{importance_path}")
    print(f"[OK] 图：{figure_path}")


if __name__ == "__main__":
    main()
