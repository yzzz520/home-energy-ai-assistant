from __future__ import annotations

import json
import sys
from pathlib import Path

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
from seasonal_naive import (
    evaluate_forecasts,
    make_dayahead_forecasts,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_COLUMNS = [
    "slot_index",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "lag_48",
    "lag_336",
    "history_week_mean",
    "history_week_std",
    "history_week_min",
    "history_week_max",
]


def build_feature_frame(
    clean: pd.DataFrame,
    *,
    target_col: str = "total_load_kwh",
) -> pd.DataFrame:
    frame = clean.sort_values("interval_start").copy()
    timestamps = pd.to_datetime(frame["interval_start"])
    target = frame[target_col].astype(float)

    frame["lag_48"] = target.shift(48)
    frame["lag_336"] = target.shift(336)
    weekly_lags = pd.concat(
        [target.shift(48 * day) for day in range(1, 8)],
        axis=1,
    )
    frame["history_week_mean"] = weekly_lags.mean(axis=1)
    frame["history_week_std"] = weekly_lags.std(axis=1)
    frame["history_week_min"] = weekly_lags.min(axis=1)
    frame["history_week_max"] = weekly_lags.max(axis=1)

    slot = frame["slot_index"].astype(float)
    day_angle = 2 * np.pi * slot / 48
    weekday_angle = 2 * np.pi * timestamps.dt.dayofweek / 7
    month_angle = 2 * np.pi * timestamps.dt.month / 12
    year_angle = 2 * np.pi * timestamps.dt.dayofyear / 365.25
    frame["hour_sin"] = np.sin(day_angle)
    frame["hour_cos"] = np.cos(day_angle)
    frame["weekday_sin"] = np.sin(weekday_angle)
    frame["weekday_cos"] = np.cos(weekday_angle)
    frame["is_weekend"] = timestamps.dt.dayofweek.ge(5).astype("int8")
    frame["month_sin"] = np.sin(month_angle)
    frame["month_cos"] = np.cos(month_angle)
    frame["dayofyear_sin"] = np.sin(year_angle)
    frame["dayofyear_cos"] = np.cos(year_angle)
    return frame


def walk_forward_folds(
    feature_frame: pd.DataFrame,
    *,
    n_folds: int = 3,
    fold_days: int = 20,
) -> list[dict[str, pd.Timestamp]]:
    latest = pd.to_datetime(
        feature_frame["interval_start"]
    ).max().normalize()
    first_test = latest - pd.Timedelta(days=n_folds * fold_days - 1)
    folds: list[dict[str, pd.Timestamp]] = []
    for fold in range(n_folds):
        start = first_test + pd.Timedelta(days=fold * fold_days)
        end = start + pd.Timedelta(days=fold_days - 1)
        folds.append(
            {
                "test_start": start,
                "test_end": end + pd.Timedelta(hours=23, minutes=30),
            }
        )
    return folds


def train_fold_model(
    frame: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    target_col: str = "total_load_kwh",
    random_state: int = 42,
) -> tuple[lgb.Booster, pd.DataFrame]:
    train = frame.loc[
        frame["interval_start"].lt(test_start)
        & frame[target_col].notna()
    ].dropna(subset=FEATURE_COLUMNS)
    if len(train) < 500:
        raise ValueError("LightGBM 训练样本不足")

    validation_start = (
        train["interval_start"].max().normalize()
        - pd.Timedelta(days=13)
    )
    fit = train.loc[train["interval_start"].lt(validation_start)]
    validation = train.loc[train["interval_start"].ge(validation_start)]
    if len(fit) < 300 or len(validation) < 100:
        raise ValueError("训练集或验证集样本不足")

    train_dataset = lgb.Dataset(
        fit[FEATURE_COLUMNS],
        label=fit[target_col],
        free_raw_data=False,
    )
    validation_dataset = lgb.Dataset(
        validation[FEATURE_COLUMNS],
        label=validation[target_col],
        reference=train_dataset,
        free_raw_data=False,
    )
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "feature_fraction": 0.85,
        "lambda_l2": 0.5,
        "seed": random_state,
        "num_threads": 4,
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=500,
        valid_sets=[validation_dataset],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    return model, importance


def run_walk_forward(
    feature_frame: pd.DataFrame,
    *,
    n_folds: int = 3,
    fold_days: int = 20,
    target_col: str = "total_load_kwh",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    folds = walk_forward_folds(
        feature_frame,
        n_folds=n_folds,
        fold_days=fold_days,
    )
    fold_predictions: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    clean_series = feature_frame[["interval_start", target_col]]
    for fold_number, fold in enumerate(folds, start=1):
        model, importance = train_fold_model(
            feature_frame,
            test_start=fold["test_start"],
            target_col=target_col,
            random_state=42 + fold_number,
        )
        test = feature_frame.loc[
            feature_frame["interval_start"].between(
                fold["test_start"],
                fold["test_end"],
                inclusive="both",
            )
            & feature_frame[target_col].notna()
        ].dropna(subset=FEATURE_COLUMNS)
        predictions = pd.DataFrame(
            {
                "origin": fold["test_start"],
                "forecast_timestamp": test["interval_start"].to_numpy(),
                "actual": test[target_col].to_numpy(),
                "lightgbm": model.predict(test[FEATURE_COLUMNS]),
            }
        )

        naive = make_dayahead_forecasts(
            clean_series,
            timestamp_col="interval_start",
            target_col=target_col,
            test_start=fold["test_start"],
        )
        naive = naive.loc[
            naive["forecast_timestamp"].between(
                fold["test_start"],
                fold["test_end"],
                inclusive="both",
            ),
            [
                "forecast_timestamp",
                "seasonal_naive_daily",
                "seasonal_naive_weekly",
            ],
        ]
        predictions = predictions.merge(
            naive,
            on="forecast_timestamp",
            how="left",
            validate="one_to_one",
        )
        predictions["horizon_step"] = (
            (
                predictions["forecast_timestamp"]
                - predictions["origin"]
            ).dt.total_seconds()
            // 1800
            + 1
        ).astype(int)
        fold_predictions.append(predictions)

        importance["fold"] = fold_number
        importance_frames.append(importance)
        fold_metrics = evaluate_forecasts(
            predictions,
            forecast_columns=(
                "lightgbm",
                "seasonal_naive_daily",
                "seasonal_naive_weekly",
            ),
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "test_start": fold["test_start"].date().isoformat(),
                "test_end": fold["test_end"].date().isoformat(),
                "metrics": fold_metrics,
                "best_iteration": int(model.best_iteration or 500),
            }
        )

    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics = evaluate_forecasts(
        all_predictions,
        forecast_columns=(
            "lightgbm",
            "seasonal_naive_daily",
            "seasonal_naive_weekly",
        ),
    )
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    metadata = {
        "folds": fold_rows,
        "overall_metrics": metrics,
        "validation": "expanding-window walk-forward; 20 days per fold",
    }
    return all_predictions, importance, metadata


def plot_example(
    predictions: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    example_date = pd.to_datetime(
        predictions["forecast_timestamp"]
    ).min().normalize()
    sample = predictions.loc[
        pd.to_datetime(predictions["forecast_timestamp"]).dt.normalize().eq(
            example_date
        )
    ].sort_values("forecast_timestamp")

    hours = (
        pd.to_datetime(sample["forecast_timestamp"]).dt.hour
        + pd.to_datetime(sample["forecast_timestamp"]).dt.minute / 60
        + 0.25
    )
    figure, axis = plt.subplots(figsize=(13, 6.2))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    axis.plot(hours, sample["actual"], color="#1E2A32", linewidth=2.8, label="实际值")
    axis.plot(
        hours,
        sample["lightgbm"],
        color="#2F7657",
        linewidth=2.4,
        label="LightGBM",
    )
    axis.plot(
        hours,
        sample["seasonal_naive_daily"],
        color="#24708F",
        linewidth=2.0,
        linestyle="--",
        label="昨日同时刻",
    )
    axis.set_title(
        f"Walk-forward 验证示例 · {example_date.date()}",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_ylabel("半小时电量 (kWh)")
    axis.set_xlabel("本地时间")
    axis.set_xlim(0, 24)
    axis.set_xticks(range(0, 25, 2))
    axis.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 2)])
    axis.legend(loc="upper left", frameon=False)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    reader = AusgridReader()
    clean = process_customer_year(
        reader,
        customer_id=1,
        source_file="Solar home 2010-2011.csv",
    )
    features = build_feature_frame(clean)
    predictions, importance, metadata = run_walk_forward(features)

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    predictions_path = output_dir / "customer_1_lightgbm_walk_forward_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    importance_path = output_dir / "customer_1_lightgbm_walk_forward_importance.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    report_path = output_dir / "customer_1_lightgbm_walk_forward_metrics.json"
    report = {
        "customer_id": 1,
        "source_year": "2010-2011",
        "target": "total_load_kwh",
        **metadata,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_path = output_dir / "customer_1_lightgbm_walk_forward_example.png"
    plot_example(predictions, output_path=figure_path)

    metrics = metadata["overall_metrics"]
    print(
        f"[OK] LightGBM MAE={metrics['lightgbm']['mae_kwh']:.4f} kWh，"
        f"RMSE={metrics['lightgbm']['rmse_kwh']:.4f} kWh，"
        f"WAPE={metrics['lightgbm']['wape_pct']:.2f}%"
    )
    print(f"[OK] 指标：{report_path}")
    print(f"[OK] 预测：{predictions_path}")
    print(f"[OK] 特征重要性：{importance_path}")
    print(f"[OK] 示例图：{figure_path}")


if __name__ == "__main__":
    main()
