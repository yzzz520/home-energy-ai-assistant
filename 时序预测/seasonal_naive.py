from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ENGINEERING_ROOT = PROJECT_ROOT / "数据格式与校验"
DATA_ENGINEERING_ROOT_TEXT = str(DATA_ENGINEERING_ROOT)
if DATA_ENGINEERING_ROOT_TEXT not in sys.path:
    sys.path.insert(0, DATA_ENGINEERING_ROOT_TEXT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import process_customer_year
from reader import AusgridReader


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def make_dayahead_forecasts(
    series: pd.DataFrame,
    *,
    timestamp_col: str = "interval_start",
    target_col: str = "total_load_kwh",
    test_start: str | pd.Timestamp,
    horizon_steps: int = 48,
    seasonal_lags: dict[str, int] | None = None,
) -> pd.DataFrame:
    if seasonal_lags is None:
        seasonal_lags = {
            "seasonal_naive_daily": 48,
            "seasonal_naive_weekly": 336,
        }

    data = series[[timestamp_col, target_col]].copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.sort_values(timestamp_col).set_index(timestamp_col)[target_col]
    if data.index.has_duplicates:
        raise ValueError("预测序列存在重复时间戳")

    test_start_ts = pd.Timestamp(test_start).normalize()
    test_end = data.index.max().normalize()
    if test_start_ts > test_end:
        raise ValueError("测试开始日期晚于数据结束日期")

    max_lag = max(seasonal_lags.values())
    earliest_needed = test_start_ts - pd.Timedelta(
        minutes=30 * max_lag
    )
    if earliest_needed < data.index.min():
        raise ValueError(
            f"季节性滞后需要 {earliest_needed} 的历史数据，"
            f"但数据从 {data.index.min()} 开始"
        )

    origins = pd.date_range(
        test_start_ts,
        test_end,
        freq="D",
    )
    frames: list[pd.DataFrame] = []
    for origin in origins:
        target_timestamps = pd.date_range(
            origin,
            periods=horizon_steps,
            freq="30min",
        )
        frame = pd.DataFrame(
            {
                "origin": origin,
                "forecast_timestamp": target_timestamps,
                "horizon_step": np.arange(1, horizon_steps + 1, dtype=int),
            }
        )
        for name, lag in seasonal_lags.items():
            source_timestamps = target_timestamps - pd.Timedelta(
                minutes=30 * lag
            )
            frame[name] = data.reindex(source_timestamps).to_numpy()
        frames.append(frame)

    forecasts = pd.concat(frames, ignore_index=True)
    forecasts["actual"] = data.reindex(
        forecasts["forecast_timestamp"]
    ).to_numpy()
    return forecasts


def evaluate_forecasts(
    forecasts: pd.DataFrame,
    forecast_columns: tuple[str, ...] = (
        "seasonal_naive_daily",
        "seasonal_naive_weekly",
    ),
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for column in forecast_columns:
        valid = forecasts[["actual", column]].dropna()
        if valid.empty:
            raise ValueError(f"{column} 没有可评估的预测值")
        error = valid[column] - valid["actual"]
        absolute = error.abs()
        actual_sum = valid["actual"].abs().sum()
        metrics[column] = {
            "rows": int(len(valid)),
            "mae_kwh": round(float(absolute.mean()), 6),
            "rmse_kwh": round(float(np.sqrt((error**2).mean())), 6),
            "wape_pct": round(
                float(absolute.sum() / actual_sum * 100)
                if actual_sum > 0
                else np.nan,
                6,
            ),
            "mean_bias_kwh": round(float(error.mean()), 6),
        }
    return metrics


def aggregate_forecast_metrics(
    forecasts: pd.DataFrame,
    forecast_columns: tuple[str, ...] = (
        "seasonal_naive_daily",
        "seasonal_naive_weekly",
    ),
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for column in forecast_columns:
        valid = forecasts[["actual", column]].dropna()
        if valid.empty:
            raise ValueError(f"{column} 没有可汇总的预测值")
        error = valid[column] - valid["actual"]
        absolute = error.abs()
        actual_sum = float(valid["actual"].abs().sum())
        metrics[column] = {
            "rows": int(len(valid)),
            "sum_abs_error": float(absolute.sum()),
            "sum_squared_error": float((error**2).sum()),
            "sum_actual": actual_sum,
            "sum_error": float(error.sum()),
        }
    return metrics


def plot_example_forecast(
    forecasts: pd.DataFrame,
    *,
    origin: str | pd.Timestamp,
    output_path: Path,
) -> None:
    sample = forecasts.loc[
        forecasts["origin"].eq(pd.Timestamp(origin).normalize())
    ].copy()
    if sample.empty:
        raise ValueError(f"没有预测起点 {origin} 的数据")

    hours = (
        sample["forecast_timestamp"].dt.hour
        + sample["forecast_timestamp"].dt.minute / 60
        + 0.25
    )
    figure, axis = plt.subplots(figsize=(13, 6.5))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    axis.plot(hours, sample["actual"], color="#1E2A32", linewidth=2.8, label="实际值")
    axis.plot(
        hours,
        sample["seasonal_naive_daily"],
        color="#24708F",
        linewidth=2.0,
        linestyle="--",
        label="昨日同时刻",
    )
    axis.plot(
        hours,
        sample["seasonal_naive_weekly"],
        color="#C4592E",
        linewidth=2.2,
        linestyle="-.",
        label="上周同时刻",
    )
    axis.set_title(
        f"日前季节性朴素预测 · 起点 {pd.Timestamp(origin).date()}",
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
    customer_id = 1
    clean = process_customer_year(
        reader,
        customer_id=customer_id,
        source_file="Solar home 2010-2011.csv",
    )
    max_date = clean["interval_start"].max().normalize()
    test_start = max_date - pd.Timedelta(days=59)

    forecasts = make_dayahead_forecasts(
        clean,
        test_start=test_start,
    )
    metrics = evaluate_forecasts(forecasts)

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    prediction_path = output_dir / (
        "customer_1_total_load_seasonal_naive_forecasts.csv"
    )
    forecasts.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    report = {
        "customer_id": customer_id,
        "source_year": "2010-2011",
        "target": "total_load_kwh",
        "horizon_steps": 48,
        "test_start": test_start.date().isoformat(),
        "test_end": max_date.date().isoformat(),
        "forecast_origins": int(forecasts["origin"].nunique()),
        "metrics": metrics,
    }
    metrics_path = output_dir / (
        "customer_1_total_load_seasonal_naive_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure_path = output_dir / (
        "customer_1_total_load_seasonal_naive_example.png"
    )
    plot_example_forecast(
        forecasts,
        origin=test_start,
        output_path=figure_path,
    )

    weekly = metrics["seasonal_naive_weekly"]
    print(
        f"[OK] 测试区间 {test_start.date()} 至 {max_date.date()}，"
        f"日前预测 {report['forecast_origins']} 天"
    )
    print(
        f"[OK] 上周同时刻基线：MAE={weekly['mae_kwh']:.4f} kWh，"
        f"RMSE={weekly['rmse_kwh']:.4f} kWh，"
        f"WAPE={weekly['wape_pct']:.2f}%"
    )
    print(f"[OK] 预测明细：{prediction_path}")
    print(f"[OK] 指标：{metrics_path}")
    print(f"[OK] 示例图：{figure_path}")


if __name__ == "__main__":
    main()
