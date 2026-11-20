from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ENGINEERING_ROOT = PROJECT_ROOT / "数据格式与校验"
if str(DATA_ENGINEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ENGINEERING_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import process_customer_year
from reader import AusgridReader
from seasonal_naive import (
    aggregate_forecast_metrics,
    evaluate_forecasts,
    make_dayahead_forecasts,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FORECAST_COLUMNS = (
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
)


def finalize_global_metrics(
    accumulated: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name, values in accumulated.items():
        rows = int(values["rows"])
        metrics[name] = {
            "rows": rows,
            "mae_kwh": round(values["sum_abs_error"] / rows, 6),
            "rmse_kwh": round(
                np.sqrt(values["sum_squared_error"] / rows),
                6,
            ),
            "wape_pct": round(
                values["sum_abs_error"] / values["sum_actual"] * 100
                if values["sum_actual"] > 0
                else np.nan,
                6,
            ),
            "mean_bias_kwh": round(values["sum_error"] / rows, 6),
        }
    return metrics


def build_global_walk_forward(
    reader: AusgridReader,
    *,
    source_file: str = "Solar home 2010-2011.csv",
    test_days: int = 60,
    customer_ids: Iterable[int] | None = None,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    spec = reader.file_specs[source_file]
    raw_year = reader.read_single_csv(source_file)
    all_customers = sorted(raw_year[reader.customer_col].unique().tolist())
    customers = (
        all_customers
        if customer_ids is None
        else sorted({int(value) for value in customer_ids})
    )
    invalid = sorted(set(customers) - set(all_customers))
    if invalid:
        raise ValueError(f"{source_file} 中不存在客户：{invalid}")
    raw_start = raw_year[reader.date_col].min().normalize()
    raw_end = raw_year[reader.date_col].max().normalize()
    test_start = raw_end - pd.Timedelta(days=test_days - 1)
    if test_start <= raw_start:
        raise ValueError("测试窗口早于可用历史数据")

    accumulated = {
        name: {
            "rows": 0,
            "sum_abs_error": 0.0,
            "sum_squared_error": 0.0,
            "sum_actual": 0.0,
            "sum_error": 0.0,
        }
        for name in FORECAST_COLUMNS
    }
    horizon_abs = {
        name: np.zeros(48, dtype=float) for name in FORECAST_COLUMNS
    }
    horizon_count = {
        name: np.zeros(48, dtype=int) for name in FORECAST_COLUMNS
    }
    customer_rows: list[dict[str, object]] = []

    for index, customer_id in enumerate(customers, start=1):
        clean = process_customer_year(
            reader,
            customer_id=customer_id,
            source_file=source_file,
            raw_year=raw_year,
        )
        forecasts = make_dayahead_forecasts(
            clean,
            test_start=test_start,
        )
        customer_metrics = evaluate_forecasts(
            forecasts,
            forecast_columns=FORECAST_COLUMNS,
        )
        customer_row: dict[str, object] = {"customer_id": customer_id}
        for name in FORECAST_COLUMNS:
            prefix = "daily" if name.endswith("daily") else "weekly"
            customer_row[f"{prefix}_mae_kwh"] = customer_metrics[name]["mae_kwh"]
            customer_row[f"{prefix}_rmse_kwh"] = customer_metrics[name]["rmse_kwh"]
            customer_row[f"{prefix}_wape_pct"] = customer_metrics[name]["wape_pct"]
        customer_rows.append(customer_row)

        aggregated = aggregate_forecast_metrics(
            forecasts,
            forecast_columns=FORECAST_COLUMNS,
        )
        for name in FORECAST_COLUMNS:
            for key in accumulated[name]:
                accumulated[name][key] += aggregated[name][key]

            valid = forecasts[["actual", name, "horizon_step"]].dropna()
            error = (valid[name] - valid["actual"]).abs()
            sums = error.groupby(valid["horizon_step"]).sum()
            counts = error.groupby(valid["horizon_step"]).size()
            steps = sums.index.to_numpy(dtype=int) - 1
            horizon_abs[name][steps] += sums.to_numpy()
            horizon_count[name][steps] += counts.to_numpy()

        if progress_every > 0 and (
            index % progress_every == 0 or index == len(customers)
        ):
            print(f"[进度] 300 户 walk-forward：{index}/{len(customers)} 户")

    global_metrics = finalize_global_metrics(accumulated)
    per_customer = pd.DataFrame(customer_rows).sort_values("customer_id")
    horizon_rows: list[dict[str, object]] = []
    for name in FORECAST_COLUMNS:
        mae_by_horizon = horizon_abs[name] / np.maximum(horizon_count[name], 1)
        for step, mae in enumerate(mae_by_horizon, start=1):
            horizon_rows.append(
                {
                    "baseline": name,
                    "horizon_step": step,
                    "horizon_minutes": step * 30,
                    "mae_kwh": round(float(mae), 6),
                }
            )
    horizon_metrics = pd.DataFrame(horizon_rows)

    report = {
        "source_year": spec.source_year,
        "source_file": source_file,
        "target": "total_load_kwh",
        "customers": len(customers),
        "forecast_origins_per_customer": test_days,
        "horizon_steps": 48,
        "test_start": test_start.date().isoformat(),
        "test_end": raw_end.date().isoformat(),
        "evaluation_points": int(global_metrics["seasonal_naive_daily"]["rows"]),
        "validation": "daily walk-forward origins; no random split",
        "global_metrics": global_metrics,
        "per_customer_distribution": {
            name: {
                "mean_mae_kwh": round(
                    float(per_customer[
                        f"{'daily' if name.endswith('daily') else 'weekly'}_mae_kwh"
                    ].mean()),
                    6,
                ),
                "median_mae_kwh": round(
                    float(per_customer[
                        f"{'daily' if name.endswith('daily') else 'weekly'}_mae_kwh"
                    ].median()),
                    6,
                ),
                "p90_mae_kwh": round(
                    float(per_customer[
                        f"{'daily' if name.endswith('daily') else 'weekly'}_mae_kwh"
                    ].quantile(0.9)),
                    6,
                ),
            }
            for name in FORECAST_COLUMNS
        },
    }
    return per_customer, report, horizon_metrics


def plot_horizon_errors(
    horizon_metrics: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.5))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    labels = {
        "seasonal_naive_daily": ("昨日同时刻", "#24708F"),
        "seasonal_naive_weekly": ("上周同时刻", "#C4592E"),
    }
    for baseline, (label, color) in labels.items():
        subset = horizon_metrics.loc[horizon_metrics["baseline"].eq(baseline)]
        axis.plot(
            subset["horizon_minutes"] / 60,
            subset["mae_kwh"],
            color=color,
            linewidth=2.5,
            label=label,
        )

    axis.set_title(
        "300 户 Walk-forward 季节性朴素基线：误差随预测步长变化",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_xlabel("预测步长（小时）")
    axis.set_ylabel("MAE（kWh / 半小时）")
    axis.set_xlim(0, 24)
    axis.set_xticks(range(0, 25, 2))
    axis.legend(loc="upper left", frameon=False)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    reader = AusgridReader()
    per_customer, report, horizon_metrics = build_global_walk_forward(reader)

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    metrics_path = output_dir / (
        "global_300_total_load_seasonal_naive_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    per_customer_path = output_dir / (
        "global_300_total_load_seasonal_naive_per_customer.csv"
    )
    per_customer.to_csv(
        per_customer_path,
        index=False,
        encoding="utf-8-sig",
    )
    horizon_path = output_dir / (
        "global_300_total_load_seasonal_naive_horizon_metrics.csv"
    )
    horizon_metrics.to_csv(
        horizon_path,
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = output_dir / (
        "global_300_total_load_seasonal_naive_horizon_error.png"
    )
    plot_horizon_errors(horizon_metrics, output_path=figure_path)

    daily = report["global_metrics"]["seasonal_naive_daily"]
    weekly = report["global_metrics"]["seasonal_naive_weekly"]
    print(
        f"[OK] {report['customers']} 户，{report['evaluation_points']:,} 个评估点"
    )
    print(
        f"[OK] 昨日同时刻：MAE={daily['mae_kwh']:.4f}，"
        f"RMSE={daily['rmse_kwh']:.4f}，WAPE={daily['wape_pct']:.2f}%"
    )
    print(
        f"[OK] 上周同时刻：MAE={weekly['mae_kwh']:.4f}，"
        f"RMSE={weekly['rmse_kwh']:.4f}，WAPE={weekly['wape_pct']:.2f}%"
    )
    print(f"[OK] 全局指标：{metrics_path}")
    print(f"[OK] 客户级指标：{per_customer_path}")
    print(f"[OK] 步长误差：{horizon_path}")
    print(f"[OK] 图：{figure_path}")


if __name__ == "__main__":
    main()
