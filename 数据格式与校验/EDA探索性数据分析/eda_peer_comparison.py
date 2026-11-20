from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_load_curve import format_hour_axis
from pipeline import process_customer_year
from reader import AusgridReader, FileSpec


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def customer_gc_weekday_profile(clean: pd.DataFrame) -> pd.DataFrame:
    usable = clean.loc[
        clean["model_usable"]
        & ~clean["is_missing_day"]
        & ~clean["is_missing_interval"]
        & clean["source_date"].dt.dayofweek.lt(5)
    ].copy()
    if usable.empty:
        raise ValueError("没有可用于 GC 工作日曲线的数据")

    usable["gc_kw"] = usable["gc_kwh"] / 0.5
    profile = (
        usable.groupby("slot_index", observed=True)["gc_kw"]
        .mean()
        .rename("avg_gc_kw")
        .reset_index()
    )
    daily_energy = (
        usable.groupby("source_date", observed=True)["gc_kwh"]
        .sum()
        .mean()
    )
    profile["mean_daily_gc_kwh"] = float(daily_energy)
    profile["customer_id"] = int(clean["Customer"].iloc[0])
    profile["source_year"] = str(clean["source_year"].iloc[0])
    profile["plot_hour"] = profile["slot_index"] / 2 + 0.25
    return profile


def collect_customer_profiles(
    reader: AusgridReader,
    spec: FileSpec,
    *,
    customer_ids: Iterable[int] | None = None,
    progress_every: int = 50,
) -> pd.DataFrame:
    raw_year = reader.read_single_csv(spec.name)
    available = sorted(raw_year[reader.customer_col].unique().tolist())
    selected = (
        available
        if customer_ids is None
        else sorted({int(value) for value in customer_ids})
    )
    invalid = sorted(set(selected) - set(available))
    if invalid:
        raise ValueError(f"{spec.name} 中不存在客户：{invalid}")

    profiles: list[pd.DataFrame] = []
    for index, customer_id in enumerate(selected, start=1):
        clean = process_customer_year(
            reader,
            customer_id=customer_id,
            source_file=spec.name,
            raw_year=raw_year,
        )
        profiles.append(customer_gc_weekday_profile(clean))
        if progress_every > 0 and (
            index % progress_every == 0 or index == len(selected)
        ):
            print(f"[进度] GC 工作日曲线：{index}/{len(selected)} 户")

    return pd.concat(profiles, ignore_index=True)


def compare_to_population(
    profiles: pd.DataFrame,
    *,
    customer_id: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    matrix = profiles.pivot(
        index="customer_id",
        columns="slot_index",
        values="avg_gc_kw",
    ).sort_index()
    if customer_id not in matrix.index:
        raise ValueError(f"比较数据中没有 Customer={customer_id}")

    population_average = matrix.mean(axis=0)
    population_p10 = matrix.quantile(0.1, axis=0)
    population_p90 = matrix.quantile(0.9, axis=0)
    customer = matrix.loc[customer_id]

    comparison = pd.DataFrame(
        {
            "slot_index": matrix.columns,
            "population_avg_kw": population_average.to_numpy(),
            "population_p10_kw": population_p10.to_numpy(),
            "population_p90_kw": population_p90.to_numpy(),
            "customer_avg_kw": customer.to_numpy(),
        }
    )
    comparison["plot_hour"] = comparison["slot_index"] / 2 + 0.25

    customer_daily = (
        profiles.loc[profiles["customer_id"].eq(customer_id), "mean_daily_gc_kwh"]
        .drop_duplicates()
        .iloc[0]
    )
    all_daily = (
        profiles.groupby("customer_id", observed=True)["mean_daily_gc_kwh"]
        .first()
    )
    percentile = float((all_daily < customer_daily).mean() * 100)

    difference = customer.to_numpy() - population_average.to_numpy()
    correlation = float(np.corrcoef(customer, population_average)[0, 1])
    metrics = {
        "source_year": str(profiles["source_year"].iloc[0]),
        "customers": int(matrix.shape[0]),
        "customer_id": customer_id,
        "correlation_with_population": round(correlation, 6),
        "mae_vs_population_kw": round(float(np.abs(difference).mean()), 6),
        "rmse_vs_population_kw": round(float(np.sqrt(np.mean(difference**2))), 6),
        "customer_mean_daily_gc_kwh": round(float(customer_daily), 6),
        "population_mean_daily_gc_kwh": round(float(all_daily.mean()), 6),
        "customer_daily_energy_percentile": round(percentile, 3),
        "customer_peak_slot_start": (
            pd.Timestamp("2000-01-01")
            + pd.Timedelta(minutes=30 * int(customer.idxmax()))
        ).strftime("%H:%M"),
        "population_peak_slot_start": (
            pd.Timestamp("2000-01-01")
            + pd.Timedelta(minutes=30 * int(population_average.idxmax()))
        ).strftime("%H:%M"),
    }
    return comparison, metrics


def plot_peer_comparison(
    comparison: pd.DataFrame,
    metrics: dict[str, object],
    *,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7.8))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    axis.fill_between(
        comparison["plot_hour"],
        comparison["population_p10_kw"],
        comparison["population_p90_kw"],
        color="#BFD4CB",
        alpha=0.42,
        linewidth=0,
        label="300 户 10%-90% 分位",
    )
    axis.plot(
        comparison["plot_hour"],
        comparison["population_avg_kw"],
        color="#24708F",
        linewidth=3.0,
        label=f"{metrics['customers']} 户平均",
    )
    axis.plot(
        comparison["plot_hour"],
        comparison["customer_avg_kw"],
        color="#C4592E",
        linewidth=2.7,
        label=f"Customer {metrics['customer_id']}",
    )

    axis.set_title(
        "2010-2011 工作日 GC 曲线：Customer 1 与 300 户总体",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    axis.set_ylabel("半小时平均功率 (kW)")
    axis.set_xlabel("本地时间")
    format_hour_axis(axis)
    axis.legend(loc="upper left", frameon=False)

    summary = (
        f"相关性 {metrics['correlation_with_population']:.3f}   "
        f"MAE {metrics['mae_vs_population_kw']:.3f} kW   "
        f"RMSE {metrics['rmse_vs_population_kw']:.3f} kW   "
        f"日均 GC 分位 {metrics['customer_daily_energy_percentile']:.1f}%   "
        f"峰值时段 C1 {metrics['customer_peak_slot_start']} / "
        f"总体 {metrics['population_peak_slot_start']}"
    )
    figure.text(0.07, 0.025, summary, fontsize=11, color="#47545D")
    figure.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    reader = AusgridReader()
    source_file = "Solar home 2010-2011.csv"
    spec = reader.file_specs[source_file]
    profiles = collect_customer_profiles(reader, spec)
    comparison, metrics = compare_to_population(profiles, customer_id=1)

    output_dir = Path(__file__).with_name("reports") / "figures"
    figure_path = output_dir / (
        "customer_1_vs_300_population_gc_weekday_profile.png"
    )
    plot_peer_comparison(
        comparison,
        metrics,
        output_path=figure_path,
    )
    metrics_path = output_dir / (
        "customer_1_vs_300_population_gc_weekday_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[OK] Customer 1 与总体相关性 "
        f"{metrics['correlation_with_population']:.3f}，"
        f"日均 GC 分位 {metrics['customer_daily_energy_percentile']:.1f}%"
    )
    print(f"[OK] 图片：{figure_path}")
    print(f"[OK] 指标：{metrics_path}")


if __name__ == "__main__":
    main()
