from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from pipeline import process_customer_year
from reader import AusgridReader


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SEASON_ORDER = ["夏季", "秋季", "冬季", "春季"]


def assign_season(dates: pd.Series) -> pd.Series:
    month = pd.to_datetime(dates).dt.month
    return pd.Series(
        np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            SEASON_ORDER,
            default="未知",
        ),
        index=dates.index,
        dtype="string",
    )


def format_hour_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 24)
    axis.set_xticks(range(0, 25, 2))
    axis.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda value, _: f"{int(value):02d}:00")
    )


def prepare_daily_curve(
    clean: pd.DataFrame,
    *,
    source_date: str,
) -> pd.DataFrame:
    target_date = pd.Timestamp(source_date).normalize()
    day = clean.loc[clean["source_date"].eq(target_date)].copy()
    if day.empty:
        raise ValueError(f"没有 {source_date} 的数据")
    if day["is_missing_day"].any():
        raise ValueError(f"{source_date} 是缺失日，不能绘制负荷曲线")
    if len(day) != 48:
        raise ValueError(f"{source_date} 应有 48 个时间点，实际 {len(day)}")

    day = day.sort_values("interval_start").reset_index(drop=True)
    day["plot_hour"] = (
        day["interval_start"].dt.hour
        + day["interval_start"].dt.minute / 60
        + 0.25
    )
    day["gc_kw"] = day["gc_kwh"] / 0.5
    day["cl_kw"] = day["cl_kwh"] / 0.5
    day["total_load_kw"] = day["total_load_kwh"] / 0.5
    day["pv_kw"] = day["gg_kwh"] / 0.5
    day["net_grid_kw"] = day["net_grid_kwh"] / 0.5
    return day


def build_average_daily_profile(clean: pd.DataFrame) -> pd.DataFrame:
    usable = clean.loc[
        clean["model_usable"]
        & ~clean["is_missing_day"]
        & ~clean["is_missing_interval"]
    ].copy()
    if usable.empty:
        raise ValueError("没有可用于平均日曲线的数据")

    usable["day_type"] = np.where(
        usable["source_date"].dt.dayofweek.ge(5),
        "周末",
        "工作日",
    )
    usable["total_load_kw"] = usable["total_load_kwh"] / 0.5
    days_per_type = (
        usable.groupby("day_type", observed=True)["source_date"]
        .nunique()
        .to_dict()
    )

    profile = (
        usable.groupby(["day_type", "slot_index"], observed=True)["total_load_kw"]
        .agg(
            avg_load_kw="mean",
            p25_load_kw=lambda values: values.quantile(0.25),
            p75_load_kw=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    profile["day_count"] = profile["day_type"].map(days_per_type)
    profile["plot_hour"] = profile["slot_index"] / 2 + 0.25
    return profile


def average_profile_metrics(
    clean: pd.DataFrame,
    profile: pd.DataFrame,
) -> dict[str, object]:
    usable = clean.loc[
        clean["model_usable"]
        & ~clean["is_missing_day"]
        & ~clean["is_missing_interval"]
    ].copy()
    usable["day_type"] = np.where(
        usable["source_date"].dt.dayofweek.ge(5),
        "周末",
        "工作日",
    )
    daily_energy = (
        usable.groupby(["source_date", "day_type"], observed=True)["total_load_kwh"]
        .sum()
        .reset_index()
    )
    daily_mean = (
        daily_energy.groupby("day_type", observed=True)["total_load_kwh"]
        .mean()
        .to_dict()
    )

    metrics: dict[str, object] = {
        "customer_id": int(clean["Customer"].iloc[0]),
        "source_year": str(clean["source_year"].iloc[0]),
    }
    for day_type in ("工作日", "周末"):
        subset = profile.loc[profile["day_type"].eq(day_type)]
        peak = subset.loc[subset["avg_load_kw"].idxmax()]
        metrics[day_type] = {
            "days": int(peak["day_count"]),
            "mean_daily_energy_kwh": round(float(daily_mean[day_type]), 6),
            "peak_average_kw": round(float(peak["avg_load_kw"]), 6),
            "peak_slot_start": (
                pd.Timestamp("2000-01-01")
                + pd.Timedelta(minutes=30 * int(peak["slot_index"]))
            ).strftime("%H:%M"),
        }
    return metrics


def plot_average_daily_profile(
    profile: pd.DataFrame,
    *,
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7.5))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    styles = {
        "工作日": ("#24708F", "#B8D8E5"),
        "周末": ("#C06A1E", "#F0C99E"),
    }
    for day_type, (line_color, band_color) in styles.items():
        subset = profile.loc[profile["day_type"].eq(day_type)]
        axis.fill_between(
            subset["plot_hour"],
            subset["p25_load_kw"],
            subset["p75_load_kw"],
            color=band_color,
            alpha=0.35,
            linewidth=0,
        )
        axis.plot(
            subset["plot_hour"],
            subset["avg_load_kw"],
            color=line_color,
            linewidth=2.8,
            label=(
                f"{day_type}均值 "
                f"({metrics[day_type]['days']} 天)"
            ),
        )

    axis.set_title(
        "Customer 1 · 2010-2011 年平均日负荷曲线",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    axis.set_ylabel("半小时平均功率 (kW)")
    axis.set_xlabel("本地时间")
    format_hour_axis(axis)
    axis.legend(loc="upper left", frameon=False)

    weekday = metrics["工作日"]
    weekend = metrics["周末"]
    summary = (
        f"工作日日均用电 {weekday['mean_daily_energy_kwh']:.2f} kWh，"
        f"峰值均值 {weekday['peak_average_kw']:.2f} kW（{weekday['peak_slot_start']}）    "
        f"周末日均用电 {weekend['mean_daily_energy_kwh']:.2f} kWh，"
        f"峰值均值 {weekend['peak_average_kw']:.2f} kW（{weekend['peak_slot_start']}）"
    )
    figure.text(0.07, 0.025, summary, fontsize=11, color="#47545D")
    figure.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _seasonal_weekday_frame(clean: pd.DataFrame) -> pd.DataFrame:
    usable = clean.loc[
        clean["model_usable"]
        & ~clean["is_missing_day"]
        & ~clean["is_missing_interval"]
        & clean["source_date"].dt.dayofweek.lt(5)
    ].copy()
    if usable.empty:
        raise ValueError("没有可用于季节性工作日曲线的数据")
    usable["season"] = assign_season(usable["source_date"])
    return usable


def build_seasonal_weekday_profile(clean: pd.DataFrame) -> pd.DataFrame:
    usable = _seasonal_weekday_frame(clean)
    usable["total_load_kw"] = usable["total_load_kwh"] / 0.5
    days_per_season = (
        usable.groupby("season", observed=True)["source_date"]
        .nunique()
        .to_dict()
    )
    profile = (
        usable.groupby(["season", "slot_index"], observed=True)["total_load_kw"]
        .mean()
        .rename("avg_load_kw")
        .reset_index()
    )
    profile["day_count"] = profile["season"].map(days_per_season)
    profile["plot_hour"] = profile["slot_index"] / 2 + 0.25
    profile["season"] = pd.Categorical(
        profile["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )
    return profile.sort_values(["season", "slot_index"]).reset_index(drop=True)


def seasonal_profile_metrics(
    clean: pd.DataFrame,
    profile: pd.DataFrame,
) -> dict[str, object]:
    usable = _seasonal_weekday_frame(clean)
    daily_energy = (
        usable.groupby(["source_date", "season"], observed=True)["total_load_kwh"]
        .sum()
        .reset_index()
    )
    daily_mean = (
        daily_energy.groupby("season", observed=True)["total_load_kwh"]
        .mean()
        .to_dict()
    )

    metrics: dict[str, object] = {
        "customer_id": int(clean["Customer"].iloc[0]),
        "source_year": str(clean["source_year"].iloc[0]),
        "day_scope": "工作日",
    }
    for season in SEASON_ORDER:
        subset = profile.loc[profile["season"].eq(season)]
        peak = subset.loc[subset["avg_load_kw"].idxmax()]
        metrics[season] = {
            "days": int(peak["day_count"]),
            "mean_daily_energy_kwh": round(float(daily_mean[season]), 6),
            "peak_average_kw": round(float(peak["avg_load_kw"]), 6),
            "peak_slot_start": (
                pd.Timestamp("2000-01-01")
                + pd.Timedelta(minutes=30 * int(peak["slot_index"]))
            ).strftime("%H:%M"),
        }
    return metrics


def build_seasonal_gc_weekday_profile(clean: pd.DataFrame) -> pd.DataFrame:
    usable = _seasonal_weekday_frame(clean)
    usable["gc_kw"] = usable["gc_kwh"] / 0.5
    days_per_season = (
        usable.groupby("season", observed=True)["source_date"]
        .nunique()
        .to_dict()
    )
    profile = (
        usable.groupby(["season", "slot_index"], observed=True)["gc_kw"]
        .mean()
        .rename("avg_gc_kw")
        .reset_index()
    )
    profile["day_count"] = profile["season"].map(days_per_season)
    profile["plot_hour"] = profile["slot_index"] / 2 + 0.25
    profile["season"] = pd.Categorical(
        profile["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )
    return profile.sort_values(["season", "slot_index"]).reset_index(drop=True)


def gc_cl_seasonal_metrics(
    clean: pd.DataFrame,
    profile: pd.DataFrame,
) -> dict[str, object]:
    usable = _seasonal_weekday_frame(clean)
    daily_energy = (
        usable.groupby(["source_date", "season"], observed=True)[
            ["gc_kwh", "cl_kwh", "total_load_kwh"]
        ]
        .sum()
        .reset_index()
    )
    daily_mean = (
        daily_energy.groupby("season", observed=True)[
            ["gc_kwh", "cl_kwh", "total_load_kwh"]
        ]
        .mean()
    )

    metrics: dict[str, object] = {
        "customer_id": int(clean["Customer"].iloc[0]),
        "source_year": str(clean["source_year"].iloc[0]),
        "day_scope": "工作日",
        "profile": "GC-only",
    }
    for season in SEASON_ORDER:
        subset = profile.loc[profile["season"].eq(season)]
        peak = subset.loc[subset["avg_gc_kw"].idxmax()]
        gc_mean = float(daily_mean.loc[season, "gc_kwh"])
        cl_mean = float(daily_mean.loc[season, "cl_kwh"])
        total_mean = float(daily_mean.loc[season, "total_load_kwh"])
        metrics[season] = {
            "days": int(peak["day_count"]),
            "mean_gc_energy_kwh": round(gc_mean, 6),
            "mean_cl_energy_kwh": round(cl_mean, 6),
            "mean_total_energy_kwh": round(total_mean, 6),
            "cl_share_pct": round(cl_mean / total_mean * 100, 6),
            "gc_peak_average_kw": round(float(peak["avg_gc_kw"]), 6),
            "gc_peak_slot_start": (
                pd.Timestamp("2000-01-01")
                + pd.Timedelta(minutes=30 * int(peak["slot_index"]))
            ).strftime("%H:%M"),
        }
    return metrics


def plot_seasonal_gc_weekday_profile(
    profile: pd.DataFrame,
    *,
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7.8))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    colors = {
        "夏季": "#C4592E",
        "秋季": "#9A7611",
        "冬季": "#24708F",
        "春季": "#2F7657",
    }
    for season in SEASON_ORDER:
        subset = profile.loc[profile["season"].eq(season)]
        axis.plot(
            subset["plot_hour"],
            subset["avg_gc_kw"],
            color=colors[season],
            linewidth=2.7,
            label=f"{season} GC（{metrics[season]['days']} 天）",
        )
        peak = subset.loc[subset["avg_gc_kw"].idxmax()]
        axis.scatter(
            peak["plot_hour"],
            peak["avg_gc_kw"],
            color=colors[season],
            s=42,
            zorder=4,
        )

    axis.set_title(
        "Customer 1 · 2010-2011 分季节工作日 GC-only 平均负荷",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    axis.set_ylabel("半小时平均功率 (kW)")
    axis.set_xlabel("本地时间")
    format_hour_axis(axis)
    axis.legend(loc="upper left", ncol=2, frameon=False)

    summary = "   ".join(
        f"{season} GC {metrics[season]['mean_gc_energy_kwh']:.1f} kWh/"
        f"CL {metrics[season]['mean_cl_energy_kwh']:.1f} kWh"
        for season in SEASON_ORDER
    )
    figure.text(
        0.06,
        0.025,
        f"工作日日均：{summary}",
        fontsize=11,
        color="#47545D",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_seasonal_weekday_profile(
    profile: pd.DataFrame,
    *,
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7.8))
    figure.patch.set_facecolor("#F7F8F5")
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    colors = {
        "夏季": "#C4592E",
        "秋季": "#9A7611",
        "冬季": "#24708F",
        "春季": "#2F7657",
    }
    for season in SEASON_ORDER:
        subset = profile.loc[profile["season"].eq(season)]
        axis.plot(
            subset["plot_hour"],
            subset["avg_load_kw"],
            color=colors[season],
            linewidth=2.7,
            label=f"{season}工作日（{metrics[season]['days']} 天）",
        )
        peak = subset.loc[subset["avg_load_kw"].idxmax()]
        axis.scatter(
            peak["plot_hour"],
            peak["avg_load_kw"],
            color=colors[season],
            s=42,
            zorder=4,
        )

    axis.set_title(
        "Customer 1 · 2010-2011 分季节工作日平均日负荷",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    axis.set_ylabel("半小时平均功率 (kW)")
    axis.set_xlabel("本地时间")
    format_hour_axis(axis)
    axis.legend(loc="upper left", ncol=2, frameon=False)

    summary = "   ".join(
        f"{season} {metrics[season]['mean_daily_energy_kwh']:.1f} kWh/"
        f"{metrics[season]['peak_average_kw']:.2f} kW"
        for season in SEASON_ORDER
    )
    figure.text(
        0.07,
        0.025,
        f"工作日日均用电 / 平均峰值：{summary}",
        fontsize=11,
        color="#47545D",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def daily_metrics(day: pd.DataFrame) -> dict[str, float]:
    return {
        "total_load_kwh": round(float(day["total_load_kwh"].sum()), 6),
        "gc_kwh": round(float(day["gc_kwh"].sum()), 6),
        "cl_kwh": round(float(day["cl_kwh"].sum()), 6),
        "pv_generation_kwh": round(float(day["gg_kwh"].sum()), 6),
        "import_kwh": round(float(day["import_kwh"].sum()), 6),
        "export_kwh": round(float(day["export_kwh"].sum()), 6),
        "peak_load_kw": round(float(day["total_load_kw"].max()), 6),
        "peak_pv_kw": round(float(day["pv_kw"].max()), 6),
        "max_import_kw": round(float(day["net_grid_kw"].max()), 6),
        "max_export_kw": round(float((-day["net_grid_kw"]).max()), 6),
    }


def plot_daily_curve(
    day: pd.DataFrame,
    *,
    customer_id: int,
    source_date: str,
    output_path: Path,
) -> dict[str, float]:
    metrics = daily_metrics(day)
    figure, (load_ax, energy_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1]},
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (load_ax, energy_ax):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    load_ax.plot(
        day["plot_hour"],
        day["total_load_kw"],
        color="#1E2A32",
        linewidth=2.8,
        label="总负荷",
    )
    load_ax.plot(
        day["plot_hour"],
        day["gc_kw"],
        color="#24708F",
        linewidth=2.0,
        label="GC 常规用电",
    )
    load_ax.plot(
        day["plot_hour"],
        day["cl_kw"],
        color="#C06A1E",
        linewidth=2.0,
        label="CL 控负荷",
    )
    load_ax.fill_between(
        day["plot_hour"],
        day["gc_kw"],
        day["total_load_kw"],
        color="#E9B979",
        alpha=0.25,
    )
    load_ax.set_ylabel("半小时平均功率 (kW)")
    load_ax.set_title(
        f"Customer {customer_id} · {source_date} 日负荷曲线",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    load_ax.legend(loc="upper left", ncol=3, frameon=False)

    energy_ax.fill_between(
        day["plot_hour"],
        0,
        day["pv_kw"],
        color="#2F7657",
        alpha=0.28,
        label="光伏发电",
    )
    energy_ax.plot(
        day["plot_hour"],
        day["pv_kw"],
        color="#2F7657",
        linewidth=2.0,
    )
    energy_ax.plot(
        day["plot_hour"],
        day["net_grid_kw"],
        color="#A44542",
        linewidth=2.2,
        linestyle="--",
        label="并网净负荷",
    )
    energy_ax.axhline(0, color="#617078", linewidth=1)
    energy_ax.set_ylabel("功率 (kW)")
    energy_ax.set_xlabel("本地时间")
    energy_ax.legend(loc="upper left", ncol=2, frameon=False)

    for axis in (load_ax, energy_ax):
        format_hour_axis(axis)

    summary_text = (
        f"总用电 {metrics['total_load_kwh']:.2f} kWh   "
        f"光伏 {metrics['pv_generation_kwh']:.2f} kWh   "
        f"购电 {metrics['import_kwh']:.2f} kWh   "
        f"送电 {metrics['export_kwh']:.2f} kWh   "
        f"峰值负荷 {metrics['peak_load_kw']:.2f} kW"
    )
    figure.text(0.08, 0.018, summary_text, fontsize=11, color="#47545D")
    figure.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.09, hspace=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)

    metrics.update(
        {
            "customer_id": customer_id,
            "source_date": source_date,
            "figure": str(output_path),
        }
    )
    return metrics


def main() -> None:
    reader = AusgridReader()
    customer_id = 1
    source_date = "2010-07-01"
    clean = process_customer_year(
        reader,
        customer_id=customer_id,
        source_file="Solar home 2010-2011.csv",
    )
    day = prepare_daily_curve(clean, source_date=source_date)

    figure_dir = Path(__file__).with_name("reports") / "figures"
    figure_path = figure_dir / (
        f"customer_{customer_id}_{source_date}_daily_load_curve.png"
    )
    metrics = plot_daily_curve(
        day,
        customer_id=customer_id,
        source_date=source_date,
        output_path=figure_path,
    )
    metrics_path = figure_dir / (
        f"customer_{customer_id}_{source_date}_daily_load_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[OK] {source_date}: 总用电 {metrics['total_load_kwh']:.2f} kWh，"
        f"光伏 {metrics['pv_generation_kwh']:.2f} kWh，"
        f"峰值负荷 {metrics['peak_load_kw']:.2f} kW"
    )
    print(f"[OK] 图片：{figure_path}")
    print(f"[OK] 指标：{metrics_path}")

    profile = build_average_daily_profile(clean)
    profile_metrics = average_profile_metrics(clean, profile)
    profile_path = figure_dir / (
        f"customer_{customer_id}_2010-2011_average_daily_load_profile.png"
    )
    plot_average_daily_profile(
        profile,
        metrics=profile_metrics,
        output_path=profile_path,
    )
    profile_metrics_path = figure_dir / (
        f"customer_{customer_id}_2010-2011_average_daily_load_metrics.json"
    )
    profile_metrics_path.write_text(
        json.dumps(profile_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 全年平均日负荷图：{profile_path}")
    print(f"[OK] 全年平均日负荷指标：{profile_metrics_path}")

    seasonal_profile = build_seasonal_weekday_profile(clean)
    seasonal_metrics = seasonal_profile_metrics(clean, seasonal_profile)
    seasonal_path = figure_dir / (
        f"customer_{customer_id}_2010-2011_seasonal_weekday_load_profile.png"
    )
    plot_seasonal_weekday_profile(
        seasonal_profile,
        metrics=seasonal_metrics,
        output_path=seasonal_path,
    )
    seasonal_metrics_path = figure_dir / (
        f"customer_{customer_id}_2010-2011_seasonal_weekday_metrics.json"
    )
    seasonal_metrics_path.write_text(
        json.dumps(seasonal_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 分季节工作日负荷图：{seasonal_path}")
    print(f"[OK] 分季节工作日指标：{seasonal_metrics_path}")

    gc_seasonal_profile = build_seasonal_gc_weekday_profile(clean)
    gc_seasonal_metrics = gc_cl_seasonal_metrics(clean, gc_seasonal_profile)
    gc_seasonal_path = figure_dir / (
        f"customer_{customer_id}_2010-2011_seasonal_weekday_gc_profile.png"
    )
    plot_seasonal_gc_weekday_profile(
        gc_seasonal_profile,
        metrics=gc_seasonal_metrics,
        output_path=gc_seasonal_path,
    )
    gc_seasonal_metrics_path = figure_dir / (
        f"customer_{customer_id}_2010-2011_seasonal_weekday_gc_metrics.json"
    )
    gc_seasonal_metrics_path.write_text(
        json.dumps(gc_seasonal_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 分季节 GC-only 工作日负荷图：{gc_seasonal_path}")
    print(f"[OK] 分季节 GC/CL 指标：{gc_seasonal_metrics_path}")


if __name__ == "__main__":
    main()
