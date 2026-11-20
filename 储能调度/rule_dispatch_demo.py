from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from battery import load_battery_parameters
from power_balance import build_rule_dispatch, summarize_balance


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_rule_dispatch(frame: pd.DataFrame, output_path: Path) -> None:
    hours = (
        pd.to_datetime(frame["interval_start"]).dt.hour
        + pd.to_datetime(frame["interval_start"]).dt.minute / 60
        + 0.25
    )
    figure, (power_axis, soc_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]},
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (power_axis, soc_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    power_axis.plot(
        hours,
        frame["total_load_kw"],
        color="#1E2A32",
        linewidth=2.7,
        label="家庭负荷",
    )
    power_axis.plot(
        hours,
        frame["pv_kw"],
        color="#2F7657",
        linewidth=2.2,
        label="光伏发电",
    )
    power_axis.plot(
        hours,
        frame["battery_net_ac_kw"],
        color="#C4592E",
        linewidth=2.2,
        label="电池净功率（放电为正）",
    )
    power_axis.plot(
        hours,
        frame["grid_net_kw"],
        color="#24708F",
        linewidth=2.4,
        linestyle="--",
        label="并网净功率（购电为正）",
    )
    power_axis.axhline(0, color="#617078", linewidth=1)
    power_axis.set_ylabel("功率 (kW)")
    power_axis.set_title(
        "规则调度：光伏余量充电，晚峰放电",
        loc="left",
        fontsize=17,
        fontweight="bold",
    )
    power_axis.legend(loc="upper left", ncol=2, frameon=False)

    soc_axis.plot(
        hours,
        frame["soc_end"],
        color="#6B4D7B",
        linewidth=2.8,
        label="日末 SOC",
    )
    soc_axis.fill_between(
        hours,
        0,
        frame["soc_end"],
        color="#D8C8DF",
        alpha=0.5,
    )
    soc_axis.set_ylim(0, 1)
    soc_axis.set_ylabel("SOC")
    soc_axis.set_xlabel("本地时间")
    soc_axis.legend(loc="upper left", frameon=False)

    for axis in (power_axis, soc_axis):
        axis.set_xlim(0, 24)
        axis.set_xticks(range(0, 25, 2))
        axis.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 2)])

    figure.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.09, hspace=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_path = (
        project_root
        / "数据格式与校验"
        / "samples"
        / "customer_1_2010-07-01_energy_balance.csv"
    )
    raw = pd.read_csv(sample_path, encoding="utf-8-sig")
    household = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(raw["interval_start"]),
            "total_load_kw": raw["total_load_kwh"] / 0.5,
            "pv_kw": raw["gg_kwh"] / 0.5,
        }
    )

    parameters = load_battery_parameters("battery.toml")
    dispatch = build_rule_dispatch(household, parameters)
    summary = summarize_balance(dispatch, dt_hours=parameters.dt_hours)

    base = household.copy()
    base["grid_net_kw"] = base["total_load_kw"] - base["pv_kw"]
    baseline_import = float(base["grid_net_kw"].clip(lower=0).sum() * 0.5)
    baseline_export = float((-base["grid_net_kw"]).clip(lower=0).sum() * 0.5)
    summary.update(
        {
            "baseline_import_kwh": baseline_import,
            "baseline_export_kwh": baseline_export,
            "import_reduction_kwh": baseline_import - summary["grid_import_kwh"],
            "final_soc": float(dispatch["soc_end"].iloc[-1]),
        }
    )

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    dispatch_path = output_dir / "rule_dispatch_customer_1_2010-07-01.csv"
    dispatch.to_csv(dispatch_path, index=False, encoding="utf-8-sig")
    summary_path = output_dir / "rule_dispatch_customer_1_2010-07-01_metrics.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_path = output_dir / "rule_dispatch_customer_1_2010-07-01.png"
    plot_rule_dispatch(dispatch, figure_path)

    print(
        f"[OK] 规则调度：购电 {baseline_import:.3f} → "
        f"{summary['grid_import_kwh']:.3f} kWh，"
        f"减少 {summary['import_reduction_kwh']:.3f} kWh"
    )
    print(f"[OK] 日末 SOC：{summary['final_soc']:.3f}")
    print(f"[OK] 调度明细：{dispatch_path}")
    print(f"[OK] 指标：{summary_path}")
    print(f"[OK] 图：{figure_path}")


if __name__ == "__main__":
    main()
