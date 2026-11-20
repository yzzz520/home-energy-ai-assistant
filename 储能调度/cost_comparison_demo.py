from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from battery import load_battery_parameters
from optimization import optimize_dispatch
from power_balance import (
    add_battery_degradation_cost,
    add_energy_cost,
    build_rule_dispatch,
    build_tou_tariff,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def summarize(
    frame: pd.DataFrame,
    parameters,
) -> dict[str, float | None]:
    return {
        "net_cost": float(frame["net_cost"].sum()),
        "degradation_cost": float(frame["degradation_cost"].sum()),
        "total_cost": float(frame["total_cost"].sum()),
        "grid_import_kwh": float(frame["grid_import_kwh"].sum()),
        "grid_export_kwh": float(frame["grid_export_kwh"].sum()),
        "charge_ac_kwh": float(
            frame["charge_kw"].sum() * parameters.dt_hours
        )
        if "charge_kw" in frame
        else 0.0,
        "discharge_ac_kwh": float(
            frame["discharge_kw"].sum() * parameters.dt_hours
        )
        if "discharge_kw" in frame
        else 0.0,
        "final_soc": (
            float(frame["soc_end"].iloc[-1])
            if "soc_end" in frame
            else None
        ),
    }


def plot_comparison(
    rule: pd.DataFrame,
    optimized: pd.DataFrame,
    summary: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    figure, (cost_axis, soc_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 1.25]},
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (cost_axis, soc_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    labels = ["规则调度", "优化调度"]
    colors = ["#24708F", "#2F7657"]
    costs = [
        summary["rule"]["total_cost"],
        summary["optimized"]["total_cost"],
    ]
    cost_axis.bar(labels, costs, color=colors, width=0.62)
    cost_axis.set_title("日净电费", loc="left", fontsize=15, fontweight="bold")
    cost_axis.set_ylabel("元")
    for index, value in enumerate(costs):
        cost_axis.text(index, value, f"{value:.2f}", ha="center", va="bottom")

    hours = (
        pd.to_datetime(rule["interval_start"]).dt.hour
        + pd.to_datetime(rule["interval_start"]).dt.minute / 60
        + 0.25
    )
    soc_axis.plot(hours, rule["soc_end"], color="#24708F", linewidth=2.5, label="规则调度")
    soc_axis.plot(
        hours,
        optimized["soc_end"],
        color="#2F7657",
        linewidth=2.5,
        linestyle="--",
        label="优化调度",
    )
    soc_axis.set_ylim(0, 1)
    soc_axis.set_xlim(0, 24)
    soc_axis.set_xticks(range(0, 25, 2))
    soc_axis.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 2)])
    soc_axis.set_title("SOC 轨迹", loc="left", fontsize=15, fontweight="bold")
    soc_axis.set_ylabel("SOC")
    soc_axis.legend(loc="upper left", frameon=False)

    figure.suptitle(
        "规则调度与 MILP 优化调度对比",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.13, wspace=0.22)
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
    household = build_tou_tariff(household)
    parameters = load_battery_parameters("battery.toml")
    terminal_soc_floor = parameters.initial_soc
    degradation_cost_per_kwh = 0.10

    rule = build_rule_dispatch(
        household,
        parameters,
        terminal_soc=terminal_soc_floor,
    )
    rule = add_energy_cost(rule)
    rule = add_battery_degradation_cost(
        rule,
        degradation_cost_per_discharge_kwh=degradation_cost_per_kwh,
        dt_hours=parameters.dt_hours,
    )

    optimized, optimization_metrics = optimize_dispatch(
        household,
        parameters,
        terminal_soc_min=terminal_soc_floor,
        degradation_cost_per_discharge_kwh=degradation_cost_per_kwh,
    )

    summary = {
        "rule": summarize(rule, parameters),
        "optimized": {
            **summarize(optimized, parameters),
            **optimization_metrics,
        },
    }
    summary["optimized_improvement_vs_rule"] = (
        summary["rule"]["total_cost"]
        - summary["optimized"]["total_cost"]
    )
    summary["tariff"] = {
        "buy_offpeak": 0.18,
        "buy_shoulder": 0.28,
        "buy_peak": 0.48,
        "sell_price": 0.08,
        "currency": "AUD",
        "degradation_cost_per_discharge_kwh": degradation_cost_per_kwh,
    }

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    comparison_path = output_dir / "cost_comparison_customer_1_2010-07-01.json"
    comparison_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    optimized.to_csv(
        output_dir / "optimized_dispatch_customer_1_2010-07-01.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = output_dir / "cost_comparison_customer_1_2010-07-01.png"
    plot_comparison(rule, optimized, summary, figure_path)

    print(
        f"[OK] 规则总成本 {summary['rule']['total_cost']:.2f}，"
        f"优化总成本 {summary['optimized']['total_cost']:.2f} AUD"
    )
    print(f"[OK] 优化日末 SOC：{summary['optimized']['final_soc']:.3f}")
    print(f"[OK] 对比指标：{comparison_path}")
    print(f"[OK] 对比图：{figure_path}")


if __name__ == "__main__":
    main()
