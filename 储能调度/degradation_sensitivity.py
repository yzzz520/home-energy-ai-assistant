from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from battery import load_battery_parameters
from optimization import optimize_dispatch
from power_balance import build_tou_tariff


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_sensitivity(table: pd.DataFrame, output_path: Path) -> None:
    figure, (cost_axis, throughput_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (cost_axis, throughput_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    cost_axis.plot(
        table["degradation_cost_per_discharge_kwh"],
        table["total_cost"],
        marker="o",
        color="#2F7657",
        linewidth=2.5,
        label="总成本",
    )
    cost_axis.plot(
        table["degradation_cost_per_discharge_kwh"],
        table["energy_cost"],
        marker="s",
        color="#24708F",
        linewidth=2.0,
        label="购售电净成本",
    )
    cost_axis.set_ylabel("AUD")
    cost_axis.set_title(
        "退化成本敏感性：经济成本",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    cost_axis.legend(loc="upper left", frameon=False)

    throughput_axis.plot(
        table["degradation_cost_per_discharge_kwh"],
        table["discharge_ac_kwh"],
        marker="o",
        color="#C4592E",
        linewidth=2.5,
        label="放电 AC 电量",
    )
    throughput_axis.set_xlabel("退化成本（AUD / 放电 kWh）")
    throughput_axis.set_ylabel("kWh")
    throughput_axis.set_title(
        "退化成本敏感性：电池放电量",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    throughput_axis.legend(loc="upper right", frameon=False)

    figure.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.09, hspace=0.2)
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

    rates = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    rows: list[dict[str, float | str]] = []
    for rate in rates:
        _, metrics = optimize_dispatch(
            household,
            parameters,
            terminal_soc_min=parameters.initial_soc,
            degradation_cost_per_discharge_kwh=rate,
        )
        rows.append(
            {
                "degradation_cost_per_discharge_kwh": rate,
                "solver_status": metrics["solver_status"],
                "energy_cost": metrics["net_cost"],
                "degradation_cost": metrics["degradation_cost"],
                "total_cost": metrics["total_cost"],
                "charge_ac_kwh": metrics["charge_ac_kwh"],
                "discharge_ac_kwh": metrics["discharge_ac_kwh"],
                "final_soc": metrics["final_soc"],
            }
        )

    table = pd.DataFrame(rows)
    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "degradation_sensitivity_customer_1_2010-07-01.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path = output_dir / "degradation_sensitivity_customer_1_2010-07-01.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_path = output_dir / "degradation_sensitivity_customer_1_2010-07-01.png"
    plot_sensitivity(table, figure_path)

    zero = table.iloc[0]
    high = table.iloc[-1]
    print(
        f"[OK] 无退化成本：放电 {zero['discharge_ac_kwh']:.3f} kWh，"
        f"总成本 {zero['total_cost']:.3f} AUD"
    )
    print(
        f"[OK] 高退化成本：放电 {high['discharge_ac_kwh']:.3f} kWh，"
        f"总成本 {high['total_cost']:.3f} AUD"
    )
    print(f"[OK] 敏感性数据：{csv_path}")
    print(f"[OK] 敏感性图：{figure_path}")


if __name__ == "__main__":
    main()
