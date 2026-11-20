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
    add_power_balance,
    build_rule_dispatch,
    build_tou_tariff,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def settle_schedule_on_actual(
    optimized_schedule: pd.DataFrame,
    actual_household: pd.DataFrame,
    *,
    parameters,
    degradation_cost_per_discharge_kwh: float,
) -> pd.DataFrame:
    schedule = optimized_schedule[
        [
            "interval_start",
            "charge_kw",
            "discharge_kw",
            "soc_start",
            "soc_end",
        ]
    ].copy()
    schedule = schedule.merge(
        actual_household[
            [
                "interval_start",
                "total_load_kw",
                "pv_kw",
                "buy_price_per_kwh",
                "sell_price_per_kwh",
            ]
        ],
        on="interval_start",
        how="inner",
        validate="one_to_one",
    )
    schedule["battery_net_ac_kw"] = (
        schedule["discharge_kw"] - schedule["charge_kw"]
    )
    result = add_power_balance(
        schedule,
        dt_hours=parameters.dt_hours,
    )
    result = add_energy_cost(result)
    result = add_battery_degradation_cost(
        result,
        degradation_cost_per_discharge_kwh=(
            degradation_cost_per_discharge_kwh
        ),
        dt_hours=parameters.dt_hours,
    )
    return result


def plot_comparison(
    daily: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
) -> None:
    figure, (cost_axis, daily_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (cost_axis, daily_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    labels = ["规则调度", "预测驱动 MILP", "完美预知 MILP"]
    values = [
        summary["rule"]["total_cost"],
        summary["forecast_milp"]["total_cost"],
        summary["perfect_milp"]["total_cost"],
    ]
    colors = ["#24708F", "#C4592E", "#2F7657"]
    cost_axis.bar(labels, values, color=colors, width=0.62)
    cost_axis.set_title("60 天实际结算总成本", loc="left", fontsize=15, fontweight="bold")
    cost_axis.set_ylabel("AUD")
    cost_axis.tick_params(axis="x", rotation=10)
    for index, value in enumerate(values):
        cost_axis.text(index, value, f"{value:.2f}", ha="center", va="bottom")

    dates = pd.to_datetime(daily["date"])
    daily_axis.plot(
        dates,
        daily["rule_cost"],
        color="#24708F",
        linewidth=1.8,
        label="规则调度",
    )
    daily_axis.plot(
        dates,
        daily["forecast_cost"],
        color="#C4592E",
        linewidth=1.8,
        label="预测驱动 MILP",
    )
    daily_axis.plot(
        dates,
        daily["perfect_cost"],
        color="#2F7657",
        linewidth=1.8,
        label="完美预知 MILP",
    )
    daily_axis.set_title("逐日实际结算成本", loc="left", fontsize=15, fontweight="bold")
    daily_axis.set_ylabel("AUD")
    daily_axis.legend(loc="upper left", frameon=False)
    daily_axis.tick_params(axis="x", rotation=25)

    figure.suptitle(
        "预测误差对储能调度成本的实际影响",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.18, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    forecast_path = (
        project_root
        / "时序预测"
        / "results"
        / "validation_300_households_predictions.csv"
    )
    clean_path = (
        project_root
        / "数据格式与校验"
        / "samples"
        / "customer_1_2010-2011_clean.csv"
    )

    forecasts = pd.read_csv(
        forecast_path,
        encoding="utf-8-sig",
        usecols=[
            "forecast_timestamp",
            "Customer",
            "actual",
            "lightgbm",
        ],
    )
    forecasts = forecasts.loc[forecasts["Customer"].eq(1)].copy()
    forecasts["interval_start"] = pd.to_datetime(
        forecasts["forecast_timestamp"]
    )
    forecasts["date"] = forecasts["interval_start"].dt.normalize()

    clean = pd.read_csv(clean_path, encoding="utf-8-sig")
    clean["interval_start"] = pd.to_datetime(clean["interval_start"])
    clean_index = clean.set_index("interval_start")
    actual_pv_kwh = clean_index["gg_kwh"].reindex(
        forecasts["interval_start"]
    ).to_numpy()
    pv_forecast_time = (
        forecasts["interval_start"] - pd.Timedelta(days=1)
    )
    forecast_pv_kwh = clean_index["gg_kwh"].reindex(
        pv_forecast_time
    ).to_numpy()

    forecasts["actual_load_kw"] = forecasts["actual"] / 0.5
    forecasts["forecast_load_kw"] = forecasts["lightgbm"] / 0.5
    forecasts["actual_pv_kw"] = actual_pv_kwh / 0.5
    forecasts["forecast_pv_kw"] = forecast_pv_kwh / 0.5

    parameters = load_battery_parameters("battery.toml")
    terminal_soc_floor = parameters.initial_soc
    degradation_cost = 0.10
    daily_rows: list[dict[str, object]] = []

    for date, day in forecasts.groupby("date"):
        if len(day) != 48:
            raise ValueError(f"{date.date()} 应有 48 个点，实际 {len(day)}")
        day = day.sort_values("interval_start").reset_index(drop=True)
        times = pd.DataFrame({"interval_start": day["interval_start"]})

        actual_household = times.copy()
        actual_household["total_load_kw"] = day["actual_load_kw"]
        actual_household["pv_kw"] = day["actual_pv_kw"]
        actual_household = build_tou_tariff(actual_household)

        forecast_household = times.copy()
        forecast_household["total_load_kw"] = day["forecast_load_kw"]
        forecast_household["pv_kw"] = day["forecast_pv_kw"]
        forecast_household = build_tou_tariff(forecast_household)

        rule = build_rule_dispatch(
            actual_household,
            parameters,
            terminal_soc=terminal_soc_floor,
        )
        rule = add_energy_cost(rule)
        rule = add_battery_degradation_cost(
            rule,
            degradation_cost_per_discharge_kwh=degradation_cost,
            dt_hours=parameters.dt_hours,
        )

        forecast_schedule, _ = optimize_dispatch(
            forecast_household,
            parameters,
            terminal_soc_min=terminal_soc_floor,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        forecast_settled = settle_schedule_on_actual(
            forecast_schedule,
            actual_household,
            parameters=parameters,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )

        perfect, perfect_metrics = optimize_dispatch(
            actual_household,
            parameters,
            terminal_soc_min=terminal_soc_floor,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        daily_rows.append(
            {
                "date": date.date().isoformat(),
                "rule_cost": float(rule["total_cost"].sum()),
                "forecast_cost": float(forecast_settled["total_cost"].sum()),
                "perfect_cost": float(perfect["total_cost"].sum()),
                "forecast_final_soc": float(
                    forecast_settled["soc_end"].iloc[-1]
                ),
                "perfect_final_soc": float(perfect["soc_end"].iloc[-1]),
                "perfect_solver_status": perfect_metrics["solver_status"],
            }
        )

    daily = pd.DataFrame(daily_rows)
    summary = {
        "customer_id": 1,
        "days": len(daily),
        "period_start": daily["date"].iloc[0],
        "period_end": daily["date"].iloc[-1],
        "settlement": "actual load and PV for all strategies",
        "forecast": "LightGBM load + previous-day PV seasonal naive",
        "degradation_cost_per_discharge_kwh": degradation_cost,
        "rule": {
            "total_cost": float(daily["rule_cost"].sum()),
            "mean_daily_cost": float(daily["rule_cost"].mean()),
        },
        "forecast_milp": {
            "total_cost": float(daily["forecast_cost"].sum()),
            "mean_daily_cost": float(daily["forecast_cost"].mean()),
        },
        "perfect_milp": {
            "total_cost": float(daily["perfect_cost"].sum()),
            "mean_daily_cost": float(daily["perfect_cost"].mean()),
        },
    }
    summary["forecast_regret"] = (
        summary["forecast_milp"]["total_cost"]
        - summary["perfect_milp"]["total_cost"]
    )
    summary["theoretical_max_saving"] = (
        summary["rule"]["total_cost"]
        - summary["perfect_milp"]["total_cost"]
    )
    if summary["theoretical_max_saving"] > 0:
        summary["forecast_capture_pct"] = (
            (
                summary["rule"]["total_cost"]
                - summary["forecast_milp"]["total_cost"]
            )
            / summary["theoretical_max_saving"]
            * 100
        )
    else:
        summary["forecast_capture_pct"] = None

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    daily_path = output_dir / "forecast_vs_perfect_daily_customer_1.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_path = output_dir / "forecast_vs_perfect_customer_1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_path = output_dir / "forecast_vs_perfect_customer_1.png"
    plot_comparison(daily, summary, figure_path)

    print(
        f"[OK] 规则 {summary['rule']['total_cost']:.2f} AUD，"
        f"预测驱动 {summary['forecast_milp']['total_cost']:.2f} AUD，"
        f"完美预知 {summary['perfect_milp']['total_cost']:.2f} AUD"
    )
    print(
        f"[OK] 预测 regret {summary['forecast_regret']:.2f} AUD，"
        f"理论上限捕获率 {summary['forecast_capture_pct']:.2f}%"
    )
    print(f"[OK] 汇总：{summary_path}")
    print(f"[OK] 图：{figure_path}")


if __name__ == "__main__":
    main()
