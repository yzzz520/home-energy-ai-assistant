from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from battery import load_battery_parameters
from forecast_impact_demo import settle_schedule_on_actual
from optimization import optimize_dispatch
from power_balance import (
    add_battery_degradation_cost,
    add_energy_cost,
    build_rule_dispatch,
    build_tou_tariff,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_diagnostic_data() -> pd.DataFrame:
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
            "seasonal_naive_daily",
        ],
    )
    forecasts = forecasts.loc[forecasts["Customer"].eq(1)].copy()
    forecasts["interval_start"] = pd.to_datetime(
        forecasts["forecast_timestamp"]
    )
    forecasts["date"] = forecasts["interval_start"].dt.normalize()
    forecasts["hour"] = forecasts["interval_start"].dt.hour

    clean = pd.read_csv(clean_path, encoding="utf-8-sig")
    clean["interval_start"] = pd.to_datetime(clean["interval_start"])
    clean_index = clean.set_index("interval_start")
    forecasts["actual_pv_kwh"] = clean_index["gg_kwh"].reindex(
        forecasts["interval_start"]
    ).to_numpy()
    forecasts["forecast_pv_kwh"] = clean_index["gg_kwh"].reindex(
        forecasts["interval_start"] - pd.Timedelta(days=1)
    ).to_numpy()
    return forecasts


def forecast_diagnostics(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    table = forecasts.copy()
    table["load_bias_kwh"] = table["lightgbm"] - table["actual"]
    table["load_abs_error_kwh"] = table["load_bias_kwh"].abs()
    table["naive_load_bias_kwh"] = (
        table["seasonal_naive_daily"] - table["actual"]
    )
    table["naive_load_abs_error_kwh"] = table[
        "naive_load_bias_kwh"
    ].abs()
    table["pv_bias_kwh"] = (
        table["forecast_pv_kwh"] - table["actual_pv_kwh"]
    )
    table["pv_abs_error_kwh"] = table["pv_bias_kwh"].abs()

    hourly = (
        table.groupby("hour")
        .agg(
            load_bias_kwh=("load_bias_kwh", "mean"),
            load_mae_kwh=("load_abs_error_kwh", "mean"),
            naive_load_bias_kwh=("naive_load_bias_kwh", "mean"),
            naive_load_mae_kwh=("naive_load_abs_error_kwh", "mean"),
            pv_bias_kwh=("pv_bias_kwh", "mean"),
            pv_mae_kwh=("pv_abs_error_kwh", "mean"),
        )
        .reset_index()
    )
    summary = {
        "load": {
            "mean_bias_kwh": float(table["load_bias_kwh"].mean()),
            "mae_kwh": float(table["load_abs_error_kwh"].mean()),
            "underprediction_share_pct": float(
                (table["load_bias_kwh"] < 0).mean() * 100
            ),
        },
        "naive_load": {
            "mean_bias_kwh": float(table["naive_load_bias_kwh"].mean()),
            "mae_kwh": float(table["naive_load_abs_error_kwh"].mean()),
            "underprediction_share_pct": float(
                (table["naive_load_bias_kwh"] < 0).mean() * 100
            ),
        },
        "pv": {
            "mean_bias_kwh": float(table["pv_bias_kwh"].mean()),
            "mae_kwh": float(table["pv_abs_error_kwh"].mean()),
            "underprediction_share_pct": float(
                (table["pv_bias_kwh"] < 0).mean() * 100
            ),
        },
    }
    return hourly, summary


def plot_diagnostics(
    hourly_forecast: pd.DataFrame,
    hourly_regret: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.patch.set_facecolor("#F7F8F5")
    for axis in axes.flat:
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0, 0].bar(
        hourly_forecast["hour"],
        hourly_forecast["load_bias_kwh"],
        color=["#A44542" if value < 0 else "#24708F" for value in hourly_forecast["load_bias_kwh"]],
    )
    axes[0, 0].axhline(0, color="#47545D", linewidth=1)
    axes[0, 0].set_title("LightGBM 负荷偏差（预测 - 实际）", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("kWh / 半小时")

    axes[0, 1].bar(
        hourly_forecast["hour"],
        hourly_forecast["pv_bias_kwh"],
        color=["#A44542" if value < 0 else "#2F7657" for value in hourly_forecast["pv_bias_kwh"]],
    )
    axes[0, 1].axhline(0, color="#47545D", linewidth=1)
    axes[0, 1].set_title("PV 朴素预测偏差（预测 - 实际）", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("kWh / 半小时")

    axes[1, 0].bar(
        hourly_regret["hour"],
        hourly_regret["regret"],
        color=["#A44542" if value > 0 else "#2F7657" for value in hourly_regret["regret"]],
    )
    axes[1, 0].axhline(0, color="#47545D", linewidth=1)
    axes[1, 0].set_title("预测驱动 MILP 成本 regret 按时段", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("AUD")
    axes[1, 0].set_xlabel("小时")

    axes[1, 1].plot(
        fallback["threshold_pct"],
        fallback["total_cost"],
        marker="o",
        color="#24708F",
        linewidth=2.5,
    )
    axes[1, 1].axhline(
        fallback["perfect_cost"].iloc[0],
        color="#2F7657",
        linestyle="--",
        label="完美预知",
    )
    axes[1, 1].axhline(
        fallback["rule_cost"].iloc[0],
        color="#C4592E",
        linestyle=":",
        label="始终规则",
    )
    axes[1, 1].set_title("规则回退安全层阈值", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("预测优势阈值（%）")
    axes[1, 1].set_ylabel("60 天实际成本 AUD")
    axes[1, 1].legend(frameon=False)

    figure.suptitle(
        "预测误差与调度损失诊断",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.08, hspace=0.32, wspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    forecasts = build_diagnostic_data()
    hourly_forecast, forecast_summary = forecast_diagnostics(forecasts)

    parameters = load_battery_parameters("battery.toml")
    terminal_soc = parameters.initial_soc
    degradation_cost = 0.10
    daily_rows: list[dict[str, float | str]] = []
    interval_regrets: list[pd.DataFrame] = []
    threshold_values = [0.0, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]

    for date, day in forecasts.groupby("date"):
        day = day.sort_values("interval_start").reset_index(drop=True)
        if len(day) != 48:
            raise ValueError(f"{date.date()} 应有 48 个点，实际 {len(day)}")
        times = pd.DataFrame({"interval_start": day["interval_start"]})

        actual_household = times.copy()
        actual_household["total_load_kw"] = day["actual"] / 0.5
        actual_household["pv_kw"] = day["actual_pv_kwh"] / 0.5
        actual_household = build_tou_tariff(actual_household)

        forecast_household = times.copy()
        forecast_household["total_load_kw"] = day["lightgbm"] / 0.5
        forecast_household["pv_kw"] = day["forecast_pv_kwh"] / 0.5
        forecast_household = build_tou_tariff(forecast_household)

        naive_household = times.copy()
        naive_household["total_load_kw"] = (
            day["seasonal_naive_daily"] / 0.5
        )
        naive_household["pv_kw"] = day["forecast_pv_kwh"] / 0.5
        naive_household = build_tou_tariff(naive_household)

        rule_actual = build_rule_dispatch(
            actual_household,
            parameters,
            terminal_soc=terminal_soc,
        )
        rule_actual = add_energy_cost(rule_actual)
        rule_actual = add_battery_degradation_cost(
            rule_actual,
            degradation_cost_per_discharge_kwh=degradation_cost,
            dt_hours=parameters.dt_hours,
        )

        rule_forecast = build_rule_dispatch(
            forecast_household,
            parameters,
            terminal_soc=terminal_soc,
        )
        rule_forecast = add_energy_cost(rule_forecast)
        rule_forecast = add_battery_degradation_cost(
            rule_forecast,
            degradation_cost_per_discharge_kwh=degradation_cost,
            dt_hours=parameters.dt_hours,
        )

        lgbm_schedule, lgbm_forecast_metrics = optimize_dispatch(
            forecast_household,
            parameters,
            terminal_soc_min=terminal_soc,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        lgbm_settled = settle_schedule_on_actual(
            lgbm_schedule,
            actual_household,
            parameters=parameters,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )

        naive_schedule, _ = optimize_dispatch(
            naive_household,
            parameters,
            terminal_soc_min=terminal_soc,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        naive_settled = settle_schedule_on_actual(
            naive_schedule,
            actual_household,
            parameters=parameters,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )

        perfect, _ = optimize_dispatch(
            actual_household,
            parameters,
            terminal_soc_min=terminal_soc,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )

        row = {
            "date": date.date().isoformat(),
            "rule_cost": float(rule_actual["total_cost"].sum()),
            "lgbm_cost": float(lgbm_settled["total_cost"].sum()),
            "naive_cost": float(naive_settled["total_cost"].sum()),
            "perfect_cost": float(perfect["total_cost"].sum()),
            "rule_forecast_cost": float(rule_forecast["total_cost"].sum()),
            "lgbm_forecast_cost": float(
                lgbm_forecast_metrics["total_cost"]
            ),
        }
        daily_rows.append(row)

        regret_frame = pd.DataFrame(
            {
                "interval_start": lgbm_settled["interval_start"],
                "hour": pd.to_datetime(
                    lgbm_settled["interval_start"]
                ).dt.hour,
                "regret": (
                    lgbm_settled["total_cost"].to_numpy()
                    - perfect["total_cost"].to_numpy()
                ),
            }
        )
        interval_regrets.append(regret_frame)

    daily = pd.DataFrame(daily_rows)
    all_regrets = pd.concat(interval_regrets, ignore_index=True)
    hourly_regret = (
        all_regrets.groupby("hour", as_index=False)["regret"]
        .sum()
        .sort_values("hour")
    )

    fallback_rows: list[dict[str, float]] = []
    for threshold in threshold_values:
        total = 0.0
        milp_days = 0
        for _, row in daily.iterrows():
            advantage_ratio = (
                row["rule_forecast_cost"] - row["lgbm_forecast_cost"]
            ) / max(abs(row["rule_forecast_cost"]), 1e-9)
            if advantage_ratio > threshold:
                total += row["lgbm_cost"]
                milp_days += 1
            else:
                total += row["rule_cost"]
        fallback_rows.append(
            {
                "threshold_pct": threshold * 100,
                "total_cost": total,
                "milp_days": milp_days,
                "rule_days": len(daily) - milp_days,
                "perfect_cost": float(daily["perfect_cost"].sum()),
                "rule_cost": float(daily["rule_cost"].sum()),
            }
        )
    fallback = pd.DataFrame(fallback_rows)
    best_fallback = fallback.loc[fallback["total_cost"].idxmin()]

    summary = {
        "forecast_quality": forecast_summary,
        "strategy_costs": {
            "rule": float(daily["rule_cost"].sum()),
            "lightgbm_milp": float(daily["lgbm_cost"].sum()),
            "naive_milp": float(daily["naive_cost"].sum()),
            "perfect_milp": float(daily["perfect_cost"].sum()),
        },
        "lgbm_regret_vs_perfect": float(
            daily["lgbm_cost"].sum() - daily["perfect_cost"].sum()
        ),
        "naive_regret_vs_perfect": float(
            daily["naive_cost"].sum() - daily["perfect_cost"].sum()
        ),
        "worst_regret_hours": hourly_regret.nlargest(5, "regret")[
            ["hour", "regret"]
        ].to_dict(orient="records"),
        "fallback_safety_layer": {
            "best_threshold_pct": float(best_fallback["threshold_pct"]),
            "total_cost": float(best_fallback["total_cost"]),
            "milp_days": int(best_fallback["milp_days"]),
            "rule_days": int(best_fallback["rule_days"]),
            "perfect_cost": float(best_fallback["perfect_cost"]),
            "rule_cost": float(best_fallback["rule_cost"]),
        },
    }

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / "forecast_diagnostics_customer_1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    daily.to_csv(
        output_dir / "forecast_diagnostics_daily_customer_1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hourly_forecast.to_csv(
        output_dir / "forecast_diagnostics_hourly_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hourly_regret.to_csv(
        output_dir / "forecast_diagnostics_hourly_regret.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fallback.to_csv(
        output_dir / "forecast_diagnostics_fallback_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = output_dir / "forecast_diagnostics_customer_1.png"
    plot_diagnostics(
        hourly_forecast,
        hourly_regret,
        fallback,
        output_path=figure_path,
    )

    print(
        f"[OK] LightGBM 偏差 {forecast_summary['load']['mean_bias_kwh']:.4f} kWh，"
        f"PV 偏差 {forecast_summary['pv']['mean_bias_kwh']:.4f} kWh"
    )
    print(
        f"[OK] 规则 {summary['strategy_costs']['rule']:.2f}，"
        f"LGBM-MILP {summary['strategy_costs']['lightgbm_milp']:.2f}，"
        f"朴素-MILP {summary['strategy_costs']['naive_milp']:.2f}，"
        f"完美 {summary['strategy_costs']['perfect_milp']:.2f}"
    )
    print(
        f"[OK] 最佳回退阈值 {best_fallback['threshold_pct']:.0f}%，"
        f"成本 {best_fallback['total_cost']:.2f}，"
        f"MILP {int(best_fallback['milp_days'])} 天 / 规则 {int(best_fallback['rule_days'])} 天"
    )
    print(f"[OK] 诊断：{summary_path}")
    print(f"[OK] 图：{figure_path}")


if __name__ == "__main__":
    main()
