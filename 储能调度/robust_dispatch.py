from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from battery import load_battery_parameters
from forecast_diagnostics import build_diagnostic_data
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

SCHEDULE_COLUMNS = [
    "customer_id",
    "interval_start",
    "schedule_date",
    "strategy",
    "is_fallback",
    "load_kw",
    "pv_kw",
    "buy_price_per_kwh",
    "sell_price_per_kwh",
    "charge_kw",
    "discharge_kw",
    "battery_net_ac_kw",
    "soc_start",
    "soc_end",
    "grid_net_kw",
    "grid_import_kw",
    "grid_export_kw",
    "grid_import_kwh",
    "grid_export_kwh",
    "import_cost",
    "export_revenue",
    "net_cost",
    "degradation_cost",
    "total_cost",
]


def finalize_schedule(
    selected: pd.DataFrame,
    *,
    customer_id: int = 1,
) -> pd.DataFrame:
    required = {
        "interval_start",
        "total_load_kw",
        "pv_kw",
        "buy_price_per_kwh",
        "sell_price_per_kwh",
        "charge_kw",
        "discharge_kw",
        "battery_net_ac_kw",
        "soc_start",
        "soc_end",
        "grid_net_kw",
        "grid_import_kw",
        "grid_export_kw",
        "grid_import_kwh",
        "grid_export_kwh",
        "import_cost",
        "export_revenue",
        "net_cost",
        "degradation_cost",
        "total_cost",
        "strategy",
        "is_fallback",
    }
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"schedule 缺少字段：{sorted(missing)}")

    schedule = selected.copy()
    schedule["customer_id"] = int(customer_id)
    schedule["interval_start"] = pd.to_datetime(
        schedule["interval_start"]
    ).astype("datetime64[ns]")
    schedule["schedule_date"] = schedule["interval_start"].dt.date
    schedule = schedule.rename(columns={"total_load_kw": "load_kw"})
    schedule = schedule[SCHEDULE_COLUMNS].sort_values("interval_start")

    schedule["customer_id"] = schedule["customer_id"].astype("int32")
    float_columns = [
        column
        for column in SCHEDULE_COLUMNS
        if column
        not in {
            "customer_id",
            "interval_start",
            "schedule_date",
            "strategy",
            "is_fallback",
        }
    ]
    schedule[float_columns] = schedule[float_columns].astype("float64")
    schedule["strategy"] = schedule["strategy"].astype("string")
    schedule["is_fallback"] = schedule["is_fallback"].astype(bool)
    return schedule.reset_index(drop=True)


def calibrate_hourly_buffers(
    history: pd.DataFrame,
    *,
    load_quantile: float = 0.90,
    pv_quantile: float = 0.90,
) -> tuple[pd.Series, pd.Series]:
    history = history.copy()
    history["underforecast_load"] = (
        history["actual"] - history["lightgbm"]
    ).clip(lower=0)
    history["overforecast_pv"] = (
        history["forecast_pv_kwh"] - history["actual_pv_kwh"]
    ).clip(lower=0)
    load_buffer = history.groupby("hour")["underforecast_load"].quantile(
        load_quantile
    )
    pv_buffer = history.groupby("hour")["overforecast_pv"].quantile(
        pv_quantile
    )
    return load_buffer, pv_buffer


def apply_conservative_scenario(
    day: pd.DataFrame,
    *,
    load_buffer: pd.Series,
    pv_buffer: pd.Series,
    peak_start_hour: int = 16,
    peak_end_hour: int = 20,
) -> pd.DataFrame:
    result = day.copy()
    load_margin = result["hour"].map(load_buffer).fillna(0.0)
    pv_margin = result["hour"].map(pv_buffer).fillna(0.0)
    peak = result["hour"].between(
        peak_start_hour,
        peak_end_hour - 1,
        inclusive="both",
    )
    result["robust_load_kwh"] = (
        result["lightgbm"] + peak * load_margin
    )
    result["robust_pv_kwh"] = (
        result["forecast_pv_kwh"] - pv_margin
    ).clip(lower=0)
    return result


def should_fallback_to_rule(
    rule_expected_cost: float,
    robust_milp_expected_cost: float,
    *,
    minimum_advantage_pct: float = 20.0,
) -> bool:
    denominator = max(abs(rule_expected_cost), 1e-9)
    advantage_pct = (
        (rule_expected_cost - robust_milp_expected_cost)
        / denominator
        * 100
    )
    return advantage_pct < minimum_advantage_pct


def plot_robust_comparison(
    summary: dict[str, object],
    daily: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, (cost_axis, day_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.1, 1.4]},
    )
    figure.patch.set_facecolor("#F7F8F5")
    for axis in (cost_axis, day_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    labels = ["规则", "点预测 MILP", "稳健 MILP", "稳健+回退", "完美预知"]
    keys = ["rule", "point_milp", "robust_milp", "robust_fallback", "perfect"]
    colors = ["#24708F", "#C4592E", "#6B4D7B", "#2F7657", "#617078"]
    values = [summary[key]["total_cost"] for key in keys]
    cost_axis.bar(labels, values, color=colors, width=0.62)
    cost_axis.tick_params(axis="x", rotation=18)
    cost_axis.set_ylabel("AUD")
    cost_axis.set_title("实际结算总成本", loc="left", fontsize=15, fontweight="bold")
    for index, value in enumerate(values):
        cost_axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    dates = pd.to_datetime(daily["date"])
    day_axis.plot(dates, daily["rule_cost"], color="#24708F", linewidth=1.6, label="规则")
    day_axis.plot(dates, daily["point_cost"], color="#C4592E", linewidth=1.6, label="点预测 MILP")
    day_axis.plot(dates, daily["robust_cost"], color="#6B4D7B", linewidth=1.8, label="稳健 MILP")
    day_axis.plot(dates, daily["selected_cost"], color="#2F7657", linewidth=1.8, label="稳健+回退")
    day_axis.plot(dates, daily["perfect_cost"], color="#617078", linewidth=1.8, linestyle="--", label="完美")
    day_axis.set_title("逐日实际成本", loc="left", fontsize=15, fontweight="bold")
    day_axis.set_ylabel("AUD")
    day_axis.legend(loc="upper left", frameon=False, fontsize=8)
    day_axis.tick_params(axis="x", rotation=25)

    figure.suptitle(
        "轻量稳健调度：保守预测 + 规则回退安全层",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.19, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-quantile", type=float, default=0.60)
    parser.add_argument("--pv-quantile", type=float, default=0.60)
    parser.add_argument("--fallback-threshold", type=float, default=20.0)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    if not 0 <= args.load_quantile <= 1 or not 0 <= args.pv_quantile <= 1:
        raise ValueError("分位数必须在 [0, 1] 内")
    if args.fallback_threshold < 0:
        raise ValueError("回退阈值不能为负")

    diagnostics = build_diagnostic_data()
    parameters = load_battery_parameters("battery.toml")
    terminal_soc = parameters.initial_soc
    degradation_cost = 0.10
    warmup_days = 14
    minimum_advantage_pct = args.fallback_threshold
    all_dates = sorted(diagnostics["date"].unique())
    if len(all_dates) <= warmup_days:
        raise ValueError("预测历史不足以完成滚动校准")

    daily_rows: list[dict[str, object]] = []
    selected_schedules: list[pd.DataFrame] = []
    for date in all_dates[warmup_days:]:
        day = diagnostics.loc[
            diagnostics["date"].eq(date)
        ].sort_values("interval_start").reset_index(drop=True)
        history_dates = all_dates[
            max(0, all_dates.index(date) - warmup_days): all_dates.index(date)
        ]
        history = diagnostics.loc[
            diagnostics["date"].isin(history_dates)
        ]
        load_buffer, pv_buffer = calibrate_hourly_buffers(
            history,
            load_quantile=args.load_quantile,
            pv_quantile=args.pv_quantile,
        )
        robust_day = apply_conservative_scenario(
            day,
            load_buffer=load_buffer,
            pv_buffer=pv_buffer,
        )
        times = pd.DataFrame({"interval_start": day["interval_start"]})

        actual_household = times.copy()
        actual_household["total_load_kw"] = day["actual"] / 0.5
        actual_household["pv_kw"] = day["actual_pv_kwh"] / 0.5
        actual_household = build_tou_tariff(actual_household)

        point_household = times.copy()
        point_household["total_load_kw"] = day["lightgbm"] / 0.5
        point_household["pv_kw"] = day["forecast_pv_kwh"] / 0.5
        point_household = build_tou_tariff(point_household)

        robust_household = times.copy()
        robust_household["total_load_kw"] = robust_day["robust_load_kwh"] / 0.5
        robust_household["pv_kw"] = robust_day["robust_pv_kwh"] / 0.5
        robust_household = build_tou_tariff(robust_household)

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

        rule_point = build_rule_dispatch(
            point_household,
            parameters,
            terminal_soc=terminal_soc,
        )
        rule_point = add_energy_cost(rule_point)
        rule_point = add_battery_degradation_cost(
            rule_point,
            degradation_cost_per_discharge_kwh=degradation_cost,
            dt_hours=parameters.dt_hours,
        )

        point_schedule, _ = optimize_dispatch(
            point_household,
            parameters,
            terminal_soc_min=terminal_soc,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        point_settled = settle_schedule_on_actual(
            point_schedule,
            actual_household,
            parameters=parameters,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )

        robust_schedule, _ = optimize_dispatch(
            robust_household,
            parameters,
            terminal_soc_min=terminal_soc,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        robust_settled = settle_schedule_on_actual(
            robust_schedule,
            actual_household,
            parameters=parameters,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        robust_expected = settle_schedule_on_actual(
            robust_schedule,
            point_household,
            parameters=parameters,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )

        perfect, _ = optimize_dispatch(
            actual_household,
            parameters,
            terminal_soc_min=terminal_soc,
            degradation_cost_per_discharge_kwh=degradation_cost,
        )
        fallback = should_fallback_to_rule(
            float(rule_point["total_cost"].sum()),
            float(robust_expected["total_cost"].sum()),
            minimum_advantage_pct=minimum_advantage_pct,
        )
        selected_cost = (
            float(rule_actual["total_cost"].sum())
            if fallback
            else float(robust_settled["total_cost"].sum())
        )
        selected = rule_actual.copy() if fallback else robust_settled.copy()
        selected["strategy"] = (
            "rule_fallback" if fallback else "robust_milp"
        )
        selected["is_fallback"] = bool(fallback)
        selected_schedules.append(selected)
        daily_rows.append(
            {
                "date": date.date().isoformat(),
                "rule_cost": float(rule_actual["total_cost"].sum()),
                "point_cost": float(point_settled["total_cost"].sum()),
                "robust_cost": float(robust_settled["total_cost"].sum()),
                "selected_cost": selected_cost,
                "perfect_cost": float(perfect["total_cost"].sum()),
                "fallback_to_rule": fallback,
                "peak_load_buffer_kw": float(
                    load_buffer.reindex(range(16, 20)).mean() / 0.5
                ),
                "pv_buffer_kw": float(
                    pv_buffer.reindex(range(7, 18)).mean() / 0.5
                ),
            }
        )

    daily = pd.DataFrame(daily_rows)
    summary = {
        "evaluation_days": len(daily),
        "warmup_days": warmup_days,
        "period_start": daily["date"].iloc[0],
        "period_end": daily["date"].iloc[-1],
        "forecast": "LightGBM load + previous-day PV",
        "robustness": {
            "load_margin": (
                f"{args.load_quantile:.0%} percentile underforecast "
                "in 16:00-19:00"
            ),
            "pv_margin": (
                f"{args.pv_quantile:.0%} percentile PV overforecast"
            ),
            "fallback_threshold_pct": minimum_advantage_pct,
        },
        "rule": {"total_cost": float(daily["rule_cost"].sum())},
        "point_milp": {"total_cost": float(daily["point_cost"].sum())},
        "robust_milp": {"total_cost": float(daily["robust_cost"].sum())},
        "robust_fallback": {
            "total_cost": float(daily["selected_cost"].sum()),
            "milp_days": int((~daily["fallback_to_rule"]).sum()),
            "rule_days": int(daily["fallback_to_rule"].sum()),
        },
        "perfect": {"total_cost": float(daily["perfect_cost"].sum())},
    }
    summary["robust_regret_vs_perfect"] = (
        summary["robust_milp"]["total_cost"]
        - summary["perfect"]["total_cost"]
    )
    summary["fallback_regret_vs_perfect"] = (
        summary["robust_fallback"]["total_cost"]
        - summary["perfect"]["total_cost"]
    )
    schedule = finalize_schedule(
        pd.concat(selected_schedules, ignore_index=True)
    )

    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    suffix = args.output_suffix
    summary_path = output_dir / f"robust_dispatch_customer_1{suffix}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    daily.to_csv(
        output_dir / f"robust_dispatch_daily_customer_1{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = output_dir / f"robust_dispatch_customer_1{suffix}.png"
    plot_robust_comparison(summary, daily, figure_path)
    schedule_path = output_dir / "schedule.parquet"
    schedule.to_parquet(schedule_path, index=False)
    schema = {
        "schema_version": "1.0",
        "primary_key": ["customer_id", "interval_start"],
        "timezone": "Australia/Sydney local wall-clock, no DST localization",
        "row_granularity": "one customer per 30-minute interval",
        "columns": {
            "customer_id": "int32",
            "interval_start": "datetime64[ns]",
            "schedule_date": "date32",
            "strategy": "string: robust_milp | rule_fallback",
            "is_fallback": "bool",
            "load_kw": "float64, settlement load",
            "pv_kw": "float64, settlement PV",
            "buy_price_per_kwh": "float64, AUD/kWh",
            "sell_price_per_kwh": "float64, AUD/kWh",
            "charge_kw": "float64, AC-side",
            "discharge_kw": "float64, AC-side",
            "battery_net_ac_kw": "float64, discharge positive",
            "soc_start": "float64, fraction 0-1",
            "soc_end": "float64, fraction 0-1",
            "grid_net_kw": "float64, import positive",
            "grid_import_kw": "float64",
            "grid_export_kw": "float64",
            "grid_import_kwh": "float64",
            "grid_export_kwh": "float64",
            "import_cost": "float64, AUD",
            "export_revenue": "float64, AUD",
            "net_cost": "float64, AUD",
            "degradation_cost": "float64, AUD",
            "total_cost": "float64, AUD",
        },
    }
    schema_path = output_dir / "schedule_schema.json"
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[OK] 规则 {summary['rule']['total_cost']:.2f}，"
        f"点预测 MILP {summary['point_milp']['total_cost']:.2f}，"
        f"稳健 MILP {summary['robust_milp']['total_cost']:.2f}，"
        f"稳健+回退 {summary['robust_fallback']['total_cost']:.2f}，"
        f"完美 {summary['perfect']['total_cost']:.2f}"
    )
    print(
        f"[OK] 回退：MILP {summary['robust_fallback']['milp_days']} 天 / "
        f"规则 {summary['robust_fallback']['rule_days']} 天"
    )
    print(f"[OK] 汇总：{summary_path}")
    print(f"[OK] 图：{figure_path}")
    print(f"[OK] schedule：{schedule_path}")
    print(f"[OK] schema：{schema_path}")


if __name__ == "__main__":
    main()
