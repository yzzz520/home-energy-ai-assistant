from __future__ import annotations

import pulp
import pandas as pd

from battery import BatteryParameters, simulate_dispatch
from power_balance import (
    add_battery_degradation_cost,
    add_energy_cost,
    add_power_balance,
)


def optimize_dispatch(
    household: pd.DataFrame,
    parameters: BatteryParameters,
    *,
    timestamp_col: str = "interval_start",
    load_col: str = "total_load_kw",
    pv_col: str = "pv_kw",
    buy_price_col: str = "buy_price_per_kwh",
    sell_price_col: str = "sell_price_per_kwh",
    terminal_soc_min: float | None = None,
    degradation_cost_per_discharge_kwh: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    required = [
        timestamp_col,
        load_col,
        pv_col,
        buy_price_col,
        sell_price_col,
    ]
    missing = [column for column in required if column not in household.columns]
    if missing:
        raise ValueError(f"MILP 输入缺少字段：{missing}")
    terminal_floor = (
        parameters.initial_soc
        if terminal_soc_min is None
        else terminal_soc_min
    )
    if terminal_floor < parameters.initial_soc:
        raise ValueError("terminal_soc_min 不能低于初始 SOC")
    if not parameters.soc_min <= terminal_floor <= parameters.soc_max:
        raise ValueError("terminal_soc_min 必须位于 SOC 上下限内")
    if degradation_cost_per_discharge_kwh < 0:
        raise ValueError("退化成本不能为负")
    terminal_safety_margin = min(1e-6, parameters.soc_max - terminal_floor)

    data = household.sort_values(timestamp_col).reset_index(drop=True)
    periods = range(len(data))
    problem = pulp.LpProblem("household_battery_dispatch", pulp.LpMinimize)

    charge = pulp.LpVariable.dicts(
        "charge",
        periods,
        lowBound=0,
        upBound=parameters.max_charge_kw,
    )
    discharge = pulp.LpVariable.dicts(
        "discharge",
        periods,
        lowBound=0,
        upBound=parameters.max_discharge_kw,
    )
    grid_import = pulp.LpVariable.dicts("grid_import", periods, lowBound=0)
    grid_export = pulp.LpVariable.dicts("grid_export", periods, lowBound=0)
    battery_mode = pulp.LpVariable.dicts(
        "battery_mode",
        periods,
        cat=pulp.LpBinary,
    )
    grid_mode = pulp.LpVariable.dicts(
        "grid_mode",
        periods,
        cat=pulp.LpBinary,
    )
    soc = {
        period: pulp.LpVariable(
            f"soc_{period}",
            lowBound=parameters.soc_min * parameters.capacity_kwh,
            upBound=parameters.soc_max * parameters.capacity_kwh,
        )
        for period in range(len(data) + 1)
    }
    problem += soc[0] == parameters.initial_soc * parameters.capacity_kwh
    problem += soc[len(data)] >= (
        terminal_floor + terminal_safety_margin
    ) * parameters.capacity_kwh

    max_load = float(data[load_col].max())
    max_pv = float(data[pv_col].max())
    grid_big_m = (
        max_load
        + max_pv
        + parameters.max_charge_kw
        + parameters.max_discharge_kw
    )
    objective_terms = []

    for index, row in data.iterrows():
        load = float(row[load_col])
        pv = float(row[pv_col])
        buy_price = float(row[buy_price_col])
        sell_price = float(row[sell_price_col])

        problem += (
            charge[index]
            <= parameters.max_charge_kw * battery_mode[index]
        )
        problem += (
            discharge[index]
            <= parameters.max_discharge_kw * (1 - battery_mode[index])
        )
        problem += grid_import[index] <= grid_big_m * grid_mode[index]
        problem += grid_export[index] <= grid_big_m * (1 - grid_mode[index])

        problem += (
            load
            == pv
            + discharge[index]
            - charge[index]
            + grid_import[index]
            - grid_export[index]
        )
        problem += (
            soc[index + 1]
            == soc[index]
            + charge[index]
            * parameters.dt_hours
            * parameters.charge_efficiency
            - discharge[index]
            * parameters.dt_hours
            / parameters.discharge_efficiency
        )
        objective_terms.append(
            parameters.dt_hours
            * (
                buy_price * grid_import[index]
                - sell_price * grid_export[index]
                + degradation_cost_per_discharge_kwh
                * discharge[index]
            )
        )

    problem += pulp.lpSum(objective_terms)
    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(
            f"MILP 未获得最优解：{pulp.LpStatus[problem.status]}"
        )

    def clean_value(value: float | None) -> float:
        numeric = 0.0 if value is None else float(value)
        return 0.0 if abs(numeric) < 1e-7 else numeric

    schedule = pd.DataFrame(
        {
            timestamp_col: data[timestamp_col],
            load_col: data[load_col],
            pv_col: data[pv_col],
            buy_price_col: data[buy_price_col],
            sell_price_col: data[sell_price_col],
            "charge_kw": [
                clean_value(charge[index].value()) for index in periods
            ],
            "discharge_kw": [
                clean_value(discharge[index].value())
                for index in periods
            ],
        }
    )
    simulation = simulate_dispatch(
        schedule[[timestamp_col, "charge_kw", "discharge_kw"]],
        parameters,
        timestamp_col=timestamp_col,
    )
    result = schedule.merge(
        simulation.drop(columns=["charge_kw", "discharge_kw"]),
        on=timestamp_col,
        how="left",
        validate="one_to_one",
    )
    result = add_power_balance(
        result,
        load_col=load_col,
        pv_col=pv_col,
        dt_hours=parameters.dt_hours,
    )
    result = add_energy_cost(
        result,
        buy_price_col=buy_price_col,
        sell_price_col=sell_price_col,
    )
    result = add_battery_degradation_cost(
        result,
        degradation_cost_per_discharge_kwh=(
            degradation_cost_per_discharge_kwh
        ),
        dt_hours=parameters.dt_hours,
    )
    metrics = {
        "solver_status": pulp.LpStatus[problem.status],
        "objective": float(pulp.value(problem.objective)),
        "net_cost": float(result["net_cost"].sum()),
        "degradation_cost": float(result["degradation_cost"].sum()),
        "total_cost": float(result["total_cost"].sum()),
        "grid_import_kwh": float(result["grid_import_kwh"].sum()),
        "grid_export_kwh": float(result["grid_export_kwh"].sum()),
        "final_soc": float(result["soc_end"].iloc[-1]),
        "charge_ac_kwh": float(
            result["charge_kw"].sum() * parameters.dt_hours
        ),
        "discharge_ac_kwh": float(
            result["discharge_kw"].sum() * parameters.dt_hours
        ),
        "terminal_soc_min": float(terminal_floor),
        "terminal_soc_constraint": float(
            terminal_floor + terminal_safety_margin
        ),
        "degradation_cost_per_discharge_kwh": float(
            degradation_cost_per_discharge_kwh
        ),
    }
    return result, metrics
