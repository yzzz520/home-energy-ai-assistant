from __future__ import annotations

import numpy as np
import pandas as pd

from battery import BatteryParameters, simulate_dispatch


def add_power_balance(
    frame: pd.DataFrame,
    *,
    load_col: str = "total_load_kw",
    pv_col: str = "pv_kw",
    battery_col: str = "battery_net_ac_kw",
    dt_hours: float = 0.5,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Apply the household power-balance convention.

    battery_net_ac_kw > 0: battery discharges to the house.
    battery_net_ac_kw < 0: battery charges from the house bus.
    grid_net_kw > 0: grid imports.
    grid_net_kw < 0: grid exports.
    """
    required = [load_col, pv_col, battery_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"功率平衡缺少字段：{missing}")
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("功率平衡输入必须是有限数值")

    result = frame.copy()
    result["house_net_without_battery_kw"] = (
        result[load_col] - result[pv_col]
    )
    raw_grid_net = (
        result["house_net_without_battery_kw"] - result[battery_col]
    )
    result["grid_net_kw"] = raw_grid_net.mask(
        raw_grid_net.abs().lt(tolerance),
        0.0,
    )
    result["grid_import_kw"] = result["grid_net_kw"].clip(lower=0.0)
    result["grid_export_kw"] = (-result["grid_net_kw"]).clip(lower=0.0)
    result["grid_import_kwh"] = result["grid_import_kw"] * dt_hours
    result["grid_export_kwh"] = result["grid_export_kw"] * dt_hours

    reconstructed = (
        result[load_col]
        - result[pv_col]
        - result["battery_net_ac_kw"]
    )
    if not np.allclose(
        result["grid_net_kw"].to_numpy(),
        reconstructed.to_numpy(),
        atol=tolerance,
    ):
        raise ValueError("并网功率平衡校验失败")
    return result


def build_rule_dispatch(
    household: pd.DataFrame,
    parameters: BatteryParameters,
    *,
    timestamp_col: str = "interval_start",
    load_col: str = "total_load_kw",
    pv_col: str = "pv_kw",
    peak_start_hour: int = 17,
    peak_end_hour: int = 21,
    terminal_soc: float | None = None,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Build a simple rule schedule:
    - Charge only from PV surplus.
    - Discharge during the evening peak to cover positive local net load.
    - Never export battery energy to the grid.
    """
    required = [timestamp_col, load_col, pv_col]
    missing = [column for column in required if column not in household.columns]
    if missing:
        raise ValueError(f"规则调度缺少字段：{missing}")
    if not 0 <= peak_start_hour < peak_end_hour <= 24:
        raise ValueError("晚峰时间窗口不合法")
    if terminal_soc is not None and not (
        parameters.soc_min <= terminal_soc <= parameters.soc_max
    ):
        raise ValueError("terminal_soc 必须位于 SOC 上下限内")
    discharge_floor_soc = (
        parameters.soc_min
        if terminal_soc is None
        else max(parameters.soc_min, terminal_soc)
    )

    ordered = household.sort_values(timestamp_col).copy()
    current_soc_kwh = parameters.initial_soc * parameters.capacity_kwh
    schedule_rows: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        timestamp = pd.Timestamp(row[timestamp_col])
        load_kw = float(row[load_col])
        pv_kw = float(row[pv_col])
        base_net_kw = load_kw - pv_kw
        charge_kw = 0.0
        discharge_kw = 0.0

        if base_net_kw < -tolerance:
            surplus_kw = -base_net_kw
            headroom_kwh = max(
                0.0,
                parameters.soc_max * parameters.capacity_kwh
                - current_soc_kwh,
            )
            max_charge_by_soc_kw = (
                headroom_kwh
                / (parameters.dt_hours * parameters.charge_efficiency)
            )
            charge_kw = min(
                surplus_kw,
                parameters.max_charge_kw,
                max_charge_by_soc_kw,
            )
        elif (
            peak_start_hour <= timestamp.hour < peak_end_hour
            and base_net_kw > tolerance
        ):
            available_kwh = max(
                0.0,
                current_soc_kwh
                - discharge_floor_soc * parameters.capacity_kwh,
            )
            max_discharge_by_soc_kw = (
                available_kwh
                * parameters.discharge_efficiency
                / parameters.dt_hours
            )
            discharge_kw = min(
                base_net_kw,
                parameters.max_discharge_kw,
                max_discharge_by_soc_kw,
            )

        current_soc_kwh += (
            charge_kw
            * parameters.dt_hours
            * parameters.charge_efficiency
            - discharge_kw
            * parameters.dt_hours
            / parameters.discharge_efficiency
        )
        schedule_record: dict[str, object] = {
            timestamp_col: timestamp,
            load_col: load_kw,
            pv_col: pv_kw,
            "charge_kw": charge_kw,
            "discharge_kw": discharge_kw,
        }
        for price_col in ("buy_price_per_kwh", "sell_price_per_kwh"):
            if price_col in ordered.columns:
                schedule_record[price_col] = row[price_col]
        schedule_rows.append(schedule_record)

    schedule = pd.DataFrame(schedule_rows)
    simulation = simulate_dispatch(
        schedule[
            [timestamp_col, "charge_kw", "discharge_kw"]
        ],
        parameters,
        timestamp_col=timestamp_col,
    )
    result = schedule.merge(
        simulation.drop(columns=["charge_kw", "discharge_kw"]),
        on=timestamp_col,
        how="left",
        validate="one_to_one",
    )
    return add_power_balance(
        result,
        load_col=load_col,
        pv_col=pv_col,
        dt_hours=parameters.dt_hours,
    )


def build_tou_tariff(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "interval_start",
    buy_offpeak: float = 0.18,
    buy_shoulder: float = 0.28,
    buy_peak: float = 0.48,
    sell_price: float = 0.08,
) -> pd.DataFrame:
    required = [timestamp_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"电价构造缺少字段：{missing}")
    if min(buy_offpeak, buy_shoulder, buy_peak, sell_price) < 0:
        raise ValueError("电价不能为负")

    result = frame.copy()
    hours = pd.to_datetime(result[timestamp_col]).dt.hour
    buy_price = np.select(
        [
            hours.lt(7) | hours.ge(22),
            hours.ge(16) & hours.lt(21),
        ],
        [buy_offpeak, buy_peak],
        default=buy_shoulder,
    )
    result["buy_price_per_kwh"] = buy_price.astype(float)
    result["sell_price_per_kwh"] = float(sell_price)
    return result


def add_energy_cost(
    frame: pd.DataFrame,
    *,
    buy_price_col: str = "buy_price_per_kwh",
    sell_price_col: str = "sell_price_per_kwh",
    import_col: str = "grid_import_kwh",
    export_col: str = "grid_export_kwh",
) -> pd.DataFrame:
    required = [buy_price_col, sell_price_col, import_col, export_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"成本计算缺少字段：{missing}")
    result = frame.copy()
    result["import_cost"] = (
        result[import_col] * result[buy_price_col]
    )
    result["export_revenue"] = (
        result[export_col] * result[sell_price_col]
    )
    result["net_cost"] = (
        result["import_cost"] - result["export_revenue"]
    )
    return result


def add_battery_degradation_cost(
    frame: pd.DataFrame,
    *,
    degradation_cost_per_discharge_kwh: float,
    dt_hours: float = 0.5,
    discharge_col: str = "discharge_kw",
    energy_cost_col: str = "net_cost",
) -> pd.DataFrame:
    if degradation_cost_per_discharge_kwh < 0:
        raise ValueError("退化成本不能为负")
    required = [discharge_col, energy_cost_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"退化成本计算缺少字段：{missing}")
    result = frame.copy()
    result["discharge_throughput_kwh"] = (
        result[discharge_col] * dt_hours
    )
    result["degradation_cost"] = (
        result["discharge_throughput_kwh"]
        * degradation_cost_per_discharge_kwh
    )
    result["total_cost"] = result[energy_cost_col] + result["degradation_cost"]
    return result


def summarize_balance(
    frame: pd.DataFrame,
    *,
    dt_hours: float = 0.5,
) -> dict[str, float]:
    required = {
        "total_load_kw",
        "pv_kw",
        "battery_net_ac_kw",
        "charge_kw",
        "discharge_kw",
        "grid_import_kwh",
        "grid_export_kwh",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"功率汇总缺少字段：{sorted(missing)}")
    return {
        "load_kwh": float(frame["total_load_kw"].sum() * dt_hours),
        "pv_kwh": float(frame["pv_kw"].sum() * dt_hours),
        "battery_charge_ac_kwh": float(
            frame["charge_kw"].sum() * dt_hours
        ),
        "battery_discharge_ac_kwh": float(
            frame["discharge_kw"].sum() * dt_hours
        ),
        "grid_import_kwh": float(frame["grid_import_kwh"].sum()),
        "grid_export_kwh": float(frame["grid_export_kwh"].sum()),
    }
