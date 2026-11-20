from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatteryParameters:
    """Battery parameters with AC-side power and AC-side energy."""

    capacity_kwh: float = 10.0
    max_charge_kw: float = 5.0
    max_discharge_kw: float = 5.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.90
    initial_soc: float = 0.50
    dt_hours: float = 0.5

    def __post_init__(self) -> None:
        numeric_values = {
            "capacity_kwh": self.capacity_kwh,
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "initial_soc": self.initial_soc,
            "dt_hours": self.dt_hours,
        }
        invalid = {
            name: value
            for name, value in numeric_values.items()
            if not math.isfinite(value)
        }
        if invalid:
            raise ValueError(f"电池参数必须是有限数值：{invalid}")
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh 必须大于 0")
        if self.max_charge_kw <= 0 or self.max_discharge_kw <= 0:
            raise ValueError("最大充电和放电功率必须大于 0")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency 必须在 (0, 1] 内")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency 必须在 (0, 1] 内")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("SOC 边界必须满足 0 <= soc_min < soc_max <= 1")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc 必须位于 SOC 上下限内")
        if self.dt_hours <= 0:
            raise ValueError("dt_hours 必须大于 0")


def load_battery_parameters(path: str | Path) -> BatteryParameters:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    return BatteryParameters(**config["battery"])


def simulate_dispatch(
    dispatch: pd.DataFrame,
    parameters: BatteryParameters,
    *,
    charge_col: str = "charge_kw",
    discharge_col: str = "discharge_kw",
    timestamp_col: str | None = "timestamp",
    tolerance: float = 1e-7,
) -> pd.DataFrame:
    """
    Simulate an AC-side battery schedule.

    charge_kw > 0 means power flows into the battery.
    discharge_kw > 0 means power flows from the battery.
    """
    required = [charge_col, discharge_col]
    missing = [column for column in required if column not in dispatch.columns]
    if missing:
        raise ValueError(f"调度输入缺少字段：{missing}")

    values = dispatch[required].to_numpy(dtype=float)
    if values.size and not np.isfinite(values).all():
        raise ValueError("充电或放电功率必须是有限数值")
    if (values < 0).any():
        raise ValueError("充电和放电功率不能为负值")
    if (dispatch[charge_col] > parameters.max_charge_kw + tolerance).any():
        raise ValueError("存在超过 max_charge_kw 的充电功率")
    if (dispatch[discharge_col] > parameters.max_discharge_kw + tolerance).any():
        raise ValueError("存在超过 max_discharge_kw 的放电功率")
    simultaneous = (
        (dispatch[charge_col] > tolerance)
        & (dispatch[discharge_col] > tolerance)
    )
    if simultaneous.any():
        raise ValueError(
            "同一时段不能同时充电和放电；异常行："
            f"{dispatch.index[simultaneous].tolist()[:10]}"
        )

    current_soc_kwh = parameters.initial_soc * parameters.capacity_kwh
    records: list[dict[str, object]] = []
    for position, row in dispatch.iterrows():
        charge_kw = float(row[charge_col])
        discharge_kw = float(row[discharge_col])
        charge_to_battery = (
            charge_kw * parameters.dt_hours * parameters.charge_efficiency
        )
        discharge_from_battery = (
            discharge_kw * parameters.dt_hours
            / parameters.discharge_efficiency
        )
        next_soc_kwh = (
            current_soc_kwh
            + charge_to_battery
            - discharge_from_battery
        )
        next_soc_fraction = next_soc_kwh / parameters.capacity_kwh
        if next_soc_fraction > parameters.soc_max + tolerance:
            raise ValueError(
                f"第 {position} 行导致 SOC 超过上限："
                f"{next_soc_fraction:.6f} > {parameters.soc_max:.6f}"
            )
        if next_soc_fraction < parameters.soc_min - tolerance:
            raise ValueError(
                f"第 {position} 行导致 SOC 低于下限："
                f"{next_soc_fraction:.6f} < {parameters.soc_min:.6f}"
            )

        record: dict[str, object] = {}
        if timestamp_col is not None and timestamp_col in dispatch.columns:
            record[timestamp_col] = row[timestamp_col]
        record.update(
            {
                charge_col: charge_kw,
                discharge_col: discharge_kw,
                "battery_net_ac_kw": discharge_kw - charge_kw,
                "charge_to_battery_kwh": charge_to_battery,
                "discharge_from_battery_kwh": discharge_from_battery,
                "soc_start_kwh": current_soc_kwh,
                "soc_end_kwh": next_soc_kwh,
                "soc_start": current_soc_kwh / parameters.capacity_kwh,
                "soc_end": next_soc_fraction,
            }
        )
        records.append(record)
        current_soc_kwh = next_soc_kwh

    columns: list[str] = []
    if timestamp_col is not None and timestamp_col in dispatch.columns:
        columns.append(timestamp_col)
    columns.extend(
        [
            charge_col,
            discharge_col,
            "battery_net_ac_kw",
            "charge_to_battery_kwh",
            "discharge_from_battery_kwh",
            "soc_start_kwh",
            "soc_end_kwh",
            "soc_start",
            "soc_end",
        ]
    )
    return pd.DataFrame(records, columns=columns)


def validate_terminal_soc(
    simulation: pd.DataFrame,
    parameters: BatteryParameters,
    *,
    target_soc: float,
    tolerance: float = 1e-6,
) -> None:
    if not 0 <= target_soc <= 1:
        raise ValueError("target_soc 必须在 [0, 1] 内")
    if simulation.empty:
        raise ValueError("仿真结果为空，无法校验终止 SOC")
    final_soc = float(simulation["soc_end"].iloc[-1])
    if abs(final_soc - target_soc) > tolerance:
        raise ValueError(
            f"终止 SOC 不满足要求：{final_soc:.6f} != {target_soc:.6f}"
        )


def summarize_dispatch(
    simulation: pd.DataFrame,
    parameters: BatteryParameters,
) -> dict[str, float]:
    required = {
        "charge_kw",
        "discharge_kw",
        "charge_to_battery_kwh",
        "discharge_from_battery_kwh",
        "soc_end",
    }
    missing = required - set(simulation.columns)
    if missing:
        raise ValueError(f"仿真结果缺少字段：{sorted(missing)}")
    return {
        "charge_ac_kwh": float(
            simulation["charge_kw"].sum() * parameters.dt_hours
        ),
        "discharge_ac_kwh": float(
            simulation["discharge_kw"].sum() * parameters.dt_hours
        ),
        "battery_charge_kwh": float(
            simulation["charge_to_battery_kwh"].sum()
        ),
        "battery_discharge_kwh": float(
            simulation["discharge_from_battery_kwh"].sum()
        ),
        "final_soc": float(simulation["soc_end"].iloc[-1])
        if len(simulation)
        else float("nan"),
    }
