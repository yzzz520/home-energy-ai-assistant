from __future__ import annotations

import unittest

import pandas as pd

from battery import BatteryParameters
from optimization import optimize_dispatch
from power_balance import (
    add_battery_degradation_cost,
    add_energy_cost,
    build_rule_dispatch,
)


class OptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = BatteryParameters(
            capacity_kwh=4.0,
            max_charge_kw=2.0,
            max_discharge_kw=2.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
            soc_min=0.1,
            soc_max=0.9,
            initial_soc=0.5,
            dt_hours=0.5,
        )
        self.household = pd.DataFrame(
            {
                "interval_start": pd.date_range(
                    "2024-01-01 00:00",
                    periods=4,
                    freq="30min",
                ),
                "total_load_kw": [0.5, 0.5, 2.0, 2.0],
                "pv_kw": [0.0, 0.0, 0.0, 0.0],
                "buy_price_per_kwh": [0.1, 0.1, 1.0, 1.0],
                "sell_price_per_kwh": [0.05] * 4,
            }
        )

    def test_milp_is_optimal_and_respects_mode_constraints(self) -> None:
        result, metrics = optimize_dispatch(
            self.household,
            self.parameters,
            terminal_soc_min=0.5,
        )
        self.assertEqual(metrics["solver_status"], "Optimal")
        self.assertGreaterEqual(metrics["final_soc"], 0.5)
        self.assertFalse(
            (
                (result["charge_kw"] > 1e-6)
                & (result["discharge_kw"] > 1e-6)
            ).any()
        )
        self.assertGreater(result["charge_kw"].sum(), 0)
        self.assertGreater(result["discharge_kw"].sum(), 0)

    def test_optimized_cost_beats_rule_dispatch(self) -> None:
        optimized, metrics = optimize_dispatch(
            self.household,
            self.parameters,
            terminal_soc_min=0.5,
        )
        rule = build_rule_dispatch(
            self.household,
            self.parameters,
            terminal_soc=0.5,
        )
        rule = add_energy_cost(rule)
        rule = add_battery_degradation_cost(
            rule,
            degradation_cost_per_discharge_kwh=0.0,
            dt_hours=self.parameters.dt_hours,
        )
        self.assertLess(
            metrics["total_cost"],
            float(rule["total_cost"].sum()),
        )

    def test_rule_terminal_soc_floor(self) -> None:
        result = build_rule_dispatch(
            self.household,
            self.parameters,
            terminal_soc=0.5,
        )
        self.assertGreaterEqual(
            float(result["soc_end"].iloc[-1]),
            0.5 - 1e-9,
        )

    def test_degradation_cost_reduces_battery_cycling(self) -> None:
        low_cost, _ = optimize_dispatch(
            self.household,
            self.parameters,
            terminal_soc_min=0.5,
            degradation_cost_per_discharge_kwh=0.0,
        )
        high_cost, _ = optimize_dispatch(
            self.household,
            self.parameters,
            terminal_soc_min=0.5,
            degradation_cost_per_discharge_kwh=1.0,
        )
        self.assertLess(
            high_cost["discharge_kw"].sum(),
            low_cost["discharge_kw"].sum(),
        )


if __name__ == "__main__":
    unittest.main()
