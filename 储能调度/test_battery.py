from __future__ import annotations

import unittest

import pandas as pd

from battery import (
    BatteryParameters,
    load_battery_parameters,
    simulate_dispatch,
    summarize_dispatch,
    validate_terminal_soc,
)


class BatteryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = BatteryParameters(
            capacity_kwh=10.0,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
            soc_min=0.1,
            soc_max=0.9,
            initial_soc=0.5,
            dt_hours=0.5,
        )

    def test_charging_updates_soc(self) -> None:
        result = simulate_dispatch(
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-01 00:00")],
                    "charge_kw": [2.0],
                    "discharge_kw": [0.0],
                }
            ),
            self.parameters,
        )
        self.assertAlmostEqual(result.iloc[0]["charge_to_battery_kwh"], 0.9)
        self.assertAlmostEqual(result.iloc[0]["soc_start"], 0.5)
        self.assertAlmostEqual(result.iloc[0]["soc_end"], 0.59)
        self.assertAlmostEqual(result.iloc[0]["battery_net_ac_kw"], -2.0)

    def test_discharging_updates_soc(self) -> None:
        result = simulate_dispatch(
            pd.DataFrame(
                {"charge_kw": [0.0], "discharge_kw": [2.0]}
            ),
            self.parameters,
        )
        expected_soc = 5.0 - (2.0 * 0.5) / 0.9
        self.assertAlmostEqual(result.iloc[0]["soc_end_kwh"], expected_soc)
        self.assertAlmostEqual(result.iloc[0]["battery_net_ac_kw"], 2.0)

    def test_idle_step_keeps_soc(self) -> None:
        result = simulate_dispatch(
            pd.DataFrame({"charge_kw": [0.0], "discharge_kw": [0.0]}),
            self.parameters,
        )
        self.assertAlmostEqual(result.iloc[0]["soc_start"], 0.5)
        self.assertAlmostEqual(result.iloc[0]["soc_end"], 0.5)

    def test_simultaneous_charge_and_discharge_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_dispatch(
                pd.DataFrame({"charge_kw": [1.0], "discharge_kw": [1.0]}),
                self.parameters,
            )

    def test_soc_upper_bound_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            simulate_dispatch(
                pd.DataFrame(
                    {
                        "charge_kw": [5.0, 5.0, 5.0],
                        "discharge_kw": [0.0, 0.0, 0.0],
                    }
                ),
                self.parameters,
            )

    def test_power_limit_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            simulate_dispatch(
                pd.DataFrame({"charge_kw": [6.0], "discharge_kw": [0.0]}),
                self.parameters,
            )

    def test_summary(self) -> None:
        result = simulate_dispatch(
            pd.DataFrame(
                {"charge_kw": [2.0, 0.0], "discharge_kw": [0.0, 2.0]}
            ),
            self.parameters,
        )
        summary = summarize_dispatch(result, self.parameters)
        self.assertAlmostEqual(summary["charge_ac_kwh"], 1.0)
        self.assertAlmostEqual(summary["discharge_ac_kwh"], 1.0)
        self.assertAlmostEqual(summary["battery_charge_kwh"], 0.9)
        self.assertAlmostEqual(summary["battery_discharge_kwh"], 1.111111111)

    def test_terminal_soc_validation(self) -> None:
        result = simulate_dispatch(
            pd.DataFrame({"charge_kw": [0.0], "discharge_kw": [0.0]}),
            self.parameters,
        )
        validate_terminal_soc(
            result,
            self.parameters,
            target_soc=0.5,
        )
        with self.assertRaises(ValueError):
            validate_terminal_soc(
                result,
                self.parameters,
                target_soc=0.6,
            )

    def test_load_toml(self) -> None:
        parameters = load_battery_parameters("battery.toml")
        self.assertEqual(parameters.capacity_kwh, 10.0)
        self.assertEqual(parameters.charge_efficiency, 0.95)


if __name__ == "__main__":
    unittest.main()
