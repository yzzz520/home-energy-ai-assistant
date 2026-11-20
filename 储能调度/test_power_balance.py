from __future__ import annotations

import unittest

import pandas as pd

from battery import BatteryParameters
from power_balance import add_power_balance, build_rule_dispatch


class PowerBalanceTest(unittest.TestCase):
    def test_import_export_and_battery_sign(self) -> None:
        frame = pd.DataFrame(
            {
                "total_load_kw": [1.0, 0.5, 1.0],
                "pv_kw": [0.0, 1.0, 0.0],
                "battery_net_ac_kw": [0.2, 0.0, -0.3],
            }
        )
        result = add_power_balance(frame)
        self.assertAlmostEqual(result.iloc[0]["grid_net_kw"], 0.8)
        self.assertAlmostEqual(result.iloc[0]["grid_import_kw"], 0.8)
        self.assertAlmostEqual(result.iloc[1]["grid_net_kw"], -0.5)
        self.assertAlmostEqual(result.iloc[1]["grid_export_kw"], 0.5)
        self.assertAlmostEqual(result.iloc[2]["grid_net_kw"], 1.3)

    def test_rule_charges_on_surplus_and_discharges_in_evening(self) -> None:
        parameters = BatteryParameters(
            capacity_kwh=10.0,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
            soc_min=0.1,
            soc_max=0.9,
            initial_soc=0.2,
            dt_hours=0.5,
        )
        household = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(
                    [
                        "2024-01-01 12:00",
                        "2024-01-01 18:00",
                    ]
                ),
                "total_load_kw": [0.5, 2.0],
                "pv_kw": [3.0, 0.0],
            }
        )
        result = build_rule_dispatch(household, parameters)
        self.assertGreater(result.iloc[0]["charge_kw"], 0)
        self.assertEqual(result.iloc[0]["discharge_kw"], 0)
        self.assertEqual(result.iloc[1]["charge_kw"], 0)
        self.assertGreater(result.iloc[1]["discharge_kw"], 0)
        self.assertFalse(
            (
                (result["charge_kw"] > 0)
                & (result["discharge_kw"] > 0)
            ).any()
        )
        self.assertTrue(
            ((result["grid_import_kw"] >= -1e-9)).all()
        )

    def test_rule_does_not_charge_from_grid(self) -> None:
        parameters = BatteryParameters(initial_soc=0.2)
        household = pd.DataFrame(
            {
                "interval_start": [pd.Timestamp("2024-01-01 12:00")],
                "total_load_kw": [2.0],
                "pv_kw": [0.0],
            }
        )
        result = build_rule_dispatch(household, parameters)
        self.assertEqual(result.iloc[0]["charge_kw"], 0)
        self.assertEqual(result.iloc[0]["discharge_kw"], 0)


if __name__ == "__main__":
    unittest.main()
