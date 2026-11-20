from __future__ import annotations

import unittest

import pandas as pd

from battery import BatteryParameters
from forecast_impact_demo import settle_schedule_on_actual


class ForecastImpactTest(unittest.TestCase):
    def test_settlement_uses_actual_household(self) -> None:
        parameters = BatteryParameters(
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
        timestamps = pd.date_range(
            "2024-01-01 00:00",
            periods=2,
            freq="30min",
        )
        schedule = pd.DataFrame(
            {
                "interval_start": timestamps,
                "charge_kw": [0.0, 0.0],
                "discharge_kw": [0.5, 0.0],
                "soc_start": [0.5, 0.45],
                "soc_end": [0.45, 0.45],
            }
        )
        actual = pd.DataFrame(
            {
                "interval_start": timestamps,
                "total_load_kw": [2.0, 2.0],
                "pv_kw": [0.0, 0.0],
                "buy_price_per_kwh": [0.2, 0.2],
                "sell_price_per_kwh": [0.1, 0.1],
            }
        )
        result = settle_schedule_on_actual(
            schedule,
            actual,
            parameters=parameters,
            degradation_cost_per_discharge_kwh=0.1,
        )
        self.assertAlmostEqual(result.iloc[0]["grid_net_kw"], 1.5)
        self.assertAlmostEqual(result.iloc[0]["grid_import_kwh"], 0.75)
        self.assertAlmostEqual(result.iloc[0]["degradation_cost"], 0.025)


if __name__ == "__main__":
    unittest.main()
