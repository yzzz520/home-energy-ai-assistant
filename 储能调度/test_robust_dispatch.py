from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from robust_dispatch import (
    SCHEDULE_COLUMNS,
    apply_conservative_scenario,
    calibrate_hourly_buffers,
    finalize_schedule,
    should_fallback_to_rule,
)


class RobustDispatchTest(unittest.TestCase):
    def test_conservative_scenario_raises_load_and_lowers_pv(self) -> None:
        day = pd.DataFrame(
            {
                "hour": [10, 17],
                "lightgbm": [1.0, 2.0],
                "forecast_pv_kwh": [2.0, 0.5],
            }
        )
        load_buffer = pd.Series({10: 0.1, 17: 0.3})
        pv_buffer = pd.Series({10: 0.2, 17: 0.1})
        result = apply_conservative_scenario(
            day,
            load_buffer=load_buffer,
            pv_buffer=pv_buffer,
        )
        self.assertAlmostEqual(result.iloc[0]["robust_load_kwh"], 1.0)
        self.assertAlmostEqual(result.iloc[1]["robust_load_kwh"], 2.3)
        self.assertAlmostEqual(result.iloc[0]["robust_pv_kwh"], 1.8)
        self.assertAlmostEqual(result.iloc[1]["robust_pv_kwh"], 0.4)

    def test_buffers_use_prior_one_sided_errors(self) -> None:
        history = pd.DataFrame(
            {
                "hour": [17, 17, 17],
                "actual": [2.0, 3.0, 4.0],
                "lightgbm": [1.8, 2.5, 3.5],
                "forecast_pv_kwh": [0.1, 0.2, 0.3],
                "actual_pv_kwh": [0.2, 0.1, 0.1],
            }
        )
        load_buffer, pv_buffer = calibrate_hourly_buffers(history)
        self.assertGreater(load_buffer.loc[17], 0)
        self.assertGreater(pv_buffer.loc[17], 0)

    def test_fallback_threshold(self) -> None:
        self.assertTrue(
            should_fallback_to_rule(10.0, 8.5, minimum_advantage_pct=20)
        )
        self.assertFalse(
            should_fallback_to_rule(10.0, 7.0, minimum_advantage_pct=20)
        )

    def test_schedule_schema_is_frozen(self) -> None:
        frame = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(
                    ["2024-01-01 00:00", "2024-01-01 00:30"]
                ),
                "total_load_kw": [1.0, 1.1],
                "pv_kw": [0.0, 0.1],
                "buy_price_per_kwh": [0.2, 0.2],
                "sell_price_per_kwh": [0.1, 0.1],
                "charge_kw": [0.0, 0.0],
                "discharge_kw": [0.0, 0.0],
                "battery_net_ac_kw": [0.0, 0.0],
                "soc_start": [0.5, 0.5],
                "soc_end": [0.5, 0.5],
                "grid_net_kw": [1.0, 1.0],
                "grid_import_kw": [1.0, 1.0],
                "grid_export_kw": [0.0, 0.0],
                "grid_import_kwh": [0.5, 0.5],
                "grid_export_kwh": [0.0, 0.0],
                "import_cost": [0.1, 0.1],
                "export_revenue": [0.0, 0.0],
                "net_cost": [0.1, 0.1],
                "degradation_cost": [0.0, 0.0],
                "total_cost": [0.1, 0.1],
                "strategy": ["rule_fallback", "robust_milp"],
                "is_fallback": [True, False],
            }
        )
        schedule = finalize_schedule(frame)
        self.assertEqual(schedule.columns.tolist(), SCHEDULE_COLUMNS)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "schedule.parquet"
            schedule.to_parquet(path, index=False)
            loaded = pd.read_parquet(path)
            self.assertEqual(loaded.columns.tolist(), SCHEDULE_COLUMNS)
            parquet_schema = pq.ParquetFile(path).schema_arrow
            self.assertEqual(
                parquet_schema.field("interval_start").type,
                pa.timestamp("ns"),
            )
            self.assertEqual(
                parquet_schema.field("schedule_date").type,
                pa.date32(),
            )
            self.assertEqual(
                parquet_schema.field("customer_id").type,
                pa.int32(),
            )


if __name__ == "__main__":
    unittest.main()
