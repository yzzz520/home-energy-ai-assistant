from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from eda_load_curve import (
    average_profile_metrics,
    build_average_daily_profile,
    build_seasonal_gc_weekday_profile,
    build_seasonal_weekday_profile,
    daily_metrics,
    gc_cl_seasonal_metrics,
    plot_daily_curve,
    plot_average_daily_profile,
    plot_seasonal_gc_weekday_profile,
    plot_seasonal_weekday_profile,
    prepare_daily_curve,
    seasonal_profile_metrics,
)
from pipeline import process_customer_year
from reader import AusgridReader


class DailyLoadCurveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = AusgridReader()
        cls.clean = process_customer_year(
            cls.reader,
            customer_id=1,
            source_file="Solar home 2010-2011.csv",
        )
        cls.day = prepare_daily_curve(cls.clean, source_date="2010-07-01")

    def test_one_day_has_48_points_and_power_columns(self) -> None:
        self.assertEqual(len(self.day), 48)
        expected = {
            "gc_kw",
            "cl_kw",
            "total_load_kw",
            "pv_kw",
            "net_grid_kw",
        }
        self.assertTrue(expected.issubset(self.day.columns))

    def test_daily_metrics_are_correct(self) -> None:
        metrics = daily_metrics(self.day)
        self.assertAlmostEqual(metrics["total_load_kwh"], 39.929001, places=6)
        self.assertAlmostEqual(metrics["pv_generation_kwh"], 10.02, places=6)
        self.assertAlmostEqual(metrics["import_kwh"], 30.964001, places=6)
        self.assertAlmostEqual(metrics["export_kwh"], 1.055, places=6)

    def test_load_curve_figure_can_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "daily_load_curve.png"
            plot_daily_curve(
                self.day,
                customer_id=1,
                source_date="2010-07-01",
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 20_000)

    def test_average_profile_has_weekday_and_weekend_48_slots(self) -> None:
        profile = build_average_daily_profile(self.clean)
        counts = profile.groupby("day_type", observed=True).size().to_dict()
        self.assertEqual(counts, {"工作日": 48, "周末": 48})

    def test_average_profile_metrics_match_full_year(self) -> None:
        profile = build_average_daily_profile(self.clean)
        metrics = average_profile_metrics(self.clean, profile)
        self.assertEqual(metrics["工作日"]["days"], 261)
        self.assertEqual(metrics["周末"]["days"], 104)
        self.assertAlmostEqual(
            metrics["工作日"]["mean_daily_energy_kwh"],
            23.597273,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["周末"]["mean_daily_energy_kwh"],
            22.937202,
            places=6,
        )

    def test_average_profile_figure_can_be_written(self) -> None:
        profile = build_average_daily_profile(self.clean)
        metrics = average_profile_metrics(self.clean, profile)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "average_daily_profile.png"
            plot_average_daily_profile(
                profile,
                metrics=metrics,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 20_000)

    def test_seasonal_weekday_profile_has_four_seasons(self) -> None:
        profile = build_seasonal_weekday_profile(self.clean)
        counts = profile.groupby("season", observed=True).size().to_dict()
        self.assertEqual(
            counts,
            {"夏季": 48, "秋季": 48, "冬季": 48, "春季": 48},
        )

    def test_seasonal_weekday_figure_can_be_written(self) -> None:
        profile = build_seasonal_weekday_profile(self.clean)
        metrics = seasonal_profile_metrics(self.clean, profile)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "seasonal_weekday_profile.png"
            plot_seasonal_weekday_profile(
                profile,
                metrics=metrics,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 20_000)

    def test_gc_only_profile_has_four_seasons(self) -> None:
        profile = build_seasonal_gc_weekday_profile(self.clean)
        counts = profile.groupby("season", observed=True).size().to_dict()
        self.assertEqual(
            counts,
            {"夏季": 48, "秋季": 48, "冬季": 48, "春季": 48},
        )

    def test_gc_only_figure_can_be_written(self) -> None:
        profile = build_seasonal_gc_weekday_profile(self.clean)
        metrics = gc_cl_seasonal_metrics(self.clean, profile)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "seasonal_gc_weekday_profile.png"
            plot_seasonal_gc_weekday_profile(
                profile,
                metrics=metrics,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
