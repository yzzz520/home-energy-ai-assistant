from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


FEATURE_NAMES = (
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
)


def build_feature_matrix(
    frame: "pd.DataFrame",
    *,
    timestamp_col: str = "forecast_timestamp",
    target_col: str = "actual",
) -> tuple[np.ndarray, np.ndarray | None]:
    import pandas as pd

    timestamps = pd.to_datetime(frame[timestamp_col])
    slot_index = timestamps.dt.hour * 2 + timestamps.dt.minute // 30
    slot_angle = 2 * np.pi * slot_index / 48
    weekday_angle = 2 * np.pi * timestamps.dt.day_of_week / 7

    features = np.column_stack(
        [
            np.sin(slot_angle),
            np.cos(slot_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            timestamps.dt.day_of_week.ge(5).astype(float),
            frame["seasonal_naive_daily"].astype(float),
            frame["seasonal_naive_weekly"].astype(float),
        ]
    ).astype(np.float32)
    target = (
        frame[target_col].astype(float).to_numpy(dtype=np.float32)
        if target_col in frame
        else None
    )
    return features, target


def interval_feature_vector(
    *,
    slot_index: int,
    weekday: int,
    seasonal_naive_daily: float,
    seasonal_naive_weekly: float,
) -> np.ndarray:
    if not 0 <= slot_index <= 47:
        raise ValueError("slot_index must be between 0 and 47")
    if not 0 <= weekday <= 6:
        raise ValueError("weekday must be between 0 and 6")

    slot_angle = 2 * np.pi * slot_index / 48
    weekday_angle = 2 * np.pi * weekday / 7
    return np.asarray(
        [
            np.sin(slot_angle),
            np.cos(slot_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            float(weekday >= 5),
            seasonal_naive_daily,
            seasonal_naive_weekly,
        ],
        dtype=np.float32,
    )


def regression_metrics(
    actual: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    if actual_values.shape != prediction_values.shape:
        raise ValueError("actual and prediction must have the same shape")
    if actual_values.size == 0:
        raise ValueError("metrics require at least one value")

    absolute_error = np.abs(actual_values - prediction_values)
    actual_sum = float(np.abs(actual_values).sum())
    return {
        "mae_kwh": float(absolute_error.mean()),
        "rmse_kwh": float(np.sqrt(np.square(actual_values - prediction_values).mean())),
        "wape_pct": (
            float(absolute_error.sum() / actual_sum * 100)
            if actual_sum > 0
            else 0.0
        ),
        "mean_bias_kwh": float((prediction_values - actual_values).mean()),
    }


def conversion_delta(
    reference: Sequence[float] | np.ndarray,
    converted: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    reference_values = np.asarray(reference, dtype=np.float64)
    converted_values = np.asarray(converted, dtype=np.float64)
    if reference_values.shape != converted_values.shape:
        raise ValueError("reference and converted must have the same shape")
    if reference_values.size == 0:
        raise ValueError("conversion comparison requires at least one value")

    difference = converted_values - reference_values
    return {
        "mae_kwh": float(np.abs(difference).mean()),
        "rmse_kwh": float(np.sqrt(np.square(difference).mean())),
        "max_abs_kwh": float(np.abs(difference).max()),
    }


def latency_summary(samples_ms: Sequence[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("latency summary requires at least one sample")
    mean_ms = float(values.mean())
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_per_second": (
            float(1000 / mean_ms) if mean_ms > 0 else 0.0
        ),
    }
