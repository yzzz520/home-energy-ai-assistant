from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from edge_features import (
    FEATURE_NAMES,
    build_feature_matrix,
    conversion_delta,
    interval_feature_vector,
    latency_summary,
    regression_metrics,
)


def test_feature_matrix_has_stable_shape_and_order() -> None:
    frame = pd.DataFrame(
        {
            "forecast_timestamp": [
                "2024-01-01 00:00",
                "2024-01-06 12:00",
            ],
            "actual": [0.2, 0.8],
            "seasonal_naive_daily": [0.18, 0.75],
            "seasonal_naive_weekly": [0.22, 0.70],
        }
    )
    features, target = build_feature_matrix(frame)

    assert features.shape == (2, len(FEATURE_NAMES))
    assert features.dtype == np.float32
    assert target is not None
    assert target.tolist() == pytest.approx([0.2, 0.8])
    assert features[0].tolist() == pytest.approx(
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.18, 0.22]
    )
    assert features[1][4] == 1.0


def test_interval_vector_matches_training_contract() -> None:
    features = interval_feature_vector(
        slot_index=36,
        weekday=5,
        seasonal_naive_daily=0.6,
        seasonal_naive_weekly=0.55,
    )
    assert len(features) == len(FEATURE_NAMES)
    assert features[0] == pytest.approx(math.sin(1.5 * math.pi))
    assert features[1] == pytest.approx(math.cos(1.5 * math.pi), abs=1e-6)
    assert features[4] == 1.0
    assert features[5] == pytest.approx(0.6)
    assert features[6] == pytest.approx(0.55)


def test_metric_and_latency_helpers() -> None:
    metrics = regression_metrics([1.0, 2.0], [1.5, 1.0])
    assert metrics["mae_kwh"] == pytest.approx(0.75)
    assert metrics["mean_bias_kwh"] == pytest.approx(-0.25)
    assert metrics["wape_pct"] == pytest.approx(50.0)

    converted = conversion_delta([1.0, 2.0], [1.1, 1.8])
    assert converted["mae_kwh"] == pytest.approx(0.15)
    assert converted["max_abs_kwh"] == pytest.approx(0.2)

    latency = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert latency["mean_ms"] == pytest.approx(2.5)
    assert latency["median_ms"] == pytest.approx(2.5)
