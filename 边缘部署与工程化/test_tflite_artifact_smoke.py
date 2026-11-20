from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edge_features import interval_feature_vector


ARTIFACT = Path(__file__).with_name("artifacts") / "load_mlp_float32.tflite"


def _runtime_available() -> bool:
    try:
        import ai_edge_litert.interpreter  # noqa: F401

        return True
    except ImportError:
        try:
            import tensorflow  # noqa: F401

            return True
        except ImportError:
            return False


@pytest.mark.tflite
def test_generated_tflite_artifact_runs() -> None:
    if not ARTIFACT.exists():
        pytest.skip("TFLite artifact has not been generated")
    if not _runtime_available():
        pytest.skip("TFLite runtime is not installed")

    from edge_service import TFLiteRunner

    runner = TFLiteRunner(ARTIFACT)
    features = interval_feature_vector(
        slot_index=36,
        weekday=1,
        seasonal_naive_daily=0.4,
        seasonal_naive_weekly=0.38,
    )
    prediction = runner.predict(features)
    assert np.isfinite(prediction)
    assert prediction >= 0
