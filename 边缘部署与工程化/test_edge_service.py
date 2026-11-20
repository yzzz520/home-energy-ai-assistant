from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from edge_service import create_app


class FakeRunner:
    def predict(self, features: np.ndarray) -> float:
        assert features.shape == (7,)
        return 0.42


def test_edge_service_health_and_prediction() -> None:
    client = TestClient(create_app(runner=FakeRunner()))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/predict",
        json={
            "slot_index": 36,
            "weekday": 1,
            "seasonal_naive_daily_kwh": 0.4,
            "seasonal_naive_weekly_kwh": 0.38,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["load_kwh"] == 0.42
    assert payload["inference_ms"] >= 0


def test_edge_service_rejects_invalid_slot() -> None:
    client = TestClient(create_app(runner=FakeRunner()))
    response = client.post(
        "/predict",
        json={
            "slot_index": 48,
            "weekday": 1,
            "seasonal_naive_daily_kwh": 0.4,
            "seasonal_naive_weekly_kwh": 0.38,
        },
    )
    assert response.status_code == 422


def test_docs_use_local_swagger_assets() -> None:
    client = TestClient(create_app(runner=FakeRunner()))
    response = client.get("/docs")

    assert response.status_code == 200
    assert "cdn.jsdelivr.net" not in response.text
    assert "/static/swagger-ui/swagger-ui-bundle.js" in response.text
    assert client.get("/static/swagger-ui/swagger-ui-bundle.js").status_code == 200
    assert client.get("/openapi.json").status_code == 200
