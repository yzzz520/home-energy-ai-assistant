from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from database import RequestLogDatabase
from service_api import create_service_app
from tool_service import ToolService


@pytest.fixture(scope="module")
def database_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("request_logs") / "requests.sqlite3"


@pytest.fixture(scope="module")
def client(database_path: Path) -> TestClient:
    app = create_service_app(
        database_path=database_path,
        service=ToolService(),
    )
    with TestClient(app) as test_client:
        yield test_client


def find_log(
    database: RequestLogDatabase,
    *,
    request_id: str,
) -> dict[str, object]:
    rows = database.list_request_logs(limit=200)
    return next(row for row in rows if row["request_id"] == request_id)


def test_successful_request_is_logged(client: TestClient, database_path: Path) -> None:
    response = client.post(
        "/tools/get_schedule",
        json={"customer_id": 1, "date": "2011-05-16", "strategy": "final"},
    )

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert request_id
    assert float(response.headers["X-Process-Time-Ms"]) >= 0

    log = find_log(RequestLogDatabase(database_path), request_id=request_id)
    assert log["method"] == "POST"
    assert log["path"] == "/tools/get_schedule"
    assert log["status_code"] == 200
    assert log["client_host"] == "testclient"
    assert log["error_code"] is None


def test_validation_error_is_logged(client: TestClient, database_path: Path) -> None:
    response = client.post(
        "/tools/get_forecast",
        json={"customer_id": 1, "target": "load", "horizon": "4h"},
    )

    assert response.status_code == 422
    assert response.json()["data"]["error_code"] == "MISSING_PARAMETER"
    log = find_log(
        RequestLogDatabase(database_path),
        request_id=response.headers["X-Request-ID"],
    )
    assert log["status_code"] == 422
    assert log["error_code"] == "MISSING_PARAMETER"


def test_tool_error_envelope_is_logged(
    client: TestClient,
    database_path: Path,
) -> None:
    response = client.post(
        "/tools/get_schedule",
        json={"customer_id": 999, "date": "2011-05-16", "strategy": "final"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["error_code"] == "OUT_OF_SCOPE"
    log = find_log(
        RequestLogDatabase(database_path),
        request_id=response.headers["X-Request-ID"],
    )
    assert log["status_code"] == 200
    assert log["error_code"] == "OUT_OF_SCOPE"


def test_request_ids_are_unique(client: TestClient) -> None:
    first = client.get("/health")
    second = client.get("/health")
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


def test_logs_survive_database_reopen(
    client: TestClient,
    database_path: Path,
) -> None:
    response = client.get("/service/health")
    request_id = response.headers["X-Request-ID"]

    reopened = RequestLogDatabase(database_path)
    log = find_log(reopened, request_id=request_id)
    assert log["path"] == "/service/health"
    assert reopened.ping() is True


def test_request_log_query_endpoint(client: TestClient) -> None:
    response = client.get(
        "/service/request-logs",
        params={"limit": 10, "error_code": "MISSING_PARAMETER"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert all(
        item["error_code"] == "MISSING_PARAMETER"
        for item in payload["items"]
    )


def test_docs_smoke_test(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/service/request-logs" in openapi.json()["paths"]
