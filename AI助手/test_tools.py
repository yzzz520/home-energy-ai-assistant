from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import create_app
from models import (
    ErrorCode,
    ErrorData,
    GetEnergySummaryRequest,
    GetForecastRequest,
    GetSavingAdviceRequest,
    GetScheduleRequest,
    GetTariffRequest,
    ResolveUserScopeRequest,
    SearchKnowledgeRequest,
    SimulateScenarioRequest,
    ToolEnvelope,
    ToolStatus,
)
from tool_service import ToolService


@pytest.fixture(scope="module")
def service() -> ToolService:
    return ToolService()


@pytest.fixture(scope="module")
def client(service: ToolService) -> TestClient:
    return TestClient(create_app(service))


def assert_error_code(
    result: ToolEnvelope,
    expected: ErrorCode,
    expected_status: ToolStatus,
) -> None:
    assert isinstance(result.data, ErrorData)
    assert result.data.error_code == expected
    assert result.status == expected_status


def test_resolve_user_scope(service: ToolService) -> None:
    result = service.resolve_user_scope(
        ResolveUserScopeRequest(customer_id=1)
    )
    assert result.status == ToolStatus.OK
    assert result.data.user_scope.value == "existing_300"


def test_forecast_returns_pydantic_rows(service: ToolService) -> None:
    result = service.get_forecast(
        GetForecastRequest(
            customer_id=1,
            date="2011-05-16",
            target="load",
            horizon="4h",
        )
    )
    assert result.status == ToolStatus.OK
    assert len(result.data.rows) == 8


def test_tariff_returns_48_rows(service: ToolService) -> None:
    result = service.get_tariff(
        GetTariffRequest(customer_id=1, date="2011-05-16")
    )
    assert result.status == ToolStatus.OK
    assert len(result.data.rows) == 48


def test_schedule_returns_selected_strategy(service: ToolService) -> None:
    result = service.get_schedule(
        GetScheduleRequest(customer_id=1, date="2011-05-16")
    )
    assert result.status == ToolStatus.OK
    assert len(result.data.rows) == 48
    assert result.data.rows[0].strategy in {"robust_milp", "rule_fallback"}


def test_energy_summary_and_advice(service: ToolService) -> None:
    summary = service.get_energy_summary(
        GetEnergySummaryRequest(customer_id=1, date="2011-05-16")
    )
    advice = service.get_saving_advice(
        GetSavingAdviceRequest(customer_id=1, date="2011-05-16")
    )
    assert summary.status == ToolStatus.OK
    assert advice.status == ToolStatus.OK
    assert advice.data.advice_items


def test_knowledge_tool(service: ToolService) -> None:
    result = service.search_knowledge(
        SearchKnowledgeRequest(query="SOC 安全边界", top_k=3)
    )
    assert result.status == ToolStatus.OK
    assert result.data.documents
    assert result.data.documents[0].doc_id == "battery-soc-safety"


def test_no_data_error(service: ToolService) -> None:
    result = service.get_schedule(
        GetScheduleRequest(customer_id=2, date="2011-05-16")
    )
    assert_error_code(result, ErrorCode.NO_DATA, ToolStatus.NOT_FOUND)


def test_date_out_of_range_error(service: ToolService) -> None:
    result = service.get_schedule(
        GetScheduleRequest(customer_id=1, date="2010-01-01")
    )
    assert_error_code(
        result,
        ErrorCode.DATE_OUT_OF_RANGE,
        ToolStatus.NOT_FOUND,
    )


def test_missing_parameter_error(service: ToolService) -> None:
    result = service.get_forecast(
        GetForecastRequest(
            date="2011-05-16",
            target="load",
            horizon="4h",
        )
    )
    assert_error_code(
        result,
        ErrorCode.MISSING_PARAMETER,
        ToolStatus.VALIDATION_ERROR,
    )


def test_out_of_scope_error(service: ToolService) -> None:
    result = service.get_schedule(
        GetScheduleRequest(customer_id=999, date="2011-05-16")
    )
    assert_error_code(
        result,
        ErrorCode.OUT_OF_SCOPE,
        ToolStatus.NOT_SUPPORTED,
    )


def test_unsupported_target_error(service: ToolService) -> None:
    result = service.get_forecast(
        GetForecastRequest(
            customer_id=1,
            date="2011-05-16",
            target="pv",
            horizon="4h",
        )
    )
    assert_error_code(
        result,
        ErrorCode.UNSUPPORTED_TARGET,
        ToolStatus.NOT_SUPPORTED,
    )


def test_tool_not_implemented_error(service: ToolService) -> None:
    result = service.simulate_scenario(
        SimulateScenarioRequest(
            customer_id=1,
            date="2011-05-16",
            scenario={"pv": 0},
        )
    )
    assert_error_code(
        result,
        ErrorCode.TOOL_NOT_IMPLEMENTED,
        ToolStatus.NOT_SUPPORTED,
    )


def test_upload_without_record_is_no_data(service: ToolService) -> None:
    from models import ValidateUploadRequest

    result = service.validate_upload(
        ValidateUploadRequest(dataset_id="missing-dataset")
    )
    assert_error_code(result, ErrorCode.NO_DATA, ToolStatus.NOT_FOUND)


def test_contract_error_codes_are_frozen() -> None:
    path = Path(__file__).with_name("contracts") / "tools.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    frozen = set(contract["envelope"]["error_codes"])
    implemented = {item.value for item in ErrorCode}
    assert frozen == implemented


def test_schedule_numbers_match_parquet(service: ToolService) -> None:
    raw = pd.read_parquet(service.schedule_path)
    raw["interval_start"] = pd.to_datetime(raw["interval_start"])
    target = raw.sort_values("interval_start").iloc[0]
    result = service.get_schedule(
        GetScheduleRequest(customer_id=1, date=str(target["schedule_date"]))
    )
    tool_row = next(
        row
        for row in result.data.rows
        if row.interval_start == target["interval_start"].to_pydatetime()
    )
    assert tool_row.load_kw == target["load_kw"]
    assert tool_row.charge_kw == target["charge_kw"]
    assert tool_row.discharge_kw == target["discharge_kw"]
    assert tool_row.soc_end == target["soc_end"]
    assert tool_row.total_cost == target["total_cost"]


def test_energy_summary_numbers_match_parquet(service: ToolService) -> None:
    raw = pd.read_parquet(service.schedule_path)
    target_date = raw["schedule_date"].min()
    selected = raw.loc[raw["schedule_date"].eq(target_date)]
    result = service.get_energy_summary(
        GetEnergySummaryRequest(customer_id=1, date=str(target_date))
    )
    assert result.data.load_kwh == pytest.approx(
        selected["load_kw"].sum() * 0.5
    )
    assert result.data.grid_import_kwh == pytest.approx(
        selected["grid_import_kwh"].sum()
    )
    assert result.data.total_cost == pytest.approx(
        selected["total_cost"].sum()
    )
    assert result.data.final_soc == selected.sort_values("interval_start")[
        "soc_end"
    ].iloc[-1]


def test_forecast_number_matches_source(service: ToolService) -> None:
    source = pd.read_csv(
        service.forecast_path,
        encoding="utf-8-sig",
        usecols=["forecast_timestamp", "Customer", "lightgbm"],
    )
    source["interval_start"] = pd.to_datetime(source["forecast_timestamp"])
    target = source.loc[
        source["Customer"].eq(1)
        & source["interval_start"].dt.date.eq(
            pd.Timestamp("2011-05-16").date()
        )
    ].sort_values("interval_start").iloc[0]
    result = service.get_forecast(
        GetForecastRequest(
            customer_id=1,
            date="2011-05-16",
            target="load",
            horizon="4h",
        )
    )
    assert result.data.rows[0].interval_start == target[
        "interval_start"
    ].to_pydatetime()
    assert result.data.rows[0].yhat == target["lightgbm"]


def test_fastapi_docs_and_health(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/tools/get_schedule" in openapi.json()["paths"]


def test_fastapi_schedule_endpoint(client: TestClient) -> None:
    response = client.post(
        "/tools/get_schedule",
        json={"customer_id": 1, "date": "2011-05-16", "strategy": "final"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["evidence"]["source"] == "schedule.parquet"
    assert len(payload["data"]["rows"]) == 48


def test_fastapi_request_validation_uses_frozen_error_code(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/get_forecast",
        json={"customer_id": 1, "target": "load", "horizon": "4h"},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "validation_error"
    assert payload["data"]["error_code"] == "MISSING_PARAMETER"


def test_extra_input_field_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/tools/get_schedule",
        json={
            "customer_id": 1,
            "date": "2011-05-16",
            "unexpected": "x",
        },
    )
    assert response.status_code == 422
