from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UserScope(str, Enum):
    EXISTING_300 = "existing_300"
    COLD_START = "cold_start"
    UNSUPPORTED = "unsupported"


class ToolStatus(str, Enum):
    OK = "ok"
    CLARIFICATION_REQUIRED = "clarification_required"
    NOT_FOUND = "not_found"
    NOT_SUPPORTED = "not_supported"
    VALIDATION_ERROR = "validation_error"
    INTERNAL_ERROR = "internal_error"


class ErrorCode(str, Enum):
    NO_DATA = "NO_DATA"
    DATE_OUT_OF_RANGE = "DATE_OUT_OF_RANGE"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNSUPPORTED_TARGET = "UNSUPPORTED_TARGET"
    TOOL_NOT_IMPLEMENTED = "TOOL_NOT_IMPLEMENTED"


class Evidence(StrictModel):
    source: str
    data_version: str
    model_version: str | None = None
    generated_at: datetime


class ErrorData(StrictModel):
    error_code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


T = TypeVar("T")


class ToolEnvelope(StrictModel, Generic[T]):
    tool: str
    tool_version: Literal["1.0"] = "1.0"
    status: ToolStatus
    user_scope: UserScope
    data: T
    evidence: Evidence
    warnings: list[str] = Field(default_factory=list)


class ResolveUserScopeRequest(StrictModel):
    customer_id: int | None = None
    dataset_id: str | None = None
    dataset_validated: bool = False


class ScopeData(StrictModel):
    user_scope: UserScope
    reason_code: str
    requires_upload_validation: bool


class ValidateUploadRequest(StrictModel):
    dataset_id: str


class GetForecastRequest(StrictModel):
    customer_id: int | None = None
    dataset_id: str | None = None
    dataset_validated: bool = False
    date: date
    target: Literal["load", "pv", "net_load"]
    horizon: Literal["4h", "24h", "48_step"] = "48_step"


class ForecastRow(StrictModel):
    interval_start: datetime
    yhat: float
    yhat_lower: float | None = None
    yhat_upper: float | None = None


class ForecastData(StrictModel):
    customer_id: int
    target: str
    rows: list[ForecastRow]


class GetTariffRequest(StrictModel):
    customer_id: int | None = None
    date: date


class TariffRow(StrictModel):
    interval_start: datetime
    buy_price_per_kwh: float
    sell_price_per_kwh: float


class TariffData(StrictModel):
    currency: Literal["AUD"] = "AUD"
    rows: list[TariffRow]


class GetScheduleRequest(StrictModel):
    customer_id: int | None = None
    dataset_id: str | None = None
    dataset_validated: bool = False
    date: date
    strategy: Literal["final", "robust_milp", "rule_fallback"] = "final"


class ScheduleRow(StrictModel):
    customer_id: int
    interval_start: datetime
    schedule_date: date
    strategy: str
    is_fallback: bool
    load_kw: float
    pv_kw: float
    buy_price_per_kwh: float
    sell_price_per_kwh: float
    charge_kw: float
    discharge_kw: float
    battery_net_ac_kw: float
    soc_start: float
    soc_end: float
    grid_net_kw: float
    grid_import_kw: float
    grid_export_kw: float
    grid_import_kwh: float
    grid_export_kwh: float
    import_cost: float
    export_revenue: float
    net_cost: float
    degradation_cost: float
    total_cost: float


class ScheduleData(StrictModel):
    schedule_schema_version: Literal["1.0"] = "1.0"
    rows: list[ScheduleRow]


class ExplainScheduleRequest(StrictModel):
    customer_id: int | None = None
    dataset_id: str | None = None
    dataset_validated: bool = False
    interval_start: datetime


class ExplanationEvidence(StrictModel):
    load_kw: float
    pv_kw: float
    buy_price_per_kwh: float
    soc_start: float
    soc_end: float


class ExplanationData(StrictModel):
    strategy: str
    is_fallback: bool
    reasons: list[str]
    evidence_values: ExplanationEvidence


class GetEnergySummaryRequest(StrictModel):
    customer_id: int | None = None
    dataset_id: str | None = None
    dataset_validated: bool = False
    date: date


class EnergySummaryData(StrictModel):
    load_kwh: float
    pv_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    net_cost: float
    degradation_cost: float
    total_cost: float
    final_soc: float


class SimulateScenarioRequest(StrictModel):
    customer_id: int | None = None
    dataset_id: str | None = None
    dataset_validated: bool = False
    date: date
    scenario: dict[str, Any]


class ScenarioResultData(StrictModel):
    scenario_echo: dict[str, Any]
    total_cost: float
    cost_delta: float
    solver_status: str


class GetSavingAdviceRequest(StrictModel):
    customer_id: int | None = None
    dataset_id: str | None = None
    dataset_validated: bool = False
    date: date


class SavingAdviceItem(StrictModel):
    code: str
    title: str
    explanation: str


class SavingAdviceData(StrictModel):
    advice_items: list[SavingAdviceItem]
    evidence: list[str]


class SearchKnowledgeRequest(StrictModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class KnowledgeDocument(StrictModel):
    doc_id: str
    title: str
    content: str
    tags: list[str]


class KnowledgeData(StrictModel):
    documents: list[KnowledgeDocument]
