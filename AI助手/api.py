from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models import (
    EnergySummaryData,
    ErrorCode,
    ErrorData,
    Evidence,
    ExplanationData,
    ForecastData,
    GetEnergySummaryRequest,
    GetForecastRequest,
    GetSavingAdviceRequest,
    GetScheduleRequest,
    GetTariffRequest,
    KnowledgeData,
    ResolveUserScopeRequest,
    SavingAdviceData,
    ScheduleData,
    ScopeData,
    SearchKnowledgeRequest,
    SimulateScenarioRequest,
    TariffData,
    ToolEnvelope,
    ToolStatus,
    UserScope,
    ValidateUploadRequest,
    ExplainScheduleRequest,
)
from tool_service import ToolService


def create_app(service: ToolService | None = None) -> FastAPI:
    app = FastAPI(
        title="家庭光储荷 AI 助手工具 API",
        version="1.0.0",
        description="只读预测、电价、调度和知识工具；不重新训练或优化。",
    )
    tool_service = service or ToolService()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        envelope = ToolEnvelope[ErrorData](
            tool="request",
            status=ToolStatus.VALIDATION_ERROR,
            user_scope=UserScope.UNSUPPORTED,
            data=ErrorData(
                error_code=ErrorCode.MISSING_PARAMETER,
                message="请求缺少必要参数或参数格式错误",
                details={"errors": jsonable_encoder(exc.errors())},
            ),
            evidence=Evidence(
                source=str(request.url.path),
                data_version="request_schema_v1",
                generated_at=datetime.now(timezone.utc),
            ),
        )
        return JSONResponse(
            status_code=422,
            content=envelope.model_dump(mode="json"),
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.0.0"}

    @app.post(
        "/tools/resolve_user_scope",
        response_model=ToolEnvelope[ScopeData],
    )
    def resolve_user_scope(
        request: ResolveUserScopeRequest,
    ) -> ToolEnvelope[ScopeData]:
        return tool_service.resolve_user_scope(request)

    @app.post(
        "/tools/validate_upload",
        response_model=ToolEnvelope[ErrorData],
    )
    def validate_upload(
        request: ValidateUploadRequest,
    ) -> ToolEnvelope[ErrorData]:
        return tool_service.validate_upload(request)

    @app.post(
        "/tools/get_forecast",
        response_model=ToolEnvelope[ForecastData] | ToolEnvelope[ErrorData],
    )
    def get_forecast(
        request: GetForecastRequest,
    ) -> ToolEnvelope[ForecastData] | ToolEnvelope[ErrorData]:
        return tool_service.get_forecast(request)

    @app.post(
        "/tools/get_tariff",
        response_model=ToolEnvelope[TariffData] | ToolEnvelope[ErrorData],
    )
    def get_tariff(
        request: GetTariffRequest,
    ) -> ToolEnvelope[TariffData] | ToolEnvelope[ErrorData]:
        return tool_service.get_tariff(request)

    @app.post(
        "/tools/get_schedule",
        response_model=ToolEnvelope[ScheduleData] | ToolEnvelope[ErrorData],
    )
    def get_schedule(
        request: GetScheduleRequest,
    ) -> ToolEnvelope[ScheduleData] | ToolEnvelope[ErrorData]:
        return tool_service.get_schedule(request)

    @app.post(
        "/tools/explain_schedule",
        response_model=ToolEnvelope[ExplanationData] | ToolEnvelope[ErrorData],
    )
    def explain_schedule(
        request: ExplainScheduleRequest,
    ) -> ToolEnvelope[ExplanationData] | ToolEnvelope[ErrorData]:
        return tool_service.explain_schedule(request)

    @app.post(
        "/tools/get_energy_summary",
        response_model=ToolEnvelope[EnergySummaryData] | ToolEnvelope[ErrorData],
    )
    def get_energy_summary(
        request: GetEnergySummaryRequest,
    ) -> ToolEnvelope[EnergySummaryData] | ToolEnvelope[ErrorData]:
        return tool_service.get_energy_summary(request)

    @app.post(
        "/tools/simulate_scenario",
        response_model=ToolEnvelope[ErrorData],
    )
    def simulate_scenario(
        request: SimulateScenarioRequest,
    ) -> ToolEnvelope[ErrorData]:
        return tool_service.simulate_scenario(request)

    @app.post(
        "/tools/get_saving_advice",
        response_model=ToolEnvelope[SavingAdviceData] | ToolEnvelope[ErrorData],
    )
    def get_saving_advice(
        request: GetSavingAdviceRequest,
    ) -> ToolEnvelope[SavingAdviceData] | ToolEnvelope[ErrorData]:
        return tool_service.get_saving_advice(request)

    @app.post(
        "/tools/search_knowledge",
        response_model=ToolEnvelope[KnowledgeData],
    )
    def search_knowledge(
        request: SearchKnowledgeRequest,
    ) -> ToolEnvelope[KnowledgeData]:
        return tool_service.search_knowledge(request)

    return app


app = create_app()
