from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from models import (
    EnergySummaryData,
    ErrorCode,
    ErrorData,
    Evidence,
    ExplanationData,
    ExplanationEvidence,
    ForecastData,
    ForecastRow,
    GetEnergySummaryRequest,
    GetForecastRequest,
    GetSavingAdviceRequest,
    GetScheduleRequest,
    GetTariffRequest,
    KnowledgeData,
    KnowledgeDocument,
    ResolveUserScopeRequest,
    SavingAdviceData,
    SavingAdviceItem,
    ScheduleData,
    ScheduleRow,
    ScopeData,
    SearchKnowledgeRequest,
    SimulateScenarioRequest,
    TariffData,
    TariffRow,
    ToolEnvelope,
    ToolStatus,
    UserScope,
    ValidateUploadRequest,
)
from scope import UserScopeResolver


DataT = TypeVar("DataT")


class ToolService:
    def __init__(self, *, project_root: str | Path | None = None) -> None:
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.root = root
        self.schedule_path = root / "储能调度" / "results" / "schedule.parquet"
        self.forecast_path = (
            root
            / "时序预测"
            / "results"
            / "validation_300_households_predictions.csv"
        )
        self.knowledge_path = Path(__file__).with_name("data") / "knowledge_base.json"
        if not self.schedule_path.exists():
            raise FileNotFoundError(f"schedule 文件不存在：{self.schedule_path}")
        if not self.forecast_path.exists():
            raise FileNotFoundError(f"预测文件不存在：{self.forecast_path}")
        if not self.knowledge_path.exists():
            raise FileNotFoundError(f"知识文件不存在：{self.knowledge_path}")

        self.schedule = pd.read_parquet(self.schedule_path)
        self.schedule["interval_start"] = pd.to_datetime(
            self.schedule["interval_start"]
        )
        self._forecast: pd.DataFrame | None = None
        self.scope_resolver = UserScopeResolver.from_config()
        self.knowledge = json.loads(
            self.knowledge_path.read_text(encoding="utf-8")
        )

    @property
    def forecast(self) -> pd.DataFrame:
        if self._forecast is None:
            frame = pd.read_csv(
                self.forecast_path,
                encoding="utf-8-sig",
                usecols=[
                    "forecast_timestamp",
                    "Customer",
                    "actual",
                    "lightgbm",
                ],
            )
            frame["interval_start"] = pd.to_datetime(
                frame["forecast_timestamp"]
            )
            self._forecast = frame
        return self._forecast

    @staticmethod
    def _evidence(
        source: str,
        data_version: str,
        model_version: str | None = None,
    ) -> Evidence:
        return Evidence(
            source=source,
            data_version=data_version,
            model_version=model_version,
            generated_at=datetime.now(timezone.utc),
        )

    def _ok(
        self,
        data: DataT,
        *,
        tool: str,
        user_scope: UserScope,
        source: str,
        data_version: str,
        model_version: str | None = None,
    ) -> ToolEnvelope[DataT]:
        return ToolEnvelope[type(data)](
            tool=tool,
            status=ToolStatus.OK,
            user_scope=user_scope,
            data=data,
            evidence=self._evidence(source, data_version, model_version),
        )

    def _error(
        self,
        *,
        tool: str,
        user_scope: UserScope,
        error_code: ErrorCode,
        message: str,
        status: ToolStatus,
        details: dict[str, Any] | None = None,
        source: str = "tool_service",
    ) -> ToolEnvelope[ErrorData]:
        return ToolEnvelope[ErrorData](
            tool=tool,
            status=status,
            user_scope=user_scope,
            data=ErrorData(
                error_code=error_code,
                message=message,
                details=details,
            ),
            evidence=self._evidence(source, "schema_1.0"),
        )

    def resolve_user_scope(
        self,
        request: ResolveUserScopeRequest,
    ) -> ToolEnvelope[ScopeData]:
        result = self.scope_resolver.resolve(
            customer_id=request.customer_id,
            dataset_id=request.dataset_id,
            dataset_validated=request.dataset_validated,
        )
        return self._ok(
            ScopeData(
                user_scope=UserScope(result.user_scope),
                reason_code=result.reason_code,
                requires_upload_validation=result.requires_upload_validation,
            ),
            tool="resolve_user_scope",
            user_scope=UserScope(result.user_scope),
            source="assistant_config.toml",
            data_version="scope_v1",
        )

    def validate_upload(
        self,
        request: ValidateUploadRequest,
    ) -> ToolEnvelope[ErrorData]:
        return self._error(
            tool="validate_upload",
            user_scope=UserScope.UNSUPPORTED,
            error_code=ErrorCode.NO_DATA,
            message=f"没有找到数据集 {request.dataset_id} 的上传记录",
            status=ToolStatus.NOT_FOUND,
            details={"dataset_id": request.dataset_id},
        )

    def get_forecast(
        self,
        request: GetForecastRequest,
    ) -> ToolEnvelope[ForecastData] | ToolEnvelope[ErrorData]:
        scope = self.scope_resolver.resolve(
            customer_id=request.customer_id,
            dataset_id=request.dataset_id,
            dataset_validated=request.dataset_validated,
        )
        user_scope = UserScope(scope.user_scope)
        if request.customer_id is None and request.dataset_id is None:
            return self._error(
                tool="get_forecast",
                user_scope=UserScope.UNSUPPORTED,
                error_code=ErrorCode.MISSING_PARAMETER,
                message="必须提供 customer_id 或 dataset_id",
                status=ToolStatus.VALIDATION_ERROR,
            )
        if user_scope != UserScope.EXISTING_300:
            return self._error(
                tool="get_forecast",
                user_scope=user_scope,
                error_code=ErrorCode.OUT_OF_SCOPE,
                message="当前预测工具只支持已有 300 户",
                status=ToolStatus.NOT_SUPPORTED,
                details={"reason_code": scope.reason_code},
            )
        if request.target != "load":
            return self._error(
                tool="get_forecast",
                user_scope=user_scope,
                error_code=ErrorCode.UNSUPPORTED_TARGET,
                message=f"预测目标 {request.target} 尚未接入只读预测文件",
                status=ToolStatus.NOT_SUPPORTED,
                details={"supported_targets": ["load"]},
            )

        frame = self.forecast
        available_dates = frame["interval_start"].dt.date
        date_min = available_dates.min()
        date_max = available_dates.max()
        if request.date < date_min or request.date > date_max:
            return self._error(
                tool="get_forecast",
                user_scope=user_scope,
                error_code=ErrorCode.DATE_OUT_OF_RANGE,
                message="请求日期超出预测数据范围",
                status=ToolStatus.NOT_FOUND,
                details={
                    "date_min": date_min.isoformat(),
                    "date_max": date_max.isoformat(),
                },
            )
        selected = frame.loc[
            frame["Customer"].eq(request.customer_id)
            & frame["interval_start"].dt.date.eq(request.date)
        ].sort_values("interval_start")
        if selected.empty:
            return self._error(
                tool="get_forecast",
                user_scope=user_scope,
                error_code=ErrorCode.NO_DATA,
                message="该客户在请求日期没有预测数据",
                status=ToolStatus.NOT_FOUND,
            )
        limit = 8 if request.horizon == "4h" else 48
        selected = selected.head(limit)
        data = ForecastData(
            customer_id=int(request.customer_id),
            target=request.target,
            rows=[
                ForecastRow(
                    interval_start=row.interval_start.to_pydatetime(),
                    yhat=float(row.lightgbm),
                )
                for row in selected.itertuples(index=False)
            ],
        )
        return self._ok(
            data,
            tool="get_forecast",
            user_scope=user_scope,
            source=self.forecast_path.name,
            data_version="walk_forward_300_v1",
            model_version="lightgbm_global_v1",
        )

    def _schedule_scope(
        self,
        *,
        tool: str,
        customer_id: int | None,
        dataset_id: str | None,
        dataset_validated: bool,
    ) -> tuple[UserScope, ToolEnvelope[ErrorData] | None]:
        if customer_id is None and dataset_id is None:
            return UserScope.UNSUPPORTED, self._error(
                tool=tool,
                user_scope=UserScope.UNSUPPORTED,
                error_code=ErrorCode.MISSING_PARAMETER,
                message="必须提供 customer_id 或 dataset_id",
                status=ToolStatus.VALIDATION_ERROR,
            )
        scope = self.scope_resolver.resolve(
            customer_id=customer_id,
            dataset_id=dataset_id,
            dataset_validated=dataset_validated,
        )
        user_scope = UserScope(scope.user_scope)
        if user_scope != UserScope.EXISTING_300:
            return user_scope, self._error(
                tool=tool,
                user_scope=user_scope,
                error_code=ErrorCode.OUT_OF_SCOPE,
                message="当前 schedule 只支持已有 300 户",
                status=ToolStatus.NOT_SUPPORTED,
                details={"reason_code": scope.reason_code},
            )
        return user_scope, None

    def get_tariff(
        self,
        request: GetTariffRequest,
    ) -> ToolEnvelope[TariffData] | ToolEnvelope[ErrorData]:
        customer_id = request.customer_id or 1
        user_scope, error = self._schedule_scope(
            tool="get_tariff",
            customer_id=customer_id,
            dataset_id=None,
            dataset_validated=False,
        )
        if error is not None:
            return error
        date_min = self.schedule["schedule_date"].min()
        date_max = self.schedule["schedule_date"].max()
        if request.date < date_min or request.date > date_max:
            return self._error(
                tool="get_tariff",
                user_scope=user_scope,
                error_code=ErrorCode.DATE_OUT_OF_RANGE,
                message="请求日期超出电价文件范围",
                status=ToolStatus.NOT_FOUND,
                details={
                    "date_min": date_min.isoformat(),
                    "date_max": date_max.isoformat(),
                },
            )
        selected = self.schedule.loc[
            self.schedule["schedule_date"].eq(request.date)
        ].sort_values("interval_start")
        if selected.empty:
            return self._error(
                tool="get_tariff",
                user_scope=user_scope,
                error_code=ErrorCode.NO_DATA,
                message="请求日期没有电价数据",
                status=ToolStatus.NOT_FOUND,
            )
        return self._ok(
            TariffData(
                rows=[
                    TariffRow(
                        interval_start=row.interval_start.to_pydatetime(),
                        buy_price_per_kwh=float(row.buy_price_per_kwh),
                        sell_price_per_kwh=float(row.sell_price_per_kwh),
                    )
                    for row in selected.itertuples(index=False)
                ]
            ),
            tool="get_tariff",
            user_scope=user_scope,
            source=self.schedule_path.name,
            data_version="schedule_schema_1.0",
        )

    def get_schedule(
        self,
        request: GetScheduleRequest,
    ) -> ToolEnvelope[ScheduleData] | ToolEnvelope[ErrorData]:
        user_scope, error = self._schedule_scope(
            tool="get_schedule",
            customer_id=request.customer_id,
            dataset_id=request.dataset_id,
            dataset_validated=request.dataset_validated,
        )
        if error is not None:
            return error
        date_min = self.schedule["schedule_date"].min()
        date_max = self.schedule["schedule_date"].max()
        if request.date < date_min or request.date > date_max:
            return self._error(
                tool="get_schedule",
                user_scope=user_scope,
                error_code=ErrorCode.DATE_OUT_OF_RANGE,
                message="请求日期超出 schedule 范围",
                status=ToolStatus.NOT_FOUND,
                details={
                    "date_min": date_min.isoformat(),
                    "date_max": date_max.isoformat(),
                },
            )
        selected = self.schedule.loc[
            self.schedule["customer_id"].eq(request.customer_id)
            & self.schedule["schedule_date"].eq(request.date)
        ].sort_values("interval_start")
        if request.strategy == "robust_milp":
            selected = selected.loc[~selected["is_fallback"]]
        elif request.strategy == "rule_fallback":
            selected = selected.loc[selected["is_fallback"]]
        if selected.empty:
            return self._error(
                tool="get_schedule",
                user_scope=user_scope,
                error_code=ErrorCode.NO_DATA,
                message="请求客户和日期没有对应调度数据",
                status=ToolStatus.NOT_FOUND,
            )
        rows = [
            ScheduleRow(
                **{
                    **row._asdict(),
                    "interval_start": row.interval_start.to_pydatetime(),
                }
            )
            for row in selected.itertuples(index=False)
        ]
        return self._ok(
            ScheduleData(rows=rows),
            tool="get_schedule",
            user_scope=user_scope,
            source=self.schedule_path.name,
            data_version="schedule_schema_1.0",
            model_version="robust_milp_rule_fallback_v1",
        )

    def explain_schedule(
        self,
        request: ExplainScheduleRequest,
    ) -> ToolEnvelope[ExplanationData] | ToolEnvelope[ErrorData]:
        user_scope, error = self._schedule_scope(
            tool="explain_schedule",
            customer_id=request.customer_id,
            dataset_id=request.dataset_id,
            dataset_validated=request.dataset_validated,
        )
        if error is not None:
            return error
        selected = self.schedule.loc[
            self.schedule["customer_id"].eq(request.customer_id)
            & self.schedule["interval_start"].eq(request.interval_start)
        ]
        if selected.empty:
            return self._error(
                tool="explain_schedule",
                user_scope=user_scope,
                error_code=ErrorCode.NO_DATA,
                message="请求客户和时段没有调度记录",
                status=ToolStatus.NOT_FOUND,
            )
        row = selected.iloc[0]
        reasons: list[str] = []
        if bool(row["is_fallback"]):
            reasons.append("预测收益不足，安全层回退规则调度")
        elif float(row["charge_kw"]) > 1e-6:
            reasons.append("低电价或光伏余量时段充电")
        elif float(row["discharge_kw"]) > 1e-6:
            reasons.append("高电价或晚峰时段放电")
        else:
            reasons.append("未触发充放电条件")
        return self._ok(
            ExplanationData(
                strategy=str(row["strategy"]),
                is_fallback=bool(row["is_fallback"]),
                reasons=reasons,
                evidence_values=ExplanationEvidence(
                    load_kw=float(row["load_kw"]),
                    pv_kw=float(row["pv_kw"]),
                    buy_price_per_kwh=float(row["buy_price_per_kwh"]),
                    soc_start=float(row["soc_start"]),
                    soc_end=float(row["soc_end"]),
                ),
            ),
            tool="explain_schedule",
            user_scope=user_scope,
            source=self.schedule_path.name,
            data_version="schedule_schema_1.0",
            model_version="selected_schedule_v1",
        )

    def get_energy_summary(
        self,
        request: GetEnergySummaryRequest,
    ) -> ToolEnvelope[EnergySummaryData] | ToolEnvelope[ErrorData]:
        user_scope, error = self._schedule_scope(
            tool="get_energy_summary",
            customer_id=request.customer_id,
            dataset_id=request.dataset_id,
            dataset_validated=request.dataset_validated,
        )
        if error is not None:
            return error
        selected = self.schedule.loc[
            self.schedule["customer_id"].eq(request.customer_id)
            & self.schedule["schedule_date"].eq(request.date)
        ].sort_values("interval_start")
        if selected.empty:
            return self._error(
                tool="get_energy_summary",
                user_scope=user_scope,
                error_code=ErrorCode.NO_DATA,
                message="请求客户和日期没有能源汇总数据",
                status=ToolStatus.NOT_FOUND,
            )
        return self._ok(
            EnergySummaryData(
                load_kwh=float(selected["load_kw"].sum() * 0.5),
                pv_kwh=float(selected["pv_kw"].sum() * 0.5),
                grid_import_kwh=float(selected["grid_import_kwh"].sum()),
                grid_export_kwh=float(selected["grid_export_kwh"].sum()),
                net_cost=float(selected["net_cost"].sum()),
                degradation_cost=float(selected["degradation_cost"].sum()),
                total_cost=float(selected["total_cost"].sum()),
                final_soc=float(selected["soc_end"].iloc[-1]),
            ),
            tool="get_energy_summary",
            user_scope=user_scope,
            source=self.schedule_path.name,
            data_version="schedule_schema_1.0",
        )

    def simulate_scenario(
        self,
        request: SimulateScenarioRequest,
    ) -> ToolEnvelope[ErrorData]:
        return self._error(
            tool="simulate_scenario",
            user_scope=UserScope.UNSUPPORTED,
            error_code=ErrorCode.TOOL_NOT_IMPLEMENTED,
            message="当前助手只读取预计算 schedule，不重新运行场景优化",
            status=ToolStatus.NOT_SUPPORTED,
            details={"scenario": request.scenario},
        )

    def get_saving_advice(
        self,
        request: GetSavingAdviceRequest,
    ) -> ToolEnvelope[SavingAdviceData] | ToolEnvelope[ErrorData]:
        summary_result = self.get_energy_summary(
            GetEnergySummaryRequest(
                customer_id=request.customer_id,
                dataset_id=request.dataset_id,
                dataset_validated=request.dataset_validated,
                date=request.date,
            )
        )
        if not isinstance(summary_result.data, EnergySummaryData):
            return summary_result
        summary = summary_result.data
        advice: list[SavingAdviceItem] = []
        if summary.grid_import_kwh > summary.pv_kwh:
            advice.append(
                SavingAdviceItem(
                    code="SHIFT_HIGH_LOAD",
                    title="减少高电价时段购电",
                    explanation="将可延迟负荷移到低电价时段，并保持在电池 SOC 安全边界内。",
                )
            )
        if summary.grid_export_kwh > 0:
            advice.append(
                SavingAdviceItem(
                    code="INCREASE_SELF_CONSUMPTION",
                    title="提高光伏自用",
                    explanation="优先用光伏供给本地负荷，减少低价上网。",
                )
            )
        if not advice:
            advice.append(
                SavingAdviceItem(
                    code="MAINTAIN_CURRENT",
                    title="维持当前用电方式",
                    explanation="当前数据没有显示明显的购售电优化空间。",
                )
            )
        return self._ok(
            SavingAdviceData(
                advice_items=advice,
                evidence=[
                    f"购电 {summary.grid_import_kwh:.3f} kWh",
                    f"光伏 {summary.pv_kwh:.3f} kWh",
                    f"售电 {summary.grid_export_kwh:.3f} kWh",
                ],
            ),
            tool="get_saving_advice",
            user_scope=summary_result.user_scope,
            source=self.schedule_path.name,
            data_version="schedule_schema_1.0",
        )

    def search_knowledge(
        self,
        request: SearchKnowledgeRequest,
    ) -> ToolEnvelope[KnowledgeData]:
        query = request.query.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for document in self.knowledge:
            text = (
                document["title"]
                + " "
                + document["content"]
                + " "
                + " ".join(document["tags"])
            ).lower()
            score = sum(
                1
                for token in query.split()
                if token and token in text
            )
            score += sum(1 for tag in document["tags"] if tag.lower() in query)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1]["doc_id"]))
        documents = [
            KnowledgeDocument(**document)
            for _, document in scored[: request.top_k]
        ]
        return self._ok(
            KnowledgeData(documents=documents),
            tool="search_knowledge",
            user_scope=UserScope.EXISTING_300,
            source=self.knowledge_path.name,
            data_version="knowledge_v1",
        )
