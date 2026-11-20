from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contract_validator import ContractValidationResult, ContractValidator
from models import (
    EnergySummaryData,
    ErrorData,
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
    SearchKnowledgeRequest,
    SimulateScenarioRequest,
    TariffData,
    ToolEnvelope,
    ExplainScheduleRequest,
    ValidateUploadRequest,
)
from numeric_check import NumericAudit, check_answer_numbers
from rule_baseline import RuleDecision, predict
from scope import UserScopeResolver
from tool_service import ToolService


@dataclass(frozen=True)
class PipelineResult:
    utterance: str
    decision: RuleDecision
    user_scope: str
    contract: ContractValidationResult
    planned_tools: list[str]
    tool_results: list[ToolEnvelope[Any]]
    answer: str
    outcome: str
    numeric_audit: NumericAudit


class DeterministicAssistant:
    def __init__(
        self,
        *,
        tool_service: ToolService | None = None,
        validator: ContractValidator | None = None,
        scope_resolver: UserScopeResolver | None = None,
    ) -> None:
        self.tools = tool_service or ToolService()
        self.validator = validator or ContractValidator()
        self.scope_resolver = scope_resolver or UserScopeResolver.from_config()

    def process(self, utterance: str) -> PipelineResult:
        decision = predict(utterance)
        user_scope, scope_reason = self._infer_user_scope(
            utterance, decision
        )
        candidate_tools = self._plan_tools(decision.intent_id)
        contract = self.validator.validate(
            intent_id=decision.intent_id,
            user_scope=user_scope,
            slots=decision.slots,
            requested_tools=candidate_tools,
        )

        unauthorized_known_user = (
            scope_reason == "unknown_customer_no_upload"
            and decision.intent_id
            not in {"new_user_guidance", "upload_scope_request"}
        )
        if unauthorized_known_user:
            answer = self._refusal_answer(decision.intent_id)
            audit = check_answer_numbers(answer, [])
            return self._result(
                utterance,
                decision,
                user_scope,
                contract,
                [],
                [],
                answer,
                "refuse",
                audit,
            )

        if contract.missing_slots:
            answer = self._clarification_answer(contract)
            audit = check_answer_numbers(answer, [])
            return self._result(
                utterance,
                decision,
                user_scope,
                contract,
                [],
                [],
                answer,
                "clarify",
                audit,
            )
        if contract.errors:
            answer = self._refusal_answer(decision.intent_id)
            audit = check_answer_numbers(answer, [])
            return self._result(
                utterance,
                decision,
                user_scope,
                contract,
                [],
                [],
                answer,
                "refuse",
                audit,
            )

        if decision.intent_id == "new_user_guidance":
            answer = self._refusal_answer(decision.intent_id)
            audit = check_answer_numbers(answer, [])
            return self._result(
                utterance,
                decision,
                user_scope,
                contract,
                [],
                [],
                answer,
                "answer",
                audit,
            )

        if decision.intent_id in {
            "device_control_request",
            "unknown_or_out_of_scope",
        }:
            answer = self._refusal_answer(decision.intent_id)
            audit = check_answer_numbers(answer, [])
            return self._result(
                utterance,
                decision,
                user_scope,
                contract,
                [],
                [],
                answer,
                "refuse",
                audit,
            )

        tool_results = self._execute_tools(decision)
        if any(result.status.value != "ok" for result in tool_results):
            answer = self._tool_error_answer(tool_results)
            audit = check_answer_numbers(
                answer,
                [result.data for result in tool_results],
            )
            return self._result(
                utterance,
                decision,
                user_scope,
                contract,
                candidate_tools,
                tool_results,
                answer,
                "tool_error",
                audit,
            )

        answer = self._success_answer(decision.intent_id, tool_results)
        audit = check_answer_numbers(
            answer,
            [result.data for result in tool_results],
        )
        outcome = "answer" if audit.valid else "tool_error"
        return self._result(
            utterance,
            decision,
            user_scope,
            contract,
            candidate_tools,
            tool_results,
            answer,
            outcome,
            audit,
        )

    @staticmethod
    def _result(
        utterance: str,
        decision: RuleDecision,
        user_scope: str,
        contract: ContractValidationResult,
        planned_tools: list[str],
        tool_results: list[ToolEnvelope[Any]],
        answer: str,
        outcome: str,
        numeric_audit: NumericAudit,
    ) -> PipelineResult:
        return PipelineResult(
            utterance=utterance,
            decision=decision,
            user_scope=user_scope,
            contract=contract,
            planned_tools=planned_tools,
            tool_results=tool_results,
            answer=answer,
            outcome=outcome,
            numeric_audit=numeric_audit,
        )

    def _infer_user_scope(
        self,
        utterance: str,
        decision: RuleDecision,
    ) -> tuple[str, str]:
        if decision.intent_id == "new_user_guidance":
            return "unsupported", "new_user_guidance"

        customer_id = decision.slots.get("customer_id")
        dataset_id = decision.slots.get("dataset_id")
        if customer_id is not None or dataset_id is not None:
            result = self.scope_resolver.resolve(
                customer_id=customer_id,
                dataset_id=dataset_id,
                dataset_validated="校验通过" in utterance,
            )
            return result.user_scope, result.reason_code

        if decision.intent_id in {
            "tariff_query",
            "energy_knowledge",
            "device_control_request",
            "unknown_or_out_of_scope",
        }:
            return "existing_300", "public_query"
        return "unsupported", "missing_user_identity"

    @staticmethod
    def _plan_tools(intent_id: str) -> list[str]:
        return {
            "status_query": ["get_energy_summary"],
            "load_forecast_query": ["get_forecast"],
            "pv_forecast_query": ["get_forecast"],
            "tariff_query": ["get_tariff"],
            "schedule_query": ["get_schedule", "get_energy_summary"],
            "schedule_explanation": [
                "explain_schedule",
                "get_schedule",
                "get_tariff",
            ],
            "cost_savings_query": ["get_energy_summary"],
            "energy_saving_advice": [
                "get_saving_advice",
                "get_energy_summary",
            ],
            "what_if_scenario": ["simulate_scenario"],
            "energy_knowledge": ["search_knowledge"],
            "upload_scope_request": ["validate_upload"],
        }.get(intent_id, [])

    def _execute_tools(
        self,
        decision: RuleDecision,
    ) -> list[ToolEnvelope[Any]]:
        slots = decision.slots
        intent = decision.intent_id
        results: list[ToolEnvelope[Any]] = []
        if intent in {"status_query", "cost_savings_query"}:
            results.append(
                self.tools.get_energy_summary(
                    GetEnergySummaryRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                    )
                )
            )
        elif intent in {"load_forecast_query", "pv_forecast_query"}:
            results.append(
                self.tools.get_forecast(
                    GetForecastRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                        target=slots.get("target", "load"),
                        horizon=slots.get("horizon", "48_step"),
                    )
                )
            )
        elif intent == "tariff_query":
            results.append(
                self.tools.get_tariff(
                    GetTariffRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                    )
                )
            )
        elif intent == "schedule_query":
            results.append(
                self.tools.get_schedule(
                    GetScheduleRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                        strategy="final",
                    )
                )
            )
            results.append(
                self.tools.get_energy_summary(
                    GetEnergySummaryRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                    )
                )
            )
        elif intent == "schedule_explanation":
            results.append(
                self.tools.explain_schedule(
                    ExplainScheduleRequest(
                        customer_id=slots.get("customer_id"),
                        interval_start=slots["interval_start"],
                    )
                )
            )
        elif intent == "energy_saving_advice":
            results.append(
                self.tools.get_energy_summary(
                    GetEnergySummaryRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                    )
                )
            )
            results.append(
                self.tools.get_saving_advice(
                    GetSavingAdviceRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                    )
                )
            )
        elif intent == "what_if_scenario":
            results.append(
                self.tools.simulate_scenario(
                    SimulateScenarioRequest(
                        customer_id=slots.get("customer_id"),
                        date=slots["date"],
                        scenario=slots["scenario"],
                    )
                )
            )
        elif intent == "energy_knowledge":
            results.append(
                self.tools.search_knowledge(
                    SearchKnowledgeRequest(
                        query=decision.utterance,
                        top_k=3,
                    )
                )
            )
        elif intent == "upload_scope_request":
            dataset_id = slots.get("dataset_id")
            if dataset_id:
                results.append(
                    self.tools.validate_upload(
                        ValidateUploadRequest(dataset_id=dataset_id)
                    )
                )
        return results

    @staticmethod
    def _clarification_answer(
        contract: ContractValidationResult,
    ) -> str:
        return "还需要补充参数：" + "、".join(contract.missing_slots) + "。"

    @staticmethod
    def _refusal_answer(intent_id: str) -> str:
        if intent_id == "device_control_request":
            return "当前助手不能直接控制真实设备，只能提供分析和建议。"
        if intent_id == "new_user_guidance":
            return "当前正式支持已有用户。新用户需要先上传连续负荷、光伏容量、邮编和电价方案。"
        if intent_id == "unknown_or_out_of_scope":
            return "当前问题超出助手范围，请询问家庭用电、光伏、电价或储能调度。"
        return "当前请求超出已授权数据或工具范围。"

    @staticmethod
    def _tool_error_answer(
        results: list[ToolEnvelope[Any]],
    ) -> str:
        for result in results:
            if isinstance(result.data, ErrorData):
                return f"工具未返回可用数据：{result.data.message}"
        return "工具未返回可用数据。"

    @staticmethod
    def _success_answer(
        intent_id: str,
        results: list[ToolEnvelope[Any]],
    ) -> str:
        data_by_tool = {result.tool: result.data for result in results}
        if intent_id in {"status_query", "cost_savings_query"}:
            data = data_by_tool["get_energy_summary"]
            assert isinstance(data, EnergySummaryData)
            return (
                f"该日总用电 {data.load_kwh:.3f} kWh，光伏 {data.pv_kwh:.3f} kWh，"
                f"购电 {data.grid_import_kwh:.3f} kWh，售电 {data.grid_export_kwh:.3f} kWh，"
                f"净电费 {data.net_cost:.3f} AUD，退化成本 {data.degradation_cost:.3f} AUD，"
                f"总成本 {data.total_cost:.3f} AUD，日末 SOC 为 {data.final_soc:.6f}。"
            )
        if intent_id in {"load_forecast_query", "pv_forecast_query"}:
            data = data_by_tool["get_forecast"]
            assert isinstance(data, ForecastData)
            first = data.rows[0]
            return f"第一个预测时段的预测值为 {first.yhat:.6f} kWh。"
        if intent_id == "tariff_query":
            data = data_by_tool["get_tariff"]
            assert isinstance(data, TariffData)
            buy_prices = [row.buy_price_per_kwh for row in data.rows]
            sell_price = data.rows[0].sell_price_per_kwh
            return (
                f"该日最高购电价为 {max(buy_prices):.3f} AUD/kWh，"
                f"最低购电价为 {min(buy_prices):.3f} AUD/kWh，"
                f"上网电价为 {sell_price:.3f} AUD/kWh。"
            )
        if intent_id == "schedule_query":
            schedule = data_by_tool["get_schedule"]
            summary = data_by_tool["get_energy_summary"]
            assert isinstance(schedule, ScheduleData)
            assert isinstance(summary, EnergySummaryData)
            return (
                f"最终策略为 {schedule.rows[0].strategy}；"
                f"购电 {summary.grid_import_kwh:.3f} kWh，售电 {summary.grid_export_kwh:.3f} kWh，"
                f"总成本 {summary.total_cost:.3f} AUD，日末 SOC 为 {summary.final_soc:.6f}。"
            )
        if intent_id == "schedule_explanation":
            data = data_by_tool["explain_schedule"]
            assert isinstance(data, ExplanationData)
            return (
                "；".join(data.reasons)
                + f"。当时负荷 {data.evidence_values.load_kw:.3f} kW，"
                f"光伏 {data.evidence_values.pv_kw:.3f} kW，"
                f"购电价 {data.evidence_values.buy_price_per_kwh:.3f} AUD/kWh。"
            )
        if intent_id == "energy_saving_advice":
            data = data_by_tool["get_saving_advice"]
            assert isinstance(data, SavingAdviceData)
            return "；".join(item.title for item in data.advice_items) + "。"
        if intent_id == "energy_knowledge":
            data = data_by_tool["search_knowledge"]
            assert isinstance(data, KnowledgeData)
            return " ".join(document.content for document in data.documents)
        return "请求已完成。"


def evaluate_pipeline(
    *,
    seed_path: str | Path | None = None,
    assistant: DeterministicAssistant | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(__file__).resolve().parent
    path = Path(seed_path) if seed_path is not None else root / "data" / "eval_seed.csv"
    assistant = assistant or DeterministicAssistant()
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, Any]] = []
    intent_correct = 0
    slot_exact = 0
    outcome_correct = 0
    numeric_valid = 0
    tool_match = 0
    for row in rows:
        result = assistant.process(row["utterance"])
        expected_slots = json.loads(row["expected_slots_json"])
        expected_tools = [
            item for item in row["required_tools"].split(";") if item
        ]
        intent_ok = result.decision.intent_id == row["expected_intent"]
        slot_ok = result.decision.slots == expected_slots
        outcome_ok = result.outcome == row["expected_outcome"]
        tools_ok = result.planned_tools == expected_tools
        intent_correct += int(intent_ok)
        slot_exact += int(slot_ok)
        outcome_correct += int(outcome_ok)
        numeric_valid += int(result.numeric_audit.valid)
        tool_match += int(tools_ok)
        output_rows.append(
            {
                "case_id": row["case_id"],
                "utterance": row["utterance"],
                "expected_intent": row["expected_intent"],
                "predicted_intent": result.decision.intent_id,
                "expected_slots_json": json.dumps(
                    expected_slots,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "predicted_slots_json": json.dumps(
                    result.decision.slots,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "expected_outcome": row["expected_outcome"],
                "predicted_outcome": result.outcome,
                "answer": result.answer,
                "numeric_audit_valid": result.numeric_audit.valid,
                "unmatched_numbers": ";".join(
                    result.numeric_audit.unmatched_numbers
                ),
                "contract_valid": result.contract.valid,
                "missing_slots": ";".join(result.contract.missing_slots),
                "contract_errors": ";".join(result.contract.errors),
                "expected_tools": ";".join(expected_tools),
                "planned_tools": ";".join(result.planned_tools),
                "tool_match": tools_ok,
                "intent_correct": intent_ok,
                "slot_exact_match": slot_ok,
                "outcome_correct": outcome_ok,
            }
        )
    count = len(rows)
    metrics = {
        "cases": count,
        "intent_accuracy": intent_correct / count,
        "slot_exact_match_rate": slot_exact / count,
        "outcome_accuracy": outcome_correct / count,
        "planned_tool_match_rate": tool_match / count,
        "numeric_consistency_rate": numeric_valid / count,
    }
    return output_rows, metrics


def main() -> None:
    root = Path(__file__).resolve().parent
    rows, metrics = evaluate_pipeline()
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)
    with (output_dir / "pipeline_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "pipeline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
