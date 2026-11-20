from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from contract_validator import ContractValidationResult, ContractValidator


class LLMConfigurationError(RuntimeError):
    pass


class LLMOutputError(RuntimeError):
    pass


class IntentPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: str
    slots: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0, le=1)


@dataclass(frozen=True)
class LLMIntentResult:
    utterance: str
    user_scope: str
    prediction: IntentPrediction
    contract: ContractValidationResult


class LLMProvider(Protocol):
    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


class DeepSeekProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("DeepSeek API key 为空")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls, *, dotenv_path: str | Path | None = None) -> "DeepSeekProvider":
        path = (
            Path(dotenv_path)
            if dotenv_path is not None
            else Path(__file__).with_name(".env")
        )
        _load_dotenv(path)
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "未配置 DEEPSEEK_API_KEY。请设置在环境变量或 AI助手/.env 中。"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 700,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMOutputError("DeepSeek 响应缺少 choices[0].message.content") from exc


class FixedResponseProvider:
    """Test provider that returns a fixed or per-utterance JSON response."""

    def __init__(self, response: str | dict[str, str]) -> None:
        self.response = response

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        if isinstance(self.response, dict):
            for utterance, response in self.response.items():
                if utterance in user_prompt:
                    return response
            raise LLMOutputError("测试 provider 没有匹配响应")
        return self.response


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


class LLMIntentClassifier:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        validator: ContractValidator | None = None,
        few_shot_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.validator = validator or ContractValidator()
        examples_path = (
            Path(few_shot_path)
            if few_shot_path is not None
            else Path(__file__).with_name("data") / "few_shot_examples.json"
        )
        self.examples = json.loads(
            examples_path.read_text(encoding="utf-8")
        )
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        intents = self.validator.intents_contract["intents"]
        slots = self.validator.slot_definitions
        intent_text = "\n".join(
            f"- {item['id']}: {item['description']}"
            for item in intents
        )
        slot_text = json.dumps(slots, ensure_ascii=False, indent=2)
        example_text = json.dumps(
            self.examples,
            ensure_ascii=False,
            indent=2,
        )
        return (
            "你是家庭能源助手的意图和槽位分类器。只输出 JSON，不要输出解释或 Markdown。\n"
            "输出必须严格符合："
            '{"intent_id": string, "slots": object, "confidence": number}。\n'
            "不要输出契约之外的 intent 或 slot。不要猜测缺失日期、客户编号或数据集编号。"
            "相对日期如“明天”在没有会话日期解析结果时不要填入 date。\n"
            "设备控制请求必须分类为 device_control_request。\n\n"
            f"可用意图：\n{intent_text}\n\n"
            f"槽位定义：\n{slot_text}\n\n"
            f"少样本示例：\n{example_text}\n"
        )

    def classify(self, *, utterance: str, user_scope: str) -> LLMIntentResult:
        user_prompt = (
            f"当前会话用户范围：{user_scope}\n"
            f"用户原话：{utterance}\n"
            "请输出 JSON。"
        )
        raw = self.provider.generate_json(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )
        try:
            payload = json.loads(_strip_code_fence(raw))
            prediction = IntentPrediction.model_validate(payload)
        except Exception as exc:
            raise LLMOutputError(f"LLM 输出不是合法契约 JSON：{exc}") from exc

        contract = self.validator.validate(
            intent_id=prediction.intent_id,
            user_scope=user_scope,
            slots=prediction.slots,
        )
        return LLMIntentResult(
            utterance=utterance,
            user_scope=user_scope,
            prediction=prediction,
            contract=contract,
        )


def evaluate_seed_cases(
    classifier: LLMIntentClassifier,
    *,
    seed_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (
        Path(seed_path)
        if seed_path is not None
        else Path(__file__).with_name("data") / "eval_seed.csv"
    )
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    predictions: list[dict[str, Any]] = []
    intent_correct = 0
    slot_exact_correct = 0
    contract_valid_count = 0
    slot_tp = 0
    slot_predicted = 0
    slot_expected = 0

    for row in rows:
        expected_slots = json.loads(row["expected_slots_json"])
        result = classifier.classify(
            utterance=row["utterance"],
            user_scope=row["user_scope"],
        )
        intent_ok = result.prediction.intent_id == row["expected_intent"]
        slot_ok = result.prediction.slots == expected_slots
        intent_correct += int(intent_ok)
        slot_exact_correct += int(slot_ok)
        contract_valid_count += int(result.contract.valid)

        expected_pairs = {
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
            for key, value in expected_slots.items()
        }
        predicted_pairs = {
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
            for key, value in result.prediction.slots.items()
        }
        slot_tp += len(expected_pairs & predicted_pairs)
        slot_predicted += len(predicted_pairs)
        slot_expected += len(expected_pairs)
        predictions.append(
            {
                "case_id": row["case_id"],
                "utterance": row["utterance"],
                "expected_intent": row["expected_intent"],
                "predicted_intent": result.prediction.intent_id,
                "expected_slots_json": json.dumps(
                    expected_slots,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "predicted_slots_json": json.dumps(
                    result.prediction.slots,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "confidence": result.prediction.confidence,
                "intent_correct": intent_ok,
                "slot_exact_match": slot_ok,
                "contract_valid": result.contract.valid,
                "contract_errors": ";".join(result.contract.errors),
                "missing_slots": ";".join(result.contract.missing_slots),
            }
        )

    precision = slot_tp / slot_predicted if slot_predicted else 0.0
    recall = slot_tp / slot_expected if slot_expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    metrics = {
        "cases": len(rows),
        "intent_accuracy": intent_correct / len(rows),
        "slot_exact_match_rate": slot_exact_correct / len(rows),
        "slot_micro_precision": precision,
        "slot_micro_recall": recall,
        "slot_micro_f1": f1,
        "contract_valid_rate": contract_valid_count / len(rows),
    }
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(exist_ok=True)
    try:
        provider = DeepSeekProvider.from_env()
    except LLMConfigurationError as exc:
        status = {
            "status": "blocked",
            "reason": "missing_api_key",
            "message": str(exc),
            "required_env": [
                "DEEPSEEK_API_KEY",
                "DEEPSEEK_BASE_URL",
                "DEEPSEEK_MODEL",
            ],
        }
        (output_dir / "llm_intent_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(str(exc))
        print(f"状态文件：{output_dir / 'llm_intent_status.json'}")
        return
    classifier = LLMIntentClassifier(provider)
    predictions, metrics = evaluate_seed_cases(classifier)
    if args.limit > 0:
        predictions = predictions[: args.limit]
        metrics["cases"] = len(predictions)

    with (output_dir / "llm_intent_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    (output_dir / "llm_intent_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
