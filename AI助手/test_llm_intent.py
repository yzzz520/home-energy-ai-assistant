from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_intent import (
    FixedResponseProvider,
    IntentPrediction,
    LLMIntentClassifier,
    LLMOutputError,
    evaluate_seed_cases,
)


def test_prediction_model_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        IntentPrediction.model_validate(
            {
                "intent_id": "status_query",
                "slots": {"customer_id": 1},
                "confidence": 0.9,
                "extra": "bad",
            }
        )


def test_classifier_parses_json_and_validates_contract() -> None:
    provider = FixedResponseProvider(
        json.dumps(
            {
                "intent_id": "schedule_query",
                "slots": {"customer_id": 1, "date": "2011-06-15"},
                "confidence": 0.95,
            }
        )
    )
    classifier = LLMIntentClassifier(provider)
    result = classifier.classify(
        utterance="客户1在2011-06-15的充电计划",
        user_scope="existing_300",
    )
    assert result.prediction.intent_id == "schedule_query"
    assert result.contract.valid


def test_classifier_accepts_json_code_fence() -> None:
    provider = FixedResponseProvider(
        "```json\n"
        '{"intent_id":"tariff_query","slots":{"date":"2011-06-15"},"confidence":0.8}'
        "\n```"
    )
    classifier = LLMIntentClassifier(provider)
    result = classifier.classify(
        utterance="2011-06-15电价",
        user_scope="existing_300",
    )
    assert result.contract.valid


def test_invalid_json_raises() -> None:
    classifier = LLMIntentClassifier(FixedResponseProvider("not json"))
    with pytest.raises(LLMOutputError):
        classifier.classify(
            utterance="客户1用电多少",
            user_scope="existing_300",
        )


def test_mock_evaluation_pipeline_runs() -> None:
    root = Path(__file__).resolve().parent
    with (root / "data" / "eval_seed.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        rows = list(__import__("csv").DictReader(handle))
    responses = {
        row["utterance"]: json.dumps(
            {
                "intent_id": row["expected_intent"],
                "slots": json.loads(row["expected_slots_json"]),
                "confidence": 1.0,
            },
            ensure_ascii=False,
        )
        for row in rows
    }
    classifier = LLMIntentClassifier(FixedResponseProvider(responses))
    predictions, metrics = evaluate_seed_cases(classifier)
    assert len(predictions) == 22
    assert metrics["intent_accuracy"] == 1.0
    assert metrics["slot_exact_match_rate"] == 1.0
