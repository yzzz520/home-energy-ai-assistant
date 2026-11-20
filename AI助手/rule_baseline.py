from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuleDecision:
    utterance: str
    intent_id: str
    slots: dict[str, Any]


def extract_slots(utterance: str) -> dict[str, Any]:
    slots: dict[str, Any] = {}

    customer_match = re.search(r"客户\s*(\d+)", utterance)
    if customer_match:
        slots["customer_id"] = int(customer_match.group(1))

    dataset_match = re.search(
        r"(?:数据集|dataset)\s*([A-Za-z0-9_-]+)",
        utterance,
        flags=re.IGNORECASE,
    )
    if dataset_match:
        slots["dataset_id"] = dataset_match.group(1)

    date_match = re.search(
        r"(?<!\d)(20\d{2})-(\d{1,2})-(\d{1,2})(?!\d)",
        utterance,
    )
    chinese_date_match = re.search(
        r"(20\d{2})年(\d{1,2})月(\d{1,2})日",
        utterance,
    )
    normalized_date: str | None = None
    if date_match or chinese_date_match:
        selected_match = date_match or chinese_date_match
        candidate = date(
            int(selected_match.group(1)),
            int(selected_match.group(2)),
            int(selected_match.group(3)),
        )
        normalized_date = candidate.isoformat()
        slots["date"] = normalized_date

    time_match = re.search(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", utterance)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        if normalized_date:
            slots["interval_start"] = (
                f"{normalized_date}T{hour:02d}:{minute:02d}:00"
            )
        else:
            slots["interval_start"] = f"T{hour:02d}:{minute:02d}:00"

    if "负荷" in utterance:
        slots["target"] = "load"
    elif "光伏" in utterance and "没有光伏" not in utterance:
        slots["target"] = "pv"

    if "如果" in utterance and "没有光伏" in utterance:
        slots["scenario"] = {"pv": 0}
    elif "如果" in utterance and "电价翻倍" in utterance:
        slots["scenario"] = {"buy_price_multiplier": 2}

    return slots


def predict_intent(utterance: str, slots: dict[str, Any]) -> str:
    text = utterance.strip()

    if any(keyword in text for keyword in ("关闭", "直接控制", "控制设备")):
        return "device_control_request"

    if any(keyword in text for keyword in ("天气", "股票", "笑话")):
        return "unknown_or_out_of_scope"

    if "为什么" in text and any(
        keyword in text for keyword in ("放电", "充电")
    ):
        return "schedule_explanation"

    if (
        "安全" in text
        or ("SOC" in text.upper() and any(word in text for word in ("低于", "低于")))
    ):
        return "energy_knowledge"

    if (
        "不是这300户" in text
        or ("冷启动" in text and "能做什么" in text)
        or ("数据" in text and "要求" in text)
    ):
        return "new_user_guidance"

    if "如果" in text and "scenario" in slots:
        return "what_if_scenario"

    if "节能建议" in text:
        return "energy_saving_advice"

    if any(
        keyword in text
        for keyword in ("省多少钱", "多花多少钱", "节省多少钱")
    ):
        return "cost_savings_query"

    if "预测" in text and "负荷" in text:
        return "load_forecast_query"

    if (
        "光伏" in text
        and ("预测" in text or "最高" in text)
    ):
        return "pv_forecast_query"

    if "计划" in text:
        return "schedule_query"

    if any(keyword in text for keyword in ("电价", "峰段", "谷段")):
        return "tariff_query"

    if any(
        keyword in text
        for keyword in ("总成本", "用了多少电", "当前状态", "SOC")
    ):
        return "status_query"

    if "上传" in text or "数据集" in text:
        return "upload_scope_request"

    return "unknown_or_out_of_scope"


def predict(utterance: str) -> RuleDecision:
    slots = extract_slots(utterance)
    intent_id = predict_intent(utterance, slots)
    return RuleDecision(
        utterance=utterance,
        intent_id=intent_id,
        slots=slots,
    )


def evaluate_seed_cases(
    seed_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with Path(seed_path).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    predictions: list[dict[str, Any]] = []
    intent_correct = 0
    slot_exact_correct = 0
    slot_tp = 0
    slot_predicted = 0
    slot_expected = 0
    per_intent: dict[str, dict[str, int]] = {}

    for row in rows:
        expected_slots = json.loads(row["expected_slots_json"])
        decision = predict(row["utterance"])
        intent_ok = decision.intent_id == row["expected_intent"]
        slot_ok = decision.slots == expected_slots
        intent_correct += int(intent_ok)
        slot_exact_correct += int(slot_ok)

        expected_items = {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            for key, value in expected_slots.items()
        }
        predicted_items = {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            for key, value in decision.slots.items()
        }
        expected_pairs = set(expected_items.items())
        predicted_pairs = set(predicted_items.items())
        slot_tp += len(expected_pairs & predicted_pairs)
        slot_predicted += len(predicted_pairs)
        slot_expected += len(expected_pairs)

        bucket = per_intent.setdefault(
            row["expected_intent"],
            {"cases": 0, "intent_correct": 0, "slot_exact": 0},
        )
        bucket["cases"] += 1
        bucket["intent_correct"] += int(intent_ok)
        bucket["slot_exact"] += int(slot_ok)
        predictions.append(
            {
                "case_id": row["case_id"],
                "utterance": row["utterance"],
                "expected_intent": row["expected_intent"],
                "predicted_intent": decision.intent_id,
                "intent_correct": intent_ok,
                "expected_slots_json": json.dumps(
                    expected_slots,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "predicted_slots_json": json.dumps(
                    decision.slots,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "slot_exact_match": slot_ok,
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
        "per_intent": {
            intent: {
                **values,
                "intent_accuracy": values["intent_correct"] / values["cases"],
            }
            for intent, values in sorted(per_intent.items())
        },
    }
    return predictions, metrics


def main() -> None:
    root = Path(__file__).resolve().parent
    predictions, metrics = evaluate_seed_cases(root / "data" / "eval_seed.csv")
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)
    prediction_path = output_dir / "rule_baseline_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    metrics_path = output_dir / "rule_baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[OK] cases={metrics['cases']} "
        f"intent_accuracy={metrics['intent_accuracy']:.3f} "
        f"slot_exact={metrics['slot_exact_match_rate']:.3f} "
        f"slot_f1={metrics['slot_micro_f1']:.3f}"
    )
    print(f"[OK] 预测：{prediction_path}")
    print(f"[OK] 指标：{metrics_path}")


if __name__ == "__main__":
    main()
