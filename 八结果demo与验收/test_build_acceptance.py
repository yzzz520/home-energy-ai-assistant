from __future__ import annotations

import csv
import json
from pathlib import Path

from build_acceptance import build_acceptance


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_acceptance_generates_three_tables(tmp_path: Path) -> None:
    summary = build_acceptance(output_dir=tmp_path)

    assert summary["overall_status"] == "ACCEPTED_WITH_NEGATIVE_SCHEDULING_RESULT"
    scheduling = read_csv(tmp_path / "three_strategy_comparison.csv")
    forecast = read_csv(tmp_path / "forecast_comparison.csv")
    assistant = read_csv(tmp_path / "assistant_evaluation.csv")
    assert len(scheduling) == 3
    assert len(forecast) == 3
    assert len(assistant) == 5

    assert scheduling[1]["策略"] == "预测驱动 MILP"
    assert float(scheduling[1]["相对规则调度_AUD"]) > 0
    assert forecast[0]["模型"] == "LightGBM（300 户全局）"
    assert float(forecast[0]["MAE_kWh"]) < float(forecast[1]["MAE_kWh"])
    assert all(row["通过率_pct"] == "100.0" for row in assistant)


def test_generated_report_is_offline_and_contains_three_sections(
    tmp_path: Path,
) -> None:
    summary = build_acceptance(output_dir=tmp_path)
    report = Path(summary["outputs"]["acceptance_report.html"]).read_text(
        encoding="utf-8"
    )

    assert "三基准对比表" in report
    assert "预测对比表" in report
    assert "助手评测表" in report
    assert "cdn.jsdelivr.net" not in report
    assert "https://" not in report
    assert "预测驱动 MILP" in report
    assert "不代表泛化准确率" in report


def test_summary_preserves_negative_scheduling_result(tmp_path: Path) -> None:
    summary = build_acceptance(output_dir=tmp_path)
    payload = json.loads(
        Path(summary["summary_path"]).read_text(encoding="utf-8")
    )

    scheduling = payload["scheduling"]
    assert (
        scheduling["forecast_milp_cost_aud"]
        > scheduling["rule_cost_aud"]
    )
    assert scheduling["forecast_capture_pct"] < 0
    assert payload["assistant"]["cases"] == 22
