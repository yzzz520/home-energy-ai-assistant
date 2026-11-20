from __future__ import annotations

import argparse
import csv
import html
import json
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")

SOURCE_PATHS = {
    "scheduling": (
        PROJECT_ROOT
        / "储能调度"
        / "results"
        / "forecast_vs_perfect_customer_1.json"
    ),
    "forecast": (
        PROJECT_ROOT
        / "时序预测"
        / "results"
        / "validation_300_households_lightgbm_metrics.json"
    ),
    "forecast_per_customer": (
        PROJECT_ROOT
        / "时序预测"
        / "results"
        / "validation_300_households_per_customer.csv"
    ),
    "assistant": PROJECT_ROOT / "AI助手" / "results" / "pipeline_metrics.json",
    "schedule_schema": (
        PROJECT_ROOT / "储能调度" / "results" / "schedule_schema.json"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"缺少验收数据源：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少验收数据源：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def build_scheduling_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    rule_cost = float(data["rule"]["total_cost"])
    forecast_cost = float(data["forecast_milp"]["total_cost"])
    perfect_cost = float(data["perfect_milp"]["total_cost"])
    return [
        {
            "策略": "规则调度",
            "实际结算总成本_AUD": _round(rule_cost),
            "日均成本_AUD": _round(data["rule"]["mean_daily_cost"]),
            "相对规则调度_AUD": 0.0,
            "相对完美预知差距_AUD": _round(rule_cost - perfect_cost),
            "阶段结论": "安全基线",
        },
        {
            "策略": "预测驱动 MILP",
            "实际结算总成本_AUD": _round(forecast_cost),
            "日均成本_AUD": _round(data["forecast_milp"]["mean_daily_cost"]),
            "相对规则调度_AUD": _round(forecast_cost - rule_cost),
            "相对完美预知差距_AUD": _round(forecast_cost - perfect_cost),
            "阶段结论": "未优于规则，负结果保留",
        },
        {
            "策略": "完美预知 MILP",
            "实际结算总成本_AUD": _round(perfect_cost),
            "日均成本_AUD": _round(data["perfect_milp"]["mean_daily_cost"]),
            "相对规则调度_AUD": _round(perfect_cost - rule_cost),
            "相对完美预知差距_AUD": 0.0,
            "阶段结论": "理论上限",
        },
    ]


def build_forecast_table(
    metrics: dict[str, Any],
    per_customer: Iterable[dict[str, str]],
) -> list[dict[str, Any]]:
    overall = metrics["overall_metrics"]
    daily_mae = float(overall["seasonal_naive_daily"]["mae_kwh"])
    rows = [
        ("LightGBM（300 户全局）", "lightgbm", "优于两种朴素基线"),
        ("昨日同时刻朴素基线", "seasonal_naive_daily", "主基线"),
        ("上周同时刻朴素基线", "seasonal_naive_weekly", "对照基线"),
    ]
    beat_daily = sum(
        float(row["improvement_vs_daily_pct"]) > 0
        for row in per_customer
    )
    output: list[dict[str, Any]] = []
    for label, key, conclusion in rows:
        item = overall[key]
        improvement = (daily_mae - float(item["mae_kwh"])) / daily_mae * 100
        output.append(
            {
                "模型": label,
                "MAE_kWh": _round(item["mae_kwh"]),
                "RMSE_kWh": _round(item["rmse_kwh"]),
                "WAPE_pct": _round(item["wape_pct"], 2),
                "平均偏差_kWh": _round(item["mean_bias_kwh"]),
                "相对昨日基线MAE改善_pct": _round(improvement, 2),
                "阶段结论": (
                    f"{conclusion}；{beat_daily}/"
                    f"{metrics['customers']} 户优于昨日基线"
                    if key == "lightgbm"
                    else conclusion
                ),
            }
        )
    return output


def build_assistant_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = [
        ("意图准确率", "intent_accuracy", "意图识别"),
        ("槽位精确匹配率", "slot_exact_match_rate", "槽位抽取"),
        ("结果类型准确率", "outcome_accuracy", "结果分支"),
        ("工具计划匹配率", "planned_tool_match_rate", "工具选择"),
        ("数字一致率", "numeric_consistency_rate", "回答忠实度"),
    ]
    cases = int(metrics["cases"])
    rows: list[dict[str, Any]] = []
    for label, key, conclusion in definitions:
        value = float(metrics[key])
        passed = round(cases * value)
        rows.append(
            {
                "评测指标": label,
                "通过数": passed,
                "总数": cases,
                "通过率_pct": _round(value * 100, 2),
                "阶段结论": conclusion,
            }
        )
    return rows


def build_checks(
    scheduling: dict[str, Any],
    forecast: dict[str, Any],
    assistant: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    forecast_metrics = forecast["overall_metrics"]
    lightgbm_mae = float(forecast_metrics["lightgbm"]["mae_kwh"])
    daily_mae = float(forecast_metrics["seasonal_naive_daily"]["mae_kwh"])
    weekly_mae = float(forecast_metrics["seasonal_naive_weekly"]["mae_kwh"])
    assistant_values = [
        float(assistant[key])
        for key in (
            "intent_accuracy",
            "slot_exact_match_rate",
            "outcome_accuracy",
            "planned_tool_match_rate",
            "numeric_consistency_rate",
        )
    ]
    rule_cost = float(scheduling["rule"]["total_cost"])
    forecast_cost = float(scheduling["forecast_milp"]["total_cost"])
    return [
        {
            "检查项": "三基准对比表",
            "结果": "通过",
            "说明": "规则调度、预测驱动 MILP、完美预知 MILP 均已汇总",
        },
        {
            "检查项": "预测优于基线",
            "结果": "通过" if lightgbm_mae < min(daily_mae, weekly_mae) else "失败",
            "说明": (
                f"LightGBM MAE {lightgbm_mae:.4f} kWh，"
                f"昨日基线 {daily_mae:.4f} kWh，"
                f"上周基线 {weekly_mae:.4f} kWh"
            ),
        },
        {
            "检查项": "调度负结果保留",
            "结果": "通过" if forecast_cost > rule_cost else "失败",
            "说明": (
                f"预测驱动 MILP 比规则调度高 "
                f"{forecast_cost - rule_cost:.4f} AUD；未美化实验结果"
            ),
        },
        {
            "检查项": "助手确定性评测",
            "结果": (
                "通过"
                if assistant.get("cases", 0) > 0 and all(value == 1.0 for value in assistant_values)
                else "失败"
            ),
            "说明": (
                f"{assistant.get('cases', 0)} 条手写种子全部通过；"
                "仅证明闭环，不代表泛化准确率"
            ),
        },
        {
            "检查项": "调度 schema 冻结",
            "结果": "通过" if schema.get("schema_version") == "1.0" else "失败",
            "说明": f"schedule schema version={schema.get('schema_version')}",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"不能写入空表：{path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}"
    return html.escape(str(value))


def _render_table(
    title: str,
    rows: list[dict[str, Any]],
) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td class=\"{'number' if isinstance(row[header], (int, float)) else ''}\">"
            f"{_format_cell(row[header])}</td>"
            for header in headers
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>{head}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
      </div>
    </section>
    """


def render_html(
    *,
    scheduling_rows: list[dict[str, Any]],
    forecast_rows: list[dict[str, Any]],
    assistant_rows: list[dict[str, Any]],
    checks: list[dict[str, str]],
    scheduling: dict[str, Any],
    forecast: dict[str, Any],
    assistant: dict[str, Any],
    source_paths: dict[str, Path],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lgb = forecast["overall_metrics"]["lightgbm"]
    daily = forecast["overall_metrics"]["seasonal_naive_daily"]
    improvement = (daily["mae_kwh"] - lgb["mae_kwh"]) / daily["mae_kwh"] * 100
    rule_cost = scheduling["rule"]["total_cost"]
    forecast_cost = scheduling["forecast_milp"]["total_cost"]
    check_rows = "".join(
        f"<tr><td>{html.escape(item['检查项'])}</td>"
        f"<td><span class=\"badge {'pass' if item['结果'] == '通过' else 'fail'}\">"
        f"{html.escape(item['结果'])}</span></td>"
        f"<td>{html.escape(item['说明'])}</td></tr>"
        for item in checks
    )
    sources = "".join(
        f"<li><code>{html.escape(str(path))}</code></li>"
        for path in source_paths.values()
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>第八步 · 结果 Demo 与验收</title>
  <style>
    :root {{
      --ink: #1d2824;
      --muted: #617069;
      --paper: #f4f5f0;
      --panel: #ffffff;
      --line: #d9ded8;
      --green: #276749;
      --green-soft: #e3f0e8;
      --amber: #9a5b13;
      --amber-soft: #fff0d7;
      --red: #a33a32;
      --red-soft: #fde8e5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif;
      line-height: 1.55;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 42px 24px 64px; }}
    header {{ margin-bottom: 28px; }}
    .eyebrow {{
      color: var(--green);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 8px 0 10px; font-size: clamp(30px, 5vw, 48px); line-height: 1.1; }}
    .subtitle {{ margin: 0; max-width: 820px; color: var(--muted); font-size: 16px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 28px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px 20px;
    }}
    .metric-label {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ margin-top: 5px; font-size: 27px; font-weight: 800; }}
    .metric-note {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .verdict {{
      border-left: 4px solid var(--green);
      background: var(--green-soft);
      padding: 14px 18px;
      margin: 22px 0 28px;
    }}
    .verdict strong {{ display: block; margin-bottom: 3px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin: 18px 0;
    }}
    h2 {{ margin: 0 0 14px; font-size: 21px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 11px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ background: #edf0eb; color: #34413b; font-size: 13px; }}
    td.number {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.pass {{ color: var(--green); background: var(--green-soft); }}
    .badge.fail {{ color: var(--red); background: var(--red-soft); }}
    .notes {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .note {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: #fafbf8;
    }}
    .note strong {{ color: var(--amber); }}
    code {{
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    ul {{ margin: 8px 0 0; padding-left: 20px; }}
    footer {{ color: var(--muted); margin-top: 24px; font-size: 12px; }}
    @media (max-width: 760px) {{
      main {{ padding: 28px 14px 44px; }}
      .metrics, .notes {{ grid-template-columns: 1fr; }}
      section {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Step 08 · Acceptance Demo</div>
    <h1>家庭光储荷 AI 助手 · 结果验收</h1>
    <p class="subtitle">
      汇总最终调度三基准、300 户负荷预测和 22 条助手确定性评测。
      本页为离线静态报告，不依赖外部 CDN。
    </p>
  </header>

  <div class="metrics">
    <div class="metric">
      <div class="metric-label">预测 MAE 改善</div>
      <div class="metric-value">{improvement:.2f}%</div>
      <div class="metric-note">LightGBM 相对昨日同时刻基线</div>
    </div>
    <div class="metric">
      <div class="metric-label">预测驱动 MILP 成本差</div>
      <div class="metric-value">+{forecast_cost - rule_cost:.2f} AUD</div>
      <div class="metric-note">相对规则调度，负结果保留</div>
    </div>
    <div class="metric">
      <div class="metric-label">助手种子评测</div>
      <div class="metric-value">{assistant['cases']}/{assistant['cases']}</div>
      <div class="metric-note">闭环通过；不代表泛化准确率</div>
    </div>
  </div>

  <div class="verdict">
    <strong>验收结论：通过，但必须带边界陈述。</strong>
    预测结果优于朴素基线；助手闭环可复现；调度预测驱动 MILP
    未优于规则调度，该负面结果已原样保留，不通过修改口径美化。
  </div>

  <section>
    <h2>验收检查</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>检查项</th><th>结果</th><th>说明</th></tr></thead>
        <tbody>{check_rows}</tbody>
      </table>
    </div>
  </section>

  {_render_table("一、三基准对比表", scheduling_rows)}
  {_render_table("二、预测对比表", forecast_rows)}
  {_render_table("三、助手评测表", assistant_rows)}

  <section>
    <h2>结论边界</h2>
    <div class="notes">
      <div class="note">
        <strong>调度不是“赢了”</strong><br>
        预测驱动 MILP 比规则调度高 {forecast_cost - rule_cost:.2f} AUD。
        规则调度继续作为安全层，完美预知仅作为理论上限。
      </div>
      <div class="note">
        <strong>助手 100% 不是泛化率</strong><br>
        22 条为手写、规则与种子同源的闭环验收样本。
        简历中应同时写明样本数和评测方法。
      </div>
    </div>
  </section>

  <section>
    <h2>数据源</h2>
    <ul>{sources}</ul>
  </section>

  <footer>Generated at {generated_at} · 八结果demo与验收</footer>
</main>
</body>
</html>
"""


def build_acceptance(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = {
        key: (
            path
            if path.is_absolute()
            else project_root / path
        )
        for key, path in SOURCE_PATHS.items()
    }
    scheduling = _load_json(paths["scheduling"])
    forecast = _load_json(paths["forecast"])
    per_customer = _load_csv(paths["forecast_per_customer"])
    assistant = _load_json(paths["assistant"])
    schema = _load_json(paths["schedule_schema"])

    scheduling_rows = build_scheduling_table(scheduling)
    forecast_rows = build_forecast_table(forecast, per_customer)
    assistant_rows = build_assistant_table(assistant)
    checks = build_checks(scheduling, forecast, assistant, schema)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "three_strategy_comparison.csv": scheduling_rows,
        "forecast_comparison.csv": forecast_rows,
        "assistant_evaluation.csv": assistant_rows,
    }
    for filename, rows in csv_outputs.items():
        write_csv(output_dir / filename, rows)

    report_html = render_html(
        scheduling_rows=scheduling_rows,
        forecast_rows=forecast_rows,
        assistant_rows=assistant_rows,
        checks=checks,
        scheduling=scheduling,
        forecast=forecast,
        assistant=assistant,
        source_paths=paths,
    )
    report_path = output_dir / "acceptance_report.html"
    report_path.write_text(report_html, encoding="utf-8")

    overall_status = (
        "ACCEPTED_WITH_NEGATIVE_SCHEDULING_RESULT"
        if all(check["结果"] == "通过" for check in checks)
        else "FAILED"
    )
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "scheduling": {
            "days": scheduling["days"],
            "period_start": scheduling["period_start"],
            "period_end": scheduling["period_end"],
            "rule_cost_aud": scheduling["rule"]["total_cost"],
            "forecast_milp_cost_aud": scheduling["forecast_milp"]["total_cost"],
            "perfect_milp_cost_aud": scheduling["perfect_milp"]["total_cost"],
            "forecast_regret_aud": scheduling["forecast_regret"],
            "forecast_capture_pct": scheduling["forecast_capture_pct"],
        },
        "forecast": {
            "customers": forecast["customers"],
            "evaluation_points": forecast["evaluation_points"],
            "lightgbm_mae_kwh": forecast["overall_metrics"]["lightgbm"]["mae_kwh"],
            "daily_naive_mae_kwh": forecast["overall_metrics"][
                "seasonal_naive_daily"
            ]["mae_kwh"],
            "weekly_naive_mae_kwh": forecast["overall_metrics"][
                "seasonal_naive_weekly"
            ]["mae_kwh"],
        },
        "assistant": assistant,
        "checks": checks,
        "outputs": {
            **{filename: str(output_dir / filename) for filename in csv_outputs},
            "acceptance_report.html": str(report_path),
        },
    }
    summary_path = output_dir / "acceptance_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成第八步结果 demo 与验收报告。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="生成后使用默认浏览器打开 HTML 验收页。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_acceptance(output_dir=args.output_dir)
    report_path = Path(summary["outputs"]["acceptance_report.html"])
    print(f"[OK] overall_status={summary['overall_status']}")
    print(f"[OK] report={report_path}")
    print(f"[OK] summary={summary['summary_path']}")
    if args.open:
        webbrowser.open(report_path.resolve().as_uri())


if __name__ == "__main__":
    main()
