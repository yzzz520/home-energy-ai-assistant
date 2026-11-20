from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline import process_customer_year
from quality import summarize_clean_data
from reader import AusgridReader, FileSpec


def process_file_customers(
    reader: AusgridReader,
    spec: FileSpec,
    *,
    customer_ids: Iterable[int] | None = None,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """逐户处理一个年度文件，只保存客户级质量汇总。"""
    raw_year = reader.read_single_csv(spec.name)
    all_customers = sorted(raw_year[reader.customer_col].unique().tolist())
    selected_customers = (
        all_customers
        if customer_ids is None
        else sorted({int(value) for value in customer_ids})
    )

    invalid = sorted(set(selected_customers) - set(all_customers))
    if invalid:
        raise ValueError(f"{spec.name} 中不存在客户：{invalid}")

    rows: list[dict[str, object]] = []
    for index, customer_id in enumerate(selected_customers, start=1):
        clean = process_customer_year(
            reader,
            customer_id=customer_id,
            source_file=spec.name,
            raw_year=raw_year,
        )
        summary = summarize_clean_data(clean)
        summary.update(
            {
                "source_year": spec.source_year,
                "source_file": spec.name,
                "customer_id": customer_id,
                "has_controlled_load": bool(
                    clean["has_controlled_load"].iloc[0]
                ),
                "availability_pct": round(
                    (1 - summary["missing_energy_rows"] / summary["rows"]) * 100,
                    6,
                ),
            }
        )
        rows.append(summary)

        if progress_every > 0 and (
            index % progress_every == 0 or index == len(selected_customers)
        ):
            print(
                f"[进度] {spec.source_year}: "
                f"{index}/{len(selected_customers)} 户"
            )

    details = pd.DataFrame(rows).sort_values("customer_id").reset_index(drop=True)
    file_summary = {
        "source_year": spec.source_year,
        "source_file": spec.name,
        "customers": len(details),
        "rows": int(details["rows"].sum()),
        "complete_customers": int(details["missing_days"].eq(0).sum()),
        "customers_with_missing_days": int(details["missing_days"].gt(0).sum()),
        "missing_customer_days": int(details["missing_days"].sum()),
        "missing_energy_rows": int(details["missing_energy_rows"].sum()),
        "non_actual_rows": int(details["quality_non_actual_rows"].sum()),
        "customers_with_non_actual_data": int(
            details["quality_non_actual_rows"].gt(0).sum()
        ),
        "pv_over_capacity_rows": int(details["pv_over_capacity_rows"].sum()),
        "load_extreme_rows": int(details["load_extreme_rows"].sum()),
        "utc_unresolved_rows": int(details["utc_unresolved_rows"].sum()),
        "dst_transition_days_per_customer": 2,
        "model_usable_rows": int(details["model_usable_rows"].sum()),
        "mean_availability_pct": round(float(details["availability_pct"].mean()), 6),
    }
    return details, file_summary


def main() -> None:
    reader = AusgridReader()
    output_dir = Path(__file__).with_name("reports")
    output_dir.mkdir(exist_ok=True)

    all_details: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for spec in reader.files:
        details, summary = process_file_customers(reader, spec)
        all_details.append(details)
        summaries.append(summary)
        print(
            f"[OK] {spec.source_year}: {summary['customers']} 户，"
            f"{summary['missing_customer_days']} 个缺失客户日，"
            f"可用率 {summary['mean_availability_pct']:.4f}%"
        )

    customer_report = pd.concat(all_details, ignore_index=True)
    customer_report_path = output_dir / "annual_quality_by_customer.csv"
    customer_report.to_csv(
        customer_report_path,
        index=False,
        encoding="utf-8-sig",
    )

    totals = {
        "files": len(summaries),
        "customers_per_file": 300,
        "customer_years": int(sum(item["customers"] for item in summaries)),
        "rows": int(sum(item["rows"] for item in summaries)),
        "complete_customer_years": int(
            sum(item["complete_customers"] for item in summaries)
        ),
        "customer_years_with_missing_days": int(
            sum(item["customers_with_missing_days"] for item in summaries)
        ),
        "missing_customer_days": int(
            sum(item["missing_customer_days"] for item in summaries)
        ),
        "missing_energy_rows": int(
            sum(item["missing_energy_rows"] for item in summaries)
        ),
        "non_actual_rows": int(sum(item["non_actual_rows"] for item in summaries)),
        "pv_over_capacity_rows": int(
            sum(item["pv_over_capacity_rows"] for item in summaries)
        ),
        "load_extreme_rows": int(
            sum(item["load_extreme_rows"] for item in summaries)
        ),
        "model_usable_rows": int(
            sum(item["model_usable_rows"] for item in summaries)
        ),
    }
    summary_report = {"totals": totals, "files": summaries}
    summary_path = output_dir / "annual_quality_summary.json"
    summary_path.write_text(
        json.dumps(summary_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] 年度质量汇总：{summary_path}")
    print(f"[OK] 客户级质量明细：{customer_report_path}")


if __name__ == "__main__":
    main()
