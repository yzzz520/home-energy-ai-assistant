from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from category_pivot import pivot_categories
from energy_balance import add_energy_balance, add_grid_flow
from quality import add_outlier_flags, add_time_metadata, summarize_clean_data
from reader import AusgridReader
from time_axis import wide_to_interval_long


def process_customer_year(
    reader: AusgridReader,
    *,
    customer_id: int,
    source_file: str,
    raw_year: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将单个客户一个年度文件转换为一行一个半小时时间点的清洗表。"""
    spec = reader.file_specs.get(source_file)
    if spec is None:
        raise KeyError(f"年度文件未在 config.toml 中声明：{source_file}")

    if raw_year is None:
        raw_year = reader.read_single_csv(source_file)
    customer_raw = raw_year.loc[
        raw_year[reader.customer_col] == customer_id
    ].copy()
    if customer_raw.empty:
        raise ValueError(
            f"{source_file} 中没有 Customer={customer_id} 的数据"
        )

    long = wide_to_interval_long(
        customer_raw,
        reader.interval_columns,
        customer_col=reader.customer_col,
        date_col=reader.date_col,
        category_col=reader.category_col,
    )
    category_wide = pivot_categories(
        long,
        customer_col=reader.customer_col,
        category_col=reader.category_col,
        capacity_col=reader.capacity_col,
        postcode_col=reader.postcode_col,
        allowed_categories=tuple(sorted(reader.allowed_categories)),
        required_categories=("GC", "GG"),
    )
    observed = add_energy_balance(category_wide)
    observed = add_grid_flow(observed)

    expected_dates = spec.expected_unique_dates
    file_start = raw_year[reader.date_col].min().normalize()
    file_end = raw_year[reader.date_col].max().normalize()
    complete_dates = pd.date_range(file_start, file_end, freq="D")
    if len(complete_dates) != expected_dates:
        raise ValueError(
            f"{source_file} 应有 {expected_dates} 天，实际日期范围包含 "
            f"{len(complete_dates)} 天"
        )

    expected_rows = expected_dates * len(reader.interval_columns)
    offsets = pd.to_timedelta(
        np.arange(len(reader.interval_columns)) * 30,
        unit="m",
    )
    grid = pd.DataFrame(
        {
            reader.customer_col: customer_id,
            "source_date": np.repeat(complete_dates.to_numpy(), len(offsets)),
            "interval_start": np.repeat(complete_dates.to_numpy(), len(offsets))
            + np.tile(offsets.to_numpy(), len(complete_dates)),
            "slot_index": np.tile(
                np.arange(len(reader.interval_columns), dtype="int8"),
                len(complete_dates),
            ),
            reader.capacity_col: float(
                customer_raw[reader.capacity_col].iloc[0]
            ),
            reader.postcode_col: int(customer_raw[reader.postcode_col].iloc[0]),
            "source_year": spec.source_year,
            "source_file": source_file,
        }
    )
    grid["interval_end"] = grid["interval_start"] + pd.Timedelta(minutes=30)

    observed_dates = set(observed["source_date"].dt.normalize())
    payload_columns = [
        "source_date",
        "interval_start",
        "gc_kwh",
        "cl_kwh",
        "gg_kwh",
        "gc_quality",
        "cl_quality",
        "gg_quality",
        "quality_non_actual",
        "has_controlled_load",
        "total_load_kwh",
        "net_grid_kwh",
        "import_kwh",
        "export_kwh",
    ]
    clean = grid.merge(
        observed[payload_columns],
        on=["source_date", "interval_start"],
        how="left",
        validate="one_to_one",
    )
    clean["is_missing_day"] = ~clean["source_date"].isin(observed_dates)
    for column in ("gc_quality", "cl_quality", "gg_quality"):
        clean[column] = clean[column].fillna("MISSING_DAY")
    clean["quality_non_actual"] = (
        clean[["gc_quality", "cl_quality", "gg_quality"]]
        .eq("NA")
        .any(axis=1)
    )
    clean["has_controlled_load"] = clean["has_controlled_load"].fillna(
        bool(observed["has_controlled_load"].iloc[0])
    )
    energy_columns = [
        "gc_kwh",
        "cl_kwh",
        "gg_kwh",
        "total_load_kwh",
        "net_grid_kwh",
        "import_kwh",
        "export_kwh",
    ]
    clean["is_missing_interval"] = clean[energy_columns].isna().any(axis=1)

    clean = add_time_metadata(
        clean,
        timezone=reader.timezone,
    )
    clean = add_outlier_flags(
        clean,
        customer_col=reader.customer_col,
        capacity_col=reader.capacity_col,
        pv_max_energy_per_kwp_kwh=reader.pv_max_energy_per_kwp_kwh,
        load_robust_z_threshold=reader.load_robust_z_threshold,
    )

    if len(clean) != expected_rows:
        raise ValueError(
            f"Customer {customer_id} 清洗后应有 {expected_rows} 行，"
            f"实际 {len(clean)} 行"
        )

    daily_counts = clean.groupby("source_date").size()
    bad_days = daily_counts[
        daily_counts != len(reader.interval_columns)
    ]
    if not bad_days.empty:
        raise ValueError(
            f"以下日期不是 {len(reader.interval_columns)} 个时段：\n"
            f"{bad_days.head(10).to_string()}"
        )

    return clean.sort_values("interval_start").reset_index(drop=True)


def main() -> None:
    reader = AusgridReader()
    sample_dir = Path(__file__).with_name("samples")
    sample_dir.mkdir(exist_ok=True)
    representatives = [
        (1, "Solar home 2010-2011.csv"),
        (11, "Solar home 2010-2011.csv"),
        (2, "Solar home 2012-2013.csv"),
    ]

    raw_cache: dict[str, pd.DataFrame] = {}
    report: list[dict[str, object]] = []
    for customer_id, source_file in representatives:
        if source_file not in raw_cache:
            raw_cache[source_file] = reader.read_single_csv(source_file)
        clean = process_customer_year(
            reader,
            customer_id=customer_id,
            source_file=source_file,
            raw_year=raw_cache[source_file],
        )
        source_year = reader.file_specs[source_file].source_year
        output_path = sample_dir / (
            f"customer_{customer_id}_{source_year}_clean.csv"
        )
        clean.to_csv(output_path, index=False, encoding="utf-8-sig")
        summary = summarize_clean_data(clean)
        summary["customer_id"] = customer_id
        summary["source_file"] = source_file
        summary["output_file"] = str(output_path)
        report.append(summary)
        print(
            f"[OK] Customer {customer_id}, {source_file}: "
            f"{summary['missing_days']} 个缺失日，"
            f"{summary['missing_energy_rows']} 个缺失时间点，"
            f"{summary['rows']:,} 行"
        )

    report_path = Path(__file__).with_name(
        "representative_regression_report.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 三户代表性报告：{report_path}")


if __name__ == "__main__":
    main()
