from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from reader import FileSpec


def _format_date_ranges(dates: list[pd.Timestamp]) -> str:
    if not dates:
        return ""

    ordered = sorted(pd.Timestamp(value).normalize() for value in dates)
    ranges: list[str] = []
    start = previous = ordered[0]

    for current in ordered[1:]:
        if current == previous + pd.Timedelta(days=1):
            previous = current
            continue
        ranges.append(
            start.date().isoformat()
            if start == previous
            else f"{start.date().isoformat()}..{previous.date().isoformat()}"
        )
        start = previous = current

    ranges.append(
        start.date().isoformat()
        if start == previous
        else f"{start.date().isoformat()}..{previous.date().isoformat()}"
    )
    return ",".join(ranges)


def screen_completeness(
    raw_year: pd.DataFrame,
    spec: FileSpec,
    *,
    customer_col: str = "Customer",
    date_col: str = "date",
    category_col: str = "Consumption Category",
    quality_col: str = "Row Quality",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按客户筛查缺天、类别行缺失和非实际数据行。"""
    file_start = raw_year[date_col].min().normalize()
    file_end = raw_year[date_col].max().normalize()
    expected_dates = pd.date_range(file_start, file_end, freq="D")

    category_daily = (
        raw_year.assign(_date=raw_year[date_col].dt.normalize())
        .groupby([customer_col, "_date", category_col], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    non_actual = (
        raw_year.assign(_non_actual=raw_year[quality_col].astype("string").eq("NA"))
        .groupby(customer_col, observed=True)["_non_actual"]
        .sum()
        .astype(int)
    )

    rows: list[dict[str, object]] = []
    for customer_id in sorted(raw_year[customer_col].unique().tolist()):
        presence = category_daily.xs(customer_id, level=customer_col).reindex(
            expected_dates, fill_value=0
        )
        observed_dates = presence.index[presence.sum(axis=1).gt(0)]
        observed_set = set(observed_dates)
        missing_dates = [
            value for value in expected_dates if value not in observed_set
        ]

        has_cl = bool(presence.get("CL", pd.Series(0, index=presence.index)).gt(0).any())
        observed_presence = presence.loc[observed_dates]
        bad_gc_days = int(observed_presence.get("GC", 0).ne(1).sum())
        bad_gg_days = int(observed_presence.get("GG", 0).ne(1).sum())
        cl_observed_days = int(
            observed_presence.get("CL", pd.Series(0, index=observed_dates))
            .gt(0)
            .sum()
        )
        cl_absent_days = int(len(observed_dates) - cl_observed_days)

        rows.append(
            {
                "customer_id": int(customer_id),
                "expected_days": len(expected_dates),
                "observed_days": len(observed_dates),
                "missing_days": len(missing_dates),
                "missing_date_ranges": _format_date_ranges(missing_dates),
                "has_controlled_load": has_cl,
                "missing_gc_category_days": bad_gc_days,
                "missing_gg_category_days": bad_gg_days,
                "cl_observed_days": cl_observed_days,
                "cl_absent_days": cl_absent_days,
                "non_actual_source_rows": int(non_actual.get(customer_id, 0)),
            }
        )

    details = pd.DataFrame(rows)
    incomplete = details[
        details["missing_days"].gt(0)
        | details["missing_gc_category_days"].gt(0)
        | details["missing_gg_category_days"].gt(0)
    ]
    summary = {
        "file": spec.name,
        "source_year": spec.source_year,
        "expected_days": len(expected_dates),
        "customers": len(details),
        "complete_customers": int(len(details) - len(incomplete)),
        "incomplete_customers": int(len(incomplete)),
        "customers_with_missing_days": int(details["missing_days"].gt(0).sum()),
        "total_missing_customer_days": int(details["missing_days"].sum()),
        "non_actual_source_rows": int(details["non_actual_source_rows"].sum()),
        "incomplete_customer_details": incomplete.to_dict(orient="records"),
    }
    return details, summary


def add_time_metadata(
    df: pd.DataFrame,
    *,
    time_col: str = "interval_start",
    source_date_col: str = "source_date",
    timezone: str = "Australia/Sydney",
) -> pd.DataFrame:
    """
    保留本地无时区时间，并增加尽力而为的 UTC 时间。

    DST 回拨时重复出现的小时和 DST 前拨时不存在的小时无法唯一映射，
    因此对应行的 UTC 时间为 NaT，而不是猜测一个偏移。
    """
    required = [time_col, source_date_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"时区处理缺少字段：{missing}")

    ZoneInfo(timezone)
    result = df.copy()
    local_naive = pd.to_datetime(result[time_col])
    localized = local_naive.dt.tz_localize(
        timezone,
        ambiguous="NaT",
        nonexistent="NaT",
    )
    result["interval_start_local"] = localized
    result["interval_start_utc"] = localized.dt.tz_convert("UTC")
    result["utc_resolvable"] = (
        local_naive.notna() & result["interval_start_utc"].notna()
    )

    offsets = localized.map(
        lambda value: value.utcoffset() if pd.notna(value) else pd.NaT
    )
    result["utc_offset_hours"] = offsets.map(
        lambda value: value.total_seconds() / 3600 if pd.notna(value) else np.nan
    )
    dst_offsets = localized.map(
        lambda value: value.dst() if pd.notna(value) else pd.NaT
    )
    result["is_dst"] = dst_offsets.map(
        lambda value: value.total_seconds() > 0 if pd.notna(value) else False
    )

    unique_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(result[source_date_col]).dt.normalize().unique())
    )
    local_midnight = unique_dates.tz_localize(timezone)
    next_midnight = (unique_dates + pd.Timedelta(days=1)).tz_localize(timezone)
    transitions = pd.Series(
        [
            left.utcoffset() != right.utcoffset()
            for left, right in zip(local_midnight, next_midnight, strict=True)
        ],
        index=unique_dates,
    )
    result["dst_transition_day"] = (
        pd.to_datetime(result[source_date_col])
        .dt.normalize()
        .map(transitions)
        .fillna(False)
        .astype(bool)
    )
    return result


def add_outlier_flags(
    df: pd.DataFrame,
    *,
    customer_col: str = "Customer",
    capacity_col: str = "Generator Capacity",
    pv_col: str = "gg_kwh",
    load_col: str = "total_load_kwh",
    pv_max_energy_per_kwp_kwh: float = 0.8,
    load_robust_z_threshold: float = 8.0,
) -> pd.DataFrame:
    """标记异常但保留原始值；仅将物理上越界的光伏电量放入清洗列。"""
    required = [customer_col, capacity_col, pv_col, load_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"异常值处理缺少字段：{missing}")

    result = df.copy()
    pv_limit = result[capacity_col] * pv_max_energy_per_kwp_kwh
    result["pv_energy_limit_kwh"] = pv_limit
    result["pv_over_capacity"] = result[pv_col].gt(pv_limit).fillna(False)
    result["gg_clean_kwh"] = result[pv_col].mask(result["pv_over_capacity"])

    log_load = np.log1p(result[load_col])
    grouped = log_load.groupby(result[customer_col], observed=True)
    median = grouped.transform("median")
    mad = (log_load - median).abs().groupby(
        result[customer_col], observed=True
    ).transform("median")
    robust_z = (0.6745 * (log_load - median) / mad.where(mad.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )
    result["load_robust_z"] = robust_z
    result["load_extreme"] = robust_z.gt(load_robust_z_threshold).fillna(False)

    result["model_usable"] = (
        ~result.get("quality_non_actual", False)
        & ~result["pv_over_capacity"]
    )
    if "is_missing_day" in result.columns:
        result["model_usable"] &= ~result["is_missing_day"]
    if "is_missing_interval" in result.columns:
        result["model_usable"] &= ~result["is_missing_interval"]
    return result


def summarize_clean_data(df: pd.DataFrame) -> dict[str, object]:
    energy_columns = [
        "gc_kwh",
        "cl_kwh",
        "gg_kwh",
        "total_load_kwh",
        "net_grid_kwh",
        "import_kwh",
        "export_kwh",
    ]
    missing_energy = df[energy_columns].isna().any(axis=1)
    summary = {
        "rows": len(df),
        "customers": int(df["Customer"].nunique()),
        "dates": int(df["source_date"].nunique()),
        "missing_energy_rows": int(missing_energy.sum()),
        "missing_days": (
            int(df.loc[df["is_missing_day"], "source_date"].nunique())
            if "is_missing_day" in df.columns
            else 0
        ),
        "quality_non_actual_rows": (
            int(df["quality_non_actual"].sum())
            if "quality_non_actual" in df.columns
            else 0
        ),
        "pv_over_capacity_rows": (
            int(df["pv_over_capacity"].sum())
            if "pv_over_capacity" in df.columns
            else 0
        ),
        "load_extreme_rows": (
            int(df["load_extreme"].sum())
            if "load_extreme" in df.columns
            else 0
        ),
        "utc_unresolved_rows": (
            int((~df["utc_resolvable"]).sum())
            if "utc_resolvable" in df.columns
            else 0
        ),
        "dst_transition_days": (
            int(df.loc[df["dst_transition_day"], "source_date"].nunique())
            if "dst_transition_day" in df.columns
            else 0
        ),
        "model_usable_rows": (
            int(df["model_usable"].sum())
            if "model_usable" in df.columns
            else 0
        ),
    }
    return summary


def count_raw_pv_physical_outliers(
    raw_year: pd.DataFrame,
    *,
    interval_columns: list[str],
    category_col: str = "Consumption Category",
    capacity_col: str = "Generator Capacity",
    customer_col: str = "Customer",
    pv_max_energy_per_kwp_kwh: float = 0.8,
) -> dict[str, int]:
    pv = raw_year.loc[raw_year[category_col].astype(str).eq("GG")].copy()
    values = pv[interval_columns].to_numpy(dtype=float)
    limits = (
        pv[capacity_col].to_numpy(dtype=float)[:, None]
        * pv_max_energy_per_kwp_kwh
    )
    over = values > limits
    row_over = over.any(axis=1)
    return {
        "cells": int(over.sum()),
        "rows": int(row_over.sum()),
        "customers": int(pv.loc[row_over, customer_col].nunique()),
    }


def build_quality_report() -> Path:
    from reader import AusgridReader

    reader = AusgridReader()
    files: list[dict[str, object]] = []
    for spec in reader.files:
        raw_year = reader.read_single_csv(spec.name)
        _, summary = screen_completeness(raw_year, spec)
        summary["pv_physical_outliers"] = count_raw_pv_physical_outliers(
            raw_year,
            interval_columns=reader.interval_columns,
            category_col=reader.category_col,
            capacity_col=reader.capacity_col,
            customer_col=reader.customer_col,
            pv_max_energy_per_kwp_kwh=reader.pv_max_energy_per_kwp_kwh,
        )
        files.append(summary)

    report = {
        "timezone": reader.timezone,
        "policy": {
            "missing_days": "保留为缺失，不插值",
            "missing_cl": "非控负荷家庭补 0，并保留 has_controlled_load",
            "row_quality_na": "保留原始值，并标记 quality_non_actual",
            "outliers": "标记异常；仅光伏物理越界值写入 gg_clean_kwh",
            "dst": "无法唯一映射的本地时间写入 NaT，不猜测偏移",
        },
        "files": files,
        "totals": {
            "expected_customer_days": int(
                sum(item["expected_days"] * item["customers"] for item in files)
            ),
            "customers_with_missing_days": int(
                sum(item["customers_with_missing_days"] for item in files)
            ),
            "missing_customer_days": int(
                sum(item["total_missing_customer_days"] for item in files)
            ),
            "non_actual_source_rows": int(
                sum(item["non_actual_source_rows"] for item in files)
            ),
            "pv_over_capacity_cells": int(
                sum(item["pv_physical_outliers"]["cells"] for item in files)
            ),
        },
    }

    output_path = Path(__file__).with_name("data_quality_extended_report.json")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    print(f"[OK] 扩展数据质量报告：{build_quality_report()}")
