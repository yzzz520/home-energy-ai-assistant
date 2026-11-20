from __future__ import annotations

from pathlib import Path

import pandas as pd

from reader import AusgridReader
from time_axis import wide_to_interval_long


def pivot_categories(
    long: pd.DataFrame,
    *,
    customer_col: str = "Customer",
    category_col: str = "Consumption Category",
    capacity_col: str = "Generator Capacity",
    postcode_col: str = "Postcode",
    quality_col: str = "Row Quality",
    allowed_categories: tuple[str, ...] = ("GC", "CL", "GG"),
    required_categories: tuple[str, ...] = ("GC", "GG"),
) -> pd.DataFrame:
    """
    把同一客户、同一时间的 GC/CL/GG 三行旋转成三个电量字段。

    非控负荷家庭没有 CL 类别，这是业务上的 0，不是数据缺失。
    GC 或 GG 缺失仍然视为异常。
    """
    actual_categories = set(long[category_col].astype(str).unique())
    unexpected = actual_categories - set(allowed_categories)
    if unexpected:
        raise ValueError(f"存在未预期类别：{sorted(unexpected)}")

    key_columns = [customer_col, "source_date", "interval_start"]
    duplicate = long.duplicated([*key_columns, category_col], keep=False)
    if duplicate.any():
        raise ValueError(
            f"旋转前存在客户+时间+类别重复记录：{int(duplicate.sum())} 条"
        )

    category_values = long.assign(
        _category_text=long[category_col].astype(str)
    )
    present_categories = category_values.groupby(
        key_columns, observed=True
    )["_category_text"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    missing_required = present_categories[
        ~present_categories.map(lambda values: set(required_categories).issubset(values))
    ]
    if not missing_required.empty:
        raise ValueError(
            "部分客户时间点缺少 GC 或 GG：\n"
            f"{missing_required.head(10).to_string()}"
        )

    has_controlled_load = (
        category_values.groupby(customer_col, observed=True)["_category_text"]
        .agg(lambda values: "CL" in tuple(values))
        .rename("has_controlled_load")
    )

    metadata_columns = [
        customer_col,
        "source_date",
        "interval_start",
        "interval_end",
        "slot_index",
        capacity_col,
        postcode_col,
        "source_year",
        "source_file",
    ]
    index_columns = [column for column in metadata_columns if column in long.columns]

    wide = long.pivot(
        index=index_columns,
        columns=category_col,
        values="energy_kwh",
    ).reset_index()
    wide.columns.name = None

    wide = wide.rename(
        columns={
            "GC": "gc_kwh",
            "CL": "cl_kwh",
            "GG": "gg_kwh",
        }
    )
    for column in ("gc_kwh", "cl_kwh", "gg_kwh"):
        if column not in wide.columns:
            wide[column] = 0.0 if column == "cl_kwh" else pd.NA

    if quality_col in long.columns:
        quality_wide = long.pivot(
            index=index_columns,
            columns=category_col,
            values=quality_col,
        ).reset_index()
        quality_wide.columns.name = None
        quality_wide = quality_wide.rename(
            columns={
                "GC": "gc_quality",
                "CL": "cl_quality",
                "GG": "gg_quality",
            }
        )
        for column in ("gc_quality", "cl_quality", "gg_quality"):
            if column not in quality_wide.columns:
                quality_wide[column] = ""
        quality_wide["cl_quality"] = quality_wide["cl_quality"].fillna("NO_CL")
        quality_wide["gc_quality"] = quality_wide["gc_quality"].fillna("")
        quality_wide["gg_quality"] = quality_wide["gg_quality"].fillna("")
        quality_columns = [
            *index_columns,
            "gc_quality",
            "cl_quality",
            "gg_quality",
        ]
        wide = wide.merge(
            quality_wide[quality_columns],
            on=index_columns,
            how="left",
            validate="one_to_one",
        )
    else:
        wide["gc_quality"] = ""
        wide["cl_quality"] = ""
        wide["gg_quality"] = ""

    wide["quality_non_actual"] = wide[
        ["gc_quality", "cl_quality", "gg_quality"]
    ].eq("NA").any(axis=1)

    wide["cl_kwh"] = wide["cl_kwh"].fillna(0.0)
    if wide[["gc_kwh", "gg_kwh"]].isna().any().any():
        raise ValueError("旋转后 GC 或 GG 仍存在缺失")

    wide = wide.join(has_controlled_load, on=customer_col)
    wide = wide.sort_values(
        [customer_col, "source_date", "interval_start"]
    ).reset_index(drop=True)

    if wide.duplicated([customer_col, "interval_start"]).any():
        raise ValueError("旋转后仍存在客户+interval_start重复")
    if len(wide) != len(present_categories):
        raise ValueError(
            f"旋转行数异常：输入时间点 {len(present_categories)}，"
            f"输出行数 {len(wide)}"
        )

    return wide


def build_pivot_sample(
    reader: AusgridReader,
    *,
    customer_id: int = 1,
    source_date: str = "2010-07-01",
    source_file: str = "Solar home 2010-2011.csv",
) -> pd.DataFrame:
    wide = reader.read_single_csv(source_file)
    target_date = pd.Timestamp(source_date)
    sample = wide.loc[
        (wide[reader.customer_col] == customer_id)
        & (wide[reader.date_col] == target_date)
    ].copy()
    if sample.empty:
        raise ValueError(
            f"没有找到 Customer={customer_id}, date={source_date} 的数据"
        )

    long = wide_to_interval_long(
        sample,
        reader.interval_columns,
        customer_col=reader.customer_col,
        date_col=reader.date_col,
        category_col=reader.category_col,
    )
    return pivot_categories(
        long,
        customer_col=reader.customer_col,
        category_col=reader.category_col,
        capacity_col=reader.capacity_col,
        postcode_col=reader.postcode_col,
        quality_col=reader.quality_col,
        allowed_categories=tuple(sorted(reader.allowed_categories)),
        required_categories=("GC", "GG"),
    )


def main() -> None:
    reader = AusgridReader()
    sample_dir = Path(__file__).with_name("samples")
    sample_dir.mkdir(exist_ok=True)

    customer_ids = (1, 11)
    outputs: list[Path] = []
    for customer_id in customer_ids:
        wide = build_pivot_sample(reader, customer_id=customer_id)
        output_path = sample_dir / (
            f"customer_{customer_id}_2010-07-01_category_wide.csv"
        )
        wide.to_csv(output_path, index=False, encoding="utf-8-sig")
        outputs.append(output_path)

        self_use_text = "有 CL" if wide["has_controlled_load"].iloc[0] else "无 CL，已补 0"
        print(
            f"[OK] Customer {customer_id}: {len(wide)} 行，"
            f"GC/CL/GG 已并排，{self_use_text}"
        )

    print("[OK] 样例文件：")
    for output_path in outputs:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
