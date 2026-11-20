from __future__ import annotations

from pathlib import Path

import pandas as pd

from reader import AusgridReader


def wide_to_interval_long(
    df: pd.DataFrame,
    interval_columns: list[str],
    *,
    customer_col: str = "Customer",
    date_col: str = "date",
    category_col: str = "Consumption Category",
) -> pd.DataFrame:
    """
    将 48 个横向半小时列展开为 48 行。

    原始 0:00 表示当天结束后的午夜，因此先把区间结束时间映射到次日
    00:00，再减 30 分钟，得到当天 23:30 的区间开始时间。
    """
    missing = [column for column in interval_columns if column not in df.columns]
    if missing:
        raise ValueError(f"缺少半小时字段：{missing}")
    if df.duplicated([customer_col, date_col, category_col]).any():
        raise ValueError("转换前存在重复的 Customer + date + Consumption Category")

    id_columns = [column for column in df.columns if column not in interval_columns]
    long = df.melt(
        id_vars=id_columns,
        value_vars=interval_columns,
        var_name="interval_end_label",
        value_name="energy_kwh",
    )
    long = long.rename(columns={date_col: "source_date"})

    labels = long["interval_end_label"].astype(str)
    hour_minute = labels.str.split(":", expand=True).astype("int16")
    minutes = hour_minute[0] * 60 + hour_minute[1]

    ends_next_day = labels.eq("0:00").astype("int8")
    long["interval_end"] = (
        long["source_date"]
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(ends_next_day, unit="D")
    )
    long["interval_start"] = long["interval_end"] - pd.Timedelta(minutes=30)
    long["slot_index"] = (
        long["interval_start"].dt.hour * 2
        + long["interval_start"].dt.minute // 30
    ).astype("int8")

    long = long.sort_values(
        [customer_col, "source_date", category_col, "interval_start"]
    ).reset_index(drop=True)

    counts = long.groupby(
        [customer_col, "source_date", category_col], observed=True
    ).size()
    bad = counts[counts != len(interval_columns)]
    if not bad.empty:
        raise ValueError(
            "每户每日每类别应展开为 "
            f"{len(interval_columns)} 行，异常组合：\n{bad.head(10).to_string()}"
        )

    return long


def build_sample(
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
    return wide_to_interval_long(
        sample,
        reader.interval_columns,
        customer_col=reader.customer_col,
        date_col=reader.date_col,
        category_col=reader.category_col,
    )


def main() -> None:
    reader = AusgridReader()
    long = build_sample(reader)

    sample_dir = Path(__file__).with_name("samples")
    sample_dir.mkdir(exist_ok=True)
    output_path = sample_dir / "customer_1_2010-07-01_interval_long.csv"
    long.to_csv(output_path, index=False, encoding="utf-8-sig")

    counts = long.groupby(reader.category_col, observed=True)["interval_start"].nunique()
    print(f"[OK] 展开后共 {len(long)} 行")
    print(f"[OK] 各类别唯一时间点数：{counts.to_dict()}")
    print(
        "[OK] 时间范围："
        f"{long['interval_start'].min()} 至 {long['interval_start'].max()}"
    )
    print(f"[OK] 样例文件：{output_path}")


if __name__ == "__main__":
    main()
