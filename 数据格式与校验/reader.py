from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FileSpec:
    name: str
    source_year: str
    date_format: str
    expected_columns_count: int
    expected_rows: int
    expected_unique_dates: int
    has_row_quality: bool


class AusgridReader:
    """按 config.toml 规则统一读取三份 Ausgrid 原始 CSV。"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else Path(__file__).with_name("config.toml").resolve()
        )
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{self.config_path}")

        with self.config_path.open("rb") as handle:
            cfg = tomllib.load(handle)

        self.cfg = cfg
        self.data_root = self._resolve_path(cfg["paths"]["data_root"])
        self.skiprows = int(cfg["read_rules"]["skiprows"])
        self.allowed_categories = {
            str(value) for value in cfg["read_rules"]["allowed_category"]
        }
        self.expected_customers = int(cfg["read_rules"]["expected_customers"])
        self.expected_postcodes = int(cfg["read_rules"]["expected_postcodes"])
        self.interval_columns = [str(value) for value in cfg["read_rules"]["interval_columns"]]
        self.timezone = str(cfg["quality_rules"]["timezone"])
        self.pv_max_energy_per_kwp_kwh = float(
            cfg["quality_rules"]["pv_max_energy_per_kwp_kwh"]
        )
        self.load_robust_z_threshold = float(
            cfg["quality_rules"]["load_robust_z_threshold"]
        )

        self.schema = cfg["schema"]
        self.customer_col = self.schema["col_customer"]
        self.capacity_col = self.schema["col_capacity"]
        self.postcode_col = self.schema["col_postcode"]
        self.category_col = self.schema["col_category"]
        self.date_col = self.schema["col_date"]
        self.quality_col = self.schema["col_quality"]

        self.files = [FileSpec(**item) for item in cfg["files"]]
        self.file_specs = {spec.name: spec for spec in self.files}
        self.last_file_reports: list[dict[str, object]] = []

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.config_path.parent / path
        return path.resolve()

    def _expected_columns(self, spec: FileSpec) -> list[str]:
        expected = [
            self.customer_col,
            self.capacity_col,
            self.postcode_col,
            self.category_col,
            self.date_col,
            *self.interval_columns,
        ]
        if spec.has_row_quality:
            expected.append(self.quality_col)
        return expected

    def _validate_columns(self, df: pd.DataFrame, spec: FileSpec) -> None:
        expected = self._expected_columns(spec)
        actual = df.columns.tolist()

        if len(actual) != spec.expected_columns_count:
            raise ValueError(
                f"[{spec.name}] 字段数量校验失败：实际 {len(actual)}，"
                f"配置预期 {spec.expected_columns_count}"
            )
        if actual != expected:
            mismatch = [
                f"第 {index + 1} 列：实际={actual_col!r}，预期={expected_col!r}"
                for index, (actual_col, expected_col) in enumerate(zip(actual, expected))
                if actual_col != expected_col
            ]
            if len(actual) != len(expected):
                mismatch.append(f"列数不同：实际={len(actual)}，预期={len(expected)}")
            raise ValueError(
                f"[{spec.name}] 字段顺序/名称校验失败：\n" + "\n".join(mismatch)
            )

    def _validate_required_values(self, df: pd.DataFrame, spec: FileSpec) -> None:
        required = [
            self.customer_col,
            self.capacity_col,
            self.postcode_col,
            self.category_col,
            self.date_col,
        ]
        missing = {col: int(df[col].isna().sum()) for col in required if df[col].isna().any()}
        if missing:
            raise ValueError(f"[{spec.name}] 关键字段存在空值：{missing}")

    def _validate_category(self, df: pd.DataFrame, spec: FileSpec) -> None:
        unique_categories = {str(value) for value in df[self.category_col].unique()}
        invalid = unique_categories - self.allowed_categories
        missing = self.allowed_categories - unique_categories
        if invalid:
            raise ValueError(
                f"[{spec.name}] 存在非法类别：{sorted(invalid)}；"
                f"只允许 {sorted(self.allowed_categories)}"
            )
        if missing:
            raise ValueError(f"[{spec.name}] 缺少类别：{sorted(missing)}")

    def _validate_customer_metadata(self, df: pd.DataFrame, spec: FileSpec) -> None:
        metadata_cardinality = (
            df.groupby(self.customer_col, observed=True)[
                [self.capacity_col, self.postcode_col]
            ]
            .nunique()
            .reset_index()
        )
        bad = metadata_cardinality[
            (metadata_cardinality[self.capacity_col] != 1)
            | (metadata_cardinality[self.postcode_col] != 1)
        ]
        if not bad.empty:
            raise ValueError(
                f"[{spec.name}] 有客户出现多个装机容量或邮编："
                f"{bad[self.customer_col].tolist()}"
            )

    def _validate_customer_and_postcode_counts(
        self, df: pd.DataFrame, spec: FileSpec
    ) -> None:
        actual_customers = int(df[self.customer_col].nunique())
        actual_postcodes = int(df[self.postcode_col].nunique())
        expected_customer_ids = set(range(1, self.expected_customers + 1))
        actual_customer_ids = set(df[self.customer_col].unique().tolist())

        if actual_customers != self.expected_customers:
            raise ValueError(
                f"[{spec.name}] 客户数错误：实际 {actual_customers}，"
                f"预期 {self.expected_customers}"
            )
        if actual_customer_ids != expected_customer_ids:
            raise ValueError(
                f"[{spec.name}] 客户编号不完整："
                f"缺少 {sorted(expected_customer_ids - actual_customer_ids)}，"
                f"多出 {sorted(actual_customer_ids - expected_customer_ids)}"
            )
        if actual_postcodes != self.expected_postcodes:
            raise ValueError(
                f"[{spec.name}] 邮编数量错误：实际 {actual_postcodes}，"
                f"预期 {self.expected_postcodes}"
            )

    def _parse_dates(self, df: pd.DataFrame, spec: FileSpec) -> pd.DataFrame:
        try:
            df[self.date_col] = pd.to_datetime(
                df[self.date_col], format=spec.date_format, errors="raise"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"[{spec.name}] 日期解析失败；配置格式为 {spec.date_format!r}。"
                "请确认日期格式与该文件一致。"
            ) from exc
        return df

    def _validate_dates_and_duplicates(self, df: pd.DataFrame, spec: FileSpec) -> None:
        unique_dates = int(df[self.date_col].nunique())
        if unique_dates != spec.expected_unique_dates:
            raise ValueError(
                f"[{spec.name}] 日期数错误：实际 {unique_dates}，"
                f"预期 {spec.expected_unique_dates}"
            )

        duplicate_mask = df.duplicated(
            [self.customer_col, self.date_col, self.category_col], keep=False
        )
        if duplicate_mask.any():
            examples = df.loc[
                duplicate_mask,
                [self.customer_col, self.date_col, self.category_col],
            ].head(10)
            raise ValueError(
                f"[{spec.name}] 客户+日期+类别存在 {int(duplicate_mask.sum())} 条重复记录；"
                f"示例：\n{examples.to_string(index=False)}"
            )

    def _validate_interval_values(self, df: pd.DataFrame, spec: FileSpec) -> None:
        values = df[self.interval_columns].to_numpy(dtype=float, copy=False)
        missing_cells = int(np.isnan(values).sum())
        non_finite_cells = int((~np.isfinite(values)).sum())
        negative_cells = int((values < 0).sum())

        if missing_cells:
            raise ValueError(f"[{spec.name}] 48 个电量字段存在 {missing_cells} 个空值")
        if non_finite_cells:
            raise ValueError(
                f"[{spec.name}] 48 个电量字段存在 {non_finite_cells} 个非有限值"
            )
        if negative_cells:
            raise ValueError(f"[{spec.name}] 48 个电量字段存在 {negative_cells} 个负值")

    def _validate_row_quality(self, df: pd.DataFrame, spec: FileSpec) -> None:
        if not spec.has_row_quality:
            return
        quality = df[self.quality_col].astype("string").fillna("")
        invalid = set(quality.unique()) - {"", "NA"}
        if invalid:
            raise ValueError(
                f"[{spec.name}] Row Quality 存在未知值：{sorted(invalid)}"
            )

    def _build_file_report(self, df: pd.DataFrame, spec: FileSpec) -> dict[str, object]:
        category_counts = {
            str(key): int(value)
            for key, value in df[self.category_col].value_counts().sort_index().items()
        }
        return {
            "file": spec.name,
            "source_year": spec.source_year,
            "rows": len(df),
            "raw_columns": spec.expected_columns_count,
            "columns_after_metadata": df.shape[1],
            "date_format": spec.date_format,
            "date_min": df[self.date_col].min().date().isoformat(),
            "date_max": df[self.date_col].max().date().isoformat(),
            "unique_dates": int(df[self.date_col].nunique()),
            "customers": int(df[self.customer_col].nunique()),
            "postcodes": int(df[self.postcode_col].nunique()),
            "categories": category_counts,
            "duplicate_keys": 0,
            "negative_energy_cells": 0,
            "missing_energy_cells": 0,
            "row_quality_present": bool(spec.has_row_quality),
            "row_quality_blank_rows": (
                int(df[self.quality_col].astype("string").eq("").sum())
                if spec.has_row_quality and self.quality_col in df.columns
                else int(len(df))
            ),
            "row_quality_na_rows": (
                int(df[self.quality_col].astype("string").eq("NA").sum())
                if spec.has_row_quality and self.quality_col in df.columns
                else 0
            ),
        }

    def read_single_csv(self, filename: str) -> pd.DataFrame:
        """读取并严格校验单份 CSV。"""
        spec = self.file_specs.get(filename)
        if spec is None:
            raise KeyError(f"文件未在 config.toml 的 [[files]] 中声明：{filename}")

        path = self.data_root / spec.name
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")

        dtype: dict[str, object] = {
            self.customer_col: "int16",
            self.capacity_col: "float32",
            self.postcode_col: "int32",
            self.category_col: "category",
            self.date_col: "string",
        }
        dtype.update({column: "float32" for column in self.interval_columns})

        try:
            df = pd.read_csv(
                path,
                skiprows=self.skiprows,
                dtype=dtype,
                low_memory=False,
                keep_default_na=False,
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"[{spec.name}] CSV 读取失败，请检查字段、编码和数值格式。"
            ) from exc

        self._validate_columns(df, spec)
        self._validate_required_values(df, spec)
        self._validate_category(df, spec)
        self._validate_customer_metadata(df, spec)
        self._validate_customer_and_postcode_counts(df, spec)
        df = self._parse_dates(df, spec)
        self._validate_dates_and_duplicates(df, spec)
        self._validate_interval_values(df, spec)
        self._validate_row_quality(df, spec)

        if len(df) != spec.expected_rows:
            raise ValueError(
                f"[{spec.name}] 行数错误：实际 {len(df):,}，"
                f"预期 {spec.expected_rows:,}"
            )
        if (df[self.capacity_col] <= 0).any():
            raise ValueError(f"[{spec.name}] 存在非正数光伏装机容量")

        if not spec.has_row_quality:
            df[self.quality_col] = ""
        df["source_year"] = spec.source_year
        df["source_file"] = spec.name
        report = self._build_file_report(df, spec)
        self.last_file_reports.append(report)

        print(
            f"[OK] {spec.name}: {len(df):,} 行，"
            f"{report['date_min']} 至 {report['date_max']}，"
            f"{report['customers']} 户"
        )
        return df

    def read_all(self) -> pd.DataFrame:
        """按固定顺序读取全部文件，并做跨年度一致性校验。"""
        self.last_file_reports = []
        frames = [self.read_single_csv(spec.name) for spec in self.files]
        df_all = pd.concat(frames, ignore_index=True)

        duplicate_mask = df_all.duplicated(
            [self.customer_col, self.date_col, self.category_col], keep=False
        )
        if duplicate_mask.any():
            raise ValueError(
                "跨文件拼接后出现客户+日期+类别重复，"
                f"数量={int(duplicate_mask.sum())}"
            )

        metadata_cardinality = (
            df_all.groupby(self.customer_col, observed=True)[
                [self.capacity_col, self.postcode_col]
            ]
            .nunique()
        )
        bad_customers = metadata_cardinality[
            (metadata_cardinality[self.capacity_col] != 1)
            | (metadata_cardinality[self.postcode_col] != 1)
        ].index.tolist()
        if bad_customers:
            raise ValueError(
                f"跨年度装机容量或邮编不一致的客户：{bad_customers}"
            )

        print(
            f"[OK] 三份文件读取完成：{len(df_all):,} 行，"
            f"{df_all[self.customer_col].nunique()} 户，"
            f"{df_all[self.postcode_col].nunique()} 个邮编"
        )
        return df_all

    def build_quality_report(self, df: pd.DataFrame) -> dict[str, object]:
        metadata_cardinality = df.groupby(self.customer_col, observed=True)[
            [self.capacity_col, self.postcode_col]
        ].nunique()
        customer_metadata_stable = bool((metadata_cardinality == 1).all().all())

        report = {
            "config_file": str(self.config_path),
            "data_root": str(self.data_root),
            "files": self.last_file_reports,
            "total_rows": len(df),
            "customers": int(df[self.customer_col].nunique()),
            "postcodes": int(df[self.postcode_col].nunique()),
            "date_min": df[self.date_col].min().date().isoformat(),
            "date_max": df[self.date_col].max().date().isoformat(),
            "categories": {
                str(key): int(value)
                for key, value in df[self.category_col].value_counts().sort_index().items()
            },
            "row_quality": {
                "blank": int(
                    sum(
                        int(report.get("row_quality_blank_rows", 0))
                        for report in self.last_file_reports
                    )
                ),
                "NA": int(
                    sum(
                        int(report.get("row_quality_na_rows", 0))
                        for report in self.last_file_reports
                    )
                ),
            },
            "checks": {
                "three_files_loaded": len(self.last_file_reports) == 3,
                "no_duplicate_keys": not df.duplicated(
                    [self.customer_col, self.date_col, self.category_col]
                ).any(),
                "no_missing_energy_cells": not df[self.interval_columns].isna().any().any(),
                "no_negative_energy_cells": not (df[self.interval_columns] < 0).any().any(),
                "customer_metadata_stable": customer_metadata_stable,
            },
        }
        return report


def main() -> None:
    reader = AusgridReader()
    df = reader.read_all()
    report = reader.build_quality_report(df)
    report_path = reader.config_path.with_name("data_quality_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 数据质量报告：{report_path}")


if __name__ == "__main__":
    main()
