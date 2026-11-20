from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from category_pivot import build_pivot_sample
from reader import AusgridReader


def add_energy_balance(
    df: pd.DataFrame,
    *,
    gc_col: str = "gc_kwh",
    cl_col: str = "cl_kwh",
    pv_col: str = "gg_kwh",
) -> pd.DataFrame:
    """
    计算家庭总用电和并网净负荷。

    total_load_kwh = GC + CL
    net_grid_kwh = total_load_kwh - GG

    net_grid_kwh > 0 表示从电网购电；
    net_grid_kwh < 0 表示向电网送电。
    本步骤不拆分购电和上网电量。
    """
    required = [gc_col, cl_col, pv_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"计算能量平衡缺少字段：{missing}")
    if df[required].isna().any().any():
        raise ValueError("GC、CL、GG 中存在空值，不能计算能量平衡")
    if (df[required] < 0).any().any():
        raise ValueError("GC、CL、GG 中存在负值，不能计算能量平衡")

    result = df.copy()
    result["total_load_kwh"] = result[gc_col] + result[cl_col]
    result["net_grid_kwh"] = result["total_load_kwh"] - result[pv_col]

    expected_net = result["total_load_kwh"] - result[pv_col]
    if not result["net_grid_kwh"].equals(expected_net):
        raise ValueError("净负荷计算校验失败")

    return result


def add_grid_flow(
    df: pd.DataFrame,
    *,
    net_grid_col: str = "net_grid_kwh",
) -> pd.DataFrame:
    """
    将带符号的净负荷拆成购电和送电两个非负字段。

    import_kwh = max(net_grid_kwh, 0)
    export_kwh = max(-net_grid_kwh, 0)
    """
    if net_grid_col not in df.columns:
        raise ValueError(f"拆分购售电缺少字段：{net_grid_col}")
    if df[net_grid_col].isna().any():
        raise ValueError("net_grid_kwh 存在空值，不能拆分购售电")

    result = df.copy()
    result["import_kwh"] = result[net_grid_col].clip(lower=0.0)
    result["export_kwh"] = (-result[net_grid_col]).clip(lower=0.0)

    if (result[["import_kwh", "export_kwh"]] < 0).any().any():
        raise ValueError("购电或送电字段出现负值")
    if ((result["import_kwh"] > 0) & (result["export_kwh"] > 0)).any():
        raise ValueError("同一时段不能同时购电和送电")

    reconstructed = result["import_kwh"] - result["export_kwh"]
    if not np.allclose(
        result[net_grid_col].to_numpy(dtype=float),
        reconstructed.to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("购售电拆分校验失败：import - export 不等于 net_grid")

    return result


def build_balance_sample(
    reader: AusgridReader,
    *,
    customer_id: int = 1,
    source_date: str = "2010-07-01",
    source_file: str = "Solar home 2010-2011.csv",
) -> pd.DataFrame:
    category_wide = build_pivot_sample(
        reader,
        customer_id=customer_id,
        source_date=source_date,
        source_file=source_file,
    )
    balance = add_energy_balance(category_wide)
    return add_grid_flow(balance)


def main() -> None:
    reader = AusgridReader()
    sample_dir = Path(__file__).with_name("samples")
    sample_dir.mkdir(exist_ok=True)

    outputs: list[Path] = []
    for customer_id in (1, 11):
        balance = build_balance_sample(reader, customer_id=customer_id)
        output_path = sample_dir / (
            f"customer_{customer_id}_2010-07-01_energy_balance.csv"
        )
        balance.to_csv(output_path, index=False, encoding="utf-8-sig")
        outputs.append(output_path)

        import_periods = int((balance["import_kwh"] > 0).sum())
        export_periods = int((balance["export_kwh"] > 0).sum())
        print(
            f"[OK] Customer {customer_id}: {len(balance)} 行，"
            f"购电时段 {import_periods}，送电时段 {export_periods}"
        )

    print("[OK] 样例文件：")
    for output_path in outputs:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
