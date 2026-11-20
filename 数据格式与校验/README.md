# Ausgrid 数据读取与校验

这一阶段只负责把三份原始 CSV 按各自规则读入，并完成结构、日期、类别、客户元数据和电量值校验。原始宽表暂不做 `melt`、`0:00` 时间轴转换或净负荷计算。

## 运行

在项目根目录或任意目录执行：

```powershell
python "数据格式与校验/reader.py"
```

成功后会生成：

- `data_quality_report.json`

## 测试

```powershell
Set-Location "数据格式与校验"
python -m unittest -v test_reader.py
```

当前测试覆盖：

- 三份文件总行数和类别行数
- 300 户、100 个邮编、客户编号完整
- 客户日期类别组合无重复
- 光伏容量与邮编在每户内唯一
- Customer 2 在 2012-10-12 至 2012-12-31 的 81 天空缺未被抹除
- 48 个电量字段无缺失、无负值

## 时间轴展开样例

```powershell
Set-Location "数据格式与校验"
python time_axis.py
python -m unittest -v test_time_axis.py
```

本步骤只处理一个样例客户日：

- 将 48 个横向时段字段展开为 48 行
- 生成 `interval_start`、`interval_end` 和 `slot_index`
- 把原始 `0:00` 正确映射到源日期次日 00:00 结束、当天 23:30 开始
- 暂时不旋转 GC/CL/GG，也不计算总负荷和净负荷

输出样例：

- `samples/customer_1_2010-07-01_interval_long.csv`

## GC/CL/GG 类别旋转样例

```powershell
Set-Location "数据格式与校验"
python category_pivot.py
python -m unittest -v test_category_pivot.py
```

本步骤只做类别旋转：

- 同一客户同一 `interval_start` 的 GC/CL/GG 三行变成一行
- 输出 `gc_kwh`、`cl_kwh`、`gg_kwh`
- 有控负荷家庭保留 CL 电量
- 无控负荷家庭把 CL 补为 0，并标记 `has_controlled_load=false`
- 暂不计算总负荷和净负荷

输出样例：

- `samples/customer_1_2010-07-01_category_wide.csv`
- `samples/customer_11_2010-07-01_category_wide.csv`

## 总负荷与净负荷样例

```powershell
Set-Location "数据格式与校验"
python energy_balance.py
python -m unittest -v test_energy_balance.py
```

计算定义：

```text
total_load_kwh = gc_kwh + cl_kwh
net_grid_kwh = total_load_kwh - gg_kwh
```

- `net_grid_kwh > 0`：从电网购电
- `net_grid_kwh < 0`：向电网送电
- `net_grid_kwh = 0`：光伏与负荷刚好平衡

本步骤不拆分购电和上网电量，也不计算成本和储能调度。

输出样例：

- `samples/customer_1_2010-07-01_energy_balance.csv`
- `samples/customer_11_2010-07-01_energy_balance.csv`

## 购电与送电拆分样例

```powershell
Set-Location "数据格式与校验"
python energy_balance.py
python -m unittest -v test_energy_balance.py
```

拆分公式：

```text
import_kwh = max(net_grid_kwh, 0)
export_kwh = max(-net_grid_kwh, 0)
```

校验关系：

```text
net_grid_kwh = import_kwh - export_kwh
```

同一时段 `import_kwh` 和 `export_kwh` 不能同时大于零。

## 单客户全年流水线

```powershell
Set-Location "数据格式与校验"
python pipeline.py
python -m unittest -v test_pipeline.py
```

当前只处理 Customer 1 的 2010-2011 年度：

```text
原始读取
-> 48 个横向时段展开
-> GC/CL/GG 类别旋转
-> 总负荷与净负荷
-> 购电与送电拆分
```

验收结果：

- 365 个日期
- 每天 48 个半小时点
- 共 17,520 行

输出样例：

- `samples/customer_1_2010-2011_clean.csv`

## 配置原则

- `config.toml` 的路径相对配置文件自身解析，不依赖启动目录。
- 每份文件单独声明日期格式、原始列数、预期行数和预期日期数。
- `0:00` 在原始表中位于最后，本阶段只保留原列名，不在读取器中提前改时间轴。
- `Row Quality` 只有 2010-2011 没有，后续两年字段取值为空串或 `NA`。
- `Row Quality=NA` 表示该行含估算或替代值，必须保留并计入质量统计。

## 缺失、异常、时区与完整性

```powershell
Set-Location "数据格式与校验"
python quality.py
python -m unittest -v test_quality.py
```

处理策略：

- 缺天保留为完整时间网格中的缺失值，不插值。
- 非控负荷家庭的 CL 补 0；有控负荷家庭某些日期没有 CL 行也属于正常业务，不记为不完整。
- `Row Quality=NA` 保留原始电量，同时生成 `quality_non_actual=true`。
- 光伏电量超过物理阈值时标记 `pv_over_capacity`，清洗列为 `gg_clean_kwh`。
- 负荷统计异常只标记 `load_extreme`，不删除真实晚高峰。
- 本地时间按 `Australia/Sydney` 保留，并生成可判定时的 UTC 时间。
- DST 重复或不存在的本地时间对应 UTC 设为 `NaT`，不猜测偏移。

扩展报告：

- `data_quality_extended_report.json`

## 300 户流式批处理

```powershell
Set-Location "数据格式与校验"
python batch_process.py
python -m unittest -v test_batch_process.py
```

每个年度文件只读取一次，然后逐户执行完整流水线。不会同时把所有客户清洗明细堆在内存中。

输出：

- `reports/annual_quality_summary.json`
- `reports/annual_quality_by_customer.csv`

年度汇总包含缺天客户数、缺失客户日、缺失时间点、Row Quality 非实际行、光伏异常行、负荷异常行、DST 无法映射行和模型可用行数。

## EDA

EDA 代码、测试、图表和聚类结果已迁移到：

- `EDA探索性数据分析/`

运行方式和输出说明见 `EDA探索性数据分析/README.md`。
