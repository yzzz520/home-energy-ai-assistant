# 储能调度

## 第一小步：电池模型

当前只包含电池参数、SOC 递推和调度合法性校验，不包含成本目标、购售电优化或滚动 MPC。

### 功率约定

- `charge_kw > 0`：向电池充电
- `discharge_kw > 0`：电池放电
- 同一时刻不允许充放电同时大于 0
- 功率均为 AC 侧功率

### SOC 递推

```text
充入电池电量 = charge_kw × dt_hours × charge_efficiency
电池释放电量 = discharge_kw × dt_hours / discharge_efficiency
SOC_next = SOC_current + 充入电池电量 - 电池释放电量
```

### 文件

- `battery.py`
- `battery.toml`
- `test_battery.py`

### 测试

```powershell
Set-Location "储能调度"
python -m unittest -v test_battery.py
```

## 第二小步：功率平衡与规则调度

并网净功率：

```text
grid_net_kw = total_load_kw - pv_kw - battery_net_ac_kw
```

- `grid_net_kw > 0`：购电
- `grid_net_kw < 0`：售电
- `battery_net_ac_kw > 0`：电池放电
- `battery_net_ac_kw < 0`：电池充电

规则调度：

- 光伏净余量只用于充电
- 17:00-21:00 晚峰放电
- 只覆盖本地净负荷，不向电网倒送电池电量

运行：

```powershell
python rule_dispatch_demo.py
python -m unittest -v test_power_balance.py
```

输出：

- `results/rule_dispatch_customer_1_2010-07-01.csv`
- `results/rule_dispatch_customer_1_2010-07-01_metrics.json`
- `results/rule_dispatch_customer_1_2010-07-01.png`

## 第三小步：MILP 成本优化

使用 PuLP + CBC 求解：

- 购电价：谷时 0.18、平时 0.28、峰时 0.48 AUD/kWh
- 上网电价：0.08 AUD/kWh
- 日末 SOC 回到初始 SOC
- 充放电互斥
- 购售电互斥

对比策略只保留：

- 规则调度
- PuLP MILP 优化调度

运行：

```powershell
python cost_comparison_demo.py
python -m unittest -v test_optimization.py
```

输出：

- `results/cost_comparison_customer_1_2010-07-01.json`
- `results/optimized_dispatch_customer_1_2010-07-01.csv`
- `results/cost_comparison_customer_1_2010-07-01.png`

### 退化成本敏感性

```powershell
python degradation_sensitivity.py
```

退化成本按每 kWh 放电量计价，扫描：

```text
0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00 AUD/kWh
```

输出：

- `results/degradation_sensitivity_customer_1_2010-07-01.csv`
- `results/degradation_sensitivity_customer_1_2010-07-01.json`
- `results/degradation_sensitivity_customer_1_2010-07-01.png`

## 预测误差对调度的真实影响

```powershell
python forecast_impact_demo.py
python -m unittest -v test_forecast_impact.py
```

对比三个 60 天实际结算成本：

- 规则调度：按实际负荷和光伏响应
- 预测驱动 MILP：LightGBM 负荷预测 + 日前 PV 季节朴素预测
- 完美预知 MILP：使用真实负荷和光伏

所有策略最终都按真实负荷、真实光伏、相同电价和相同退化成本结算。

输出：

- `results/forecast_vs_perfect_customer_1.json`
- `results/forecast_vs_perfect_daily_customer_1.csv`
- `results/forecast_vs_perfect_customer_1.png`

## 预测误差诊断与规则回退安全层

```powershell
python forecast_diagnostics.py
python -m unittest -v test_forecast_diagnostics.py
```

诊断内容：

- LightGBM 负荷系统偏差与 MAE
- PV 朴素预测系统偏差与 MAE
- 预测驱动 MILP 的 regret 按小时汇总
- LightGBM 与朴素预测在相同 MILP 下的效果
- “预测收益不足则回退规则”的阈值扫描

输出：

- `results/forecast_diagnostics_customer_1.json`
- `results/forecast_diagnostics_daily_customer_1.csv`
- `results/forecast_diagnostics_hourly_quality.csv`
- `results/forecast_diagnostics_hourly_regret.csv`
- `results/forecast_diagnostics_fallback_sweep.csv`
- `results/forecast_diagnostics_customer_1.png`

## 轻量稳健调度与规则安全层

```powershell
python robust_dispatch.py
python -m unittest -v test_robust_dispatch.py
```

稳健机制：

- 使用过去 14 天滚动预测误差
- 16:00-19:00 负荷使用 60% 分位低估误差作为轻量安全余量
- PV 使用 60% 分位高估误差向下修正
- 稳健 MILP 为主方案
- 若稳健 MILP 在中心预测场景下相对规则的预计优势不足 20%，当天回退规则

输出：

- `results/robust_dispatch_customer_1.json`
- `results/robust_dispatch_daily_customer_1.csv`
- `results/robust_dispatch_customer_1.png`

当前 60% 分位测试结果：

- 规则：160.79 AUD
- 点预测 MILP：167.43 AUD
- 稳健 MILP：168.43 AUD
- 稳健 + 20% 回退：160.32 AUD
- 完美预知：143.43 AUD

安全层只在 5 天启用稳健 MILP，其余 41 天回退规则。收益改善有限，但显著避免了确定型 MILP 的损失。

当前测试日的敏感性结论：

- 退化成本 0-0.08 AUD/kWh：放电量约 8.55 kWh
- 0.10 AUD/kWh：放电量降至约 7.60 kWh
- 0.30 AUD/kWh：放电量降至约 0.95 kWh
- 0.50 AUD/kWh 及以上：电池基本退出调度

日末 SOC 使用“不低于初始 SOC”的约束，并加入 `1e-6` 的数值安全裕量，避免求解器容差造成轻微越界。

## 收尾标准

### 三基准表

Customer 1 的 60 天 walk-forward 实际结算：

| 策略 | 实际结算总成本 | 相对完美预知 |
|---|---:|---:|
| 规则调度 | 209.24 AUD | +21.06 AUD |
| 预测驱动 MILP | 219.78 AUD | +31.60 AUD |
| 完美预知 MILP | 188.18 AUD | 0 |

结论：

- 预测驱动 MILP 没有获胜，甚至比规则调度差 10.54 AUD。
- 预测误差的实际 regret 为 31.60 AUD。
- 理论上限捕获率为 -50.04%，表示预测驱动 MILP 不仅未捕获规则到完美预知的收益，还产生了额外损失。
- 该负面结论保留，不通过修改评价口径美化结果。

### 分位数稳健实验

已跑过一次轻量分位数稳健实验：

- 过去 14 天滚动误差校准
- 16:00-19:00 负荷使用 60% 分位低估误差
- PV 使用 60% 分位高估误差
- 预测收益不足规则 20% 时回退规则

14 天校准后的 46 天结果：

| 策略 | 实际结算总成本 |
|---|---:|
| 规则调度 | 160.79 AUD |
| 点预测 MILP | 167.43 AUD |
| 稳健 MILP | 168.43 AUD |
| 稳健 MILP + 规则回退 | 160.32 AUD |
| 完美预知 MILP | 143.43 AUD |

结论：

- 规则安全层有效阻止了确定型 MILP 的明显损失。
- 稳健加回退只在 5 天使用 MILP，其余 41 天回退规则。
- 它相对始终规则仅改善 0.47 AUD，距离完美预知仍差 16.85 AUD。
- 规则回退是安全层，不是主方案；当前稳健 MILP 还不能单独承担主策略。

### 最终 schedule schema

`schedule.parquet` 的 schema 固定为 `1.0`，后续不再修改列名或单位。

```text
customer_id            int32
interval_start         datetime64[ns]
schedule_date          date32
strategy               string: robust_milp | rule_fallback
is_fallback            bool
load_kw                float64
pv_kw                  float64
buy_price_per_kwh      float64, AUD/kWh
sell_price_per_kwh     float64, AUD/kWh
charge_kw              float64, AC-side
discharge_kw           float64, AC-side
battery_net_ac_kw      float64, discharge positive
soc_start              float64, fraction 0-1
soc_end                float64, fraction 0-1
grid_net_kw            float64, import positive
grid_import_kw         float64
grid_export_kw         float64
grid_import_kwh        float64
grid_export_kwh        float64
import_cost            float64, AUD
export_revenue         float64, AUD
net_cost               float64, AUD
degradation_cost       float64, AUD
total_cost             float64, AUD
```

主键：

```text
customer_id + interval_start
```

时间约定：

- 行粒度为每户每 30 分钟一行。
- `interval_start` 保存 Australia/Sydney 本地墙钟时间，不做 DST 重定位。
- 功率为 AC 侧 kW，电量为 kWh。
- 调度文件保存最终选中策略，不混放三个候选策略。
