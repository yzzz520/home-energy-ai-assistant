# 时序预测

## 当前任务

- 目标：Customer 1 的总负荷 `total_load_kwh`
- 粒度：30 分钟
- 预测方式：每天 00:00 生成未来 48 个半小时点
- 测试区间：2010-2011 年最后 60 天
- 基线：
  - 昨日同时刻：滞后 48 步
  - 上周同时刻：滞后 336 步

## 运行

```powershell
Set-Location "时序预测"
python seasonal_naive.py
python -m unittest -v test_seasonal_naive.py
```

## 输出

- `results/customer_1_total_load_seasonal_naive_forecasts.csv`
- `results/customer_1_total_load_seasonal_naive_metrics.json`
- `results/customer_1_total_load_seasonal_naive_example.png`

## 300 户全局 Walk-forward 基线

```powershell
Set-Location "时序预测"
python run_global_baseline.py
```

验证方式：

- 每个客户每天 00:00 生成未来 48 步预测
- 连续评测量后 60 天
- 不使用随机切分
- 300 户共 864,000 个评估点

输出：

- `results/global_300_total_load_seasonal_naive_metrics.json`
- `results/global_300_total_load_seasonal_naive_per_customer.csv`
- `results/global_300_total_load_seasonal_naive_horizon_metrics.csv`
- `results/global_300_total_load_seasonal_naive_horizon_error.png`

## LightGBM Walk-forward

```powershell
Set-Location "时序预测"
python lightgbm_walk_forward.py
python -m unittest -v test_lightgbm_walk_forward.py
```

当前先验证 Customer 1：

- 目标：`total_load_kwh`
- 3 折扩展窗口 walk-forward
- 每折测试 20 天
- 每折重新训练 LightGBM
- 每折最后 14 天作为时间验证集和 early stopping
- 与昨日同时刻、上周同时刻基线在同一测试段比较

特征只使用预测时已知的信息：

- 日历周期、周末标记
- 48 步前和 336 步前的滞后值
- 过去 1-7 天同一先验滞后窗口的统计量

## 全局 LightGBM 小步路线

全局模型逻辑写在 `lightgbm_global.py`，但不一次直接跑 300 户。

1. 冒烟：10 户 × 1 折
2. 修正类别编码：Customer、Postcode、cluster_id 使用 category
3. 小规模 walk-forward：10 户 × 3 折
4. 中规模 walk-forward：100 户 × 3 折
5. 正式验证：300 户 × 3 折

当前已经完成 300 户 × 3 折全局 Walk-forward。家庭级特征包括：

- `Customer`：客户类别
- `Generator Capacity`：光伏装机容量
- `Postcode`：邮编类别
- `cluster_id`：用户 GC 工作日曲线聚类标签

轻量冒烟测试不会重跑 300 户全量，只用前 8 户训练 10 轮，并检查正式 300 户产物：

```powershell
Set-Location "时序预测"
python -m unittest -v test_lightgbm_global_smoke.py
```

执行指定户数 × 3 折：

```powershell
Set-Location "时序预测"
python validate_households.py --households 30
```

输出：

- `results/validation_10_households_lightgbm_metrics.json`
- `results/validation_10_households_per_customer.csv`
- `results/validation_10_households_predictions.csv`
- `results/validation_10_households_importance.csv`
- `results/validation_10_households_overall_metrics.png`
- `results/validation_10_households_fold_mae.png`

### 迭代上限与低估偏差实验

```powershell
Set-Location "时序预测"
python experiment_30_bias_rounds.py
```

固定 30 户 × 3 折，比较：

- L1 目标，最大 800 轮
- L1 + 验证集偏差修正
- L2 目标，最大 800 轮

输出：

- `results/experiment_30_households_rounds_bias.json`
- `results/experiment_30_households_rounds_bias.png`
- `results/experiment_30_households_rounds_bias_predictions.csv`

独立检查 1200 轮上限：

```powershell
python experiment_30_l1_1200.py
```

输出：

- `results/experiment_30_households_l1_1200.json`

实验结论：

- L1 800 轮：总体 MAE 0.2189 kWh，偏差 -0.0950 kWh
- L1 1200 轮：总体 MAE 0.2201 kWh，偏差 -0.0989 kWh
- L1 + 验证集偏差修正：总体 MAE 0.2456 kWh
- L2 800 轮：总体 MAE 0.2453 kWh，偏差接近 0

因此进入 100 户时保留 L1、最大 800 轮。L1 的低估来自中位数目标本身；若后续调度更重视成本偏差，再单独评估 L2 或成本加权校准，不直接改动主预测基线。

## 300 户最终 Walk-forward

```powershell
python validate_households.py --households 300
```

当前最终结果：

- 评估点：864,000
- LightGBM MAE：0.2084 kWh
- 昨日同时刻 MAE：0.2642 kWh
- LightGBM WAPE：46.30%
- 297/300 户优于昨日同时刻
- 特征构建约 211 秒
- 训练与预测约 374 秒
- 完整脚本约 586 秒
- 运行中观测内存约 2.3 GB

输出前缀：

- `results/validation_300_households_*`

## 指标

- MAE：平均绝对误差，单位 kWh
- RMSE：均方根误差，单位 kWh
- WAPE：绝对误差总和除以实际值总和
- Mean Bias：预测减实际的平均偏差
