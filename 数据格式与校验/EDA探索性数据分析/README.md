# Ausgrid EDA 探索性数据分析

本目录只包含 EDA 相关代码、测试、图表和聚类结果。数据读取、清洗、质量筛查和年度批处理仍保留在上一级目录。

## 运行方式

```powershell
Set-Location "数据格式与校验/EDA探索性数据分析"
python eda_load_curve.py
python eda_peer_comparison.py
python eda_user_clustering.py
```

脚本会自动把上一级目录加入导入路径，因此可以直接调用 `reader.py` 和 `pipeline.py`。

## 已包含内容

### 单日与平均日负荷

- `eda_load_curve.py`
- `test_eda_load_curve.py`

输出：

- `reports/figures/customer_1_2010-07-01_daily_load_curve.png`
- `reports/figures/customer_1_2010-07-01_daily_load_metrics.json`
- `reports/figures/customer_1_2010-2011_average_daily_load_profile.png`
- `reports/figures/customer_1_2010-2011_average_daily_load_metrics.json`
- `reports/figures/customer_1_2010-2011_seasonal_weekday_load_profile.png`
- `reports/figures/customer_1_2010-2011_seasonal_weekday_metrics.json`
- `reports/figures/customer_1_2010-2011_seasonal_weekday_gc_profile.png`
- `reports/figures/customer_1_2010-2011_seasonal_weekday_gc_metrics.json`

### 单户与 300 户总体比较

- `eda_peer_comparison.py`
- `test_eda_peer_comparison.py`

输出：

- `reports/figures/customer_1_vs_300_population_gc_weekday_profile.png`
- `reports/figures/customer_1_vs_300_population_gc_weekday_metrics.json`

### 300 户 GC 曲线聚类

- `eda_user_clustering.py`
- `test_eda_user_clustering.py`

输出：

- `reports/customer_gc_weekday_profiles.csv`
- `reports/customer_gc_weekday_clusters.csv`
- `reports/customer_gc_weekday_cluster_summary.json`
- `reports/figures/customer_gc_weekday_clusters.png`

## 冒烟测试

```powershell
Set-Location "数据格式与校验/EDA探索性数据分析"
python -c "import eda_load_curve, eda_peer_comparison, eda_user_clustering; print('EDA imports OK')"
```
