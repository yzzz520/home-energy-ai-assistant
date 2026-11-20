# 第八步：结果 Demo 与验收

## 一条命令

```powershell
python build_acceptance.py --open
```

命令只读取前面阶段已经生成的 JSON/CSV，不重新训练预测模型，也不重新运行 MILP。

生成：

- `results/three_strategy_comparison.csv`
- `results/forecast_comparison.csv`
- `results/assistant_evaluation.csv`
- `results/acceptance_report.html`
- `results/acceptance_summary.json`

HTML 为完全离线页面，不依赖 Swagger、CDN 或外部字体。

## 验收结论

- 预测：300 户全局 LightGBM 的 MAE 优于昨日和上周同时刻朴素基线。
- 调度：预测驱动 MILP 的实际结算成本高于规则调度，负面结果原样保留。
- 助手：22 条手写种子评测全部通过，但只证明端到端闭环，不代表泛化准确率。

## 测试

```powershell
python -m pytest -q
```
