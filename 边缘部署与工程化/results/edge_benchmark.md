# Edge Model Benchmark

- Data source: `C:\Users\Y\Desktop\家庭光储荷预测与储能调度 AI 助手\时序预测\results\validation_300_households_predictions.csv`
- Train rows used: 250,000
- Test rows: 288,000
- Parameters: 273

| Variant | Size KB | MAE vs actual | MAE vs Keras | Median latency ms | P95 latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| keras | 28.35 | 0.252248 | 0.00000000 | 0.3890 | 0.5520 |
| tflite_float32 | 2.95 | 0.252248 | 0.00000004 | 0.0020 | 0.0022 |
| tflite_dynamic_range | 2.95 | 0.252248 | 0.00000004 | 0.0021 | 0.0027 |

Latency is measured on the current CPU with batch size 1 and one thread.
It is a local engineering baseline, not a phone or embedded-device benchmark.
