# 边缘部署与工程化

这一阶段只保留四项交付：

1. Pytest 轻量测试。
2. GitHub Actions 自动测试与 TFLite 冒烟。
3. 小模型 TFLite 服务及 Dockerfile。
4. Keras、TFLite Float32、TFLite 动态量化之间的体积、误差、推理耗时对比。

## 模型边界

小模型不是主 LightGBM 的替代品，而是一个可部署样例：

```text
日内时段 + 星期 + 周末 + 昨日同时刻 + 上周同时刻
→ 未来半小时负荷 kWh
```

训练数据来自：

```text
时序预测/results/validation_300_households_predictions.csv
```

模型结构是 `7 → 16 → 8 → 1` 的小型 MLP，参数量约 273 个，适合验证 TFLite 工程链路。

## 1. Pytest

```powershell
Set-Location "边缘部署与工程化"
python -m pytest -q -m "not tflite"
```

TFLite 产物生成后，再运行：

```powershell
python -m pytest -q -m tflite
```

## 2. TFLite 转换与基准

Windows 主环境是 Python 3.14，TensorFlow 当前不支持该版本。使用独立 Python 3.11 环境：

```powershell
uv venv "C:\Users\Y\AppData\Local\Temp\edge-tflite-311" --python 3.11 --seed
C:\Users\Y\AppData\Local\Temp\edge-tflite-311\Scripts\python.exe -m pip install `
  -i https://pypi.tuna.tsinghua.edu.cn/simple `
  -r requirements-tflite.txt
C:\Users\Y\AppData\Local\Temp\edge-tflite-311\Scripts\python.exe `
  tflite_benchmark.py
```

不依赖真实数据器的 CI 合成冒烟：

```powershell
python tflite_benchmark.py --synthetic --epochs 2 --max-train-rows 5000 --repeats 50 --output-dir ci_results
```

输出：

- `results/edge_benchmark.json`
- `results/edge_benchmark.md`
- `artifacts/load_mlp.keras`
- `artifacts/load_mlp_float32.tflite`
- `artifacts/load_mlp_dynamic_range.tflite`

延迟是当前 CPU、batch size 1、单线程的本机基线，不代表手机或嵌入式板卡结果。

当前小模型的动态量化版本与 Float32 版本体积和误差相同，FlatBuffer
元数据开销占主导，因此服务默认使用 `load_mlp_float32.tflite`，动态量化结果
仅保留作工程对照。

## 3. 本地运行 TFLite 服务

容器运行时只安装 LiteRT，不需要 TensorFlow。本机若安装了 LiteRT/TensorFlow：

```powershell
python -m uvicorn edge_service:create_app --factory --host 127.0.0.1 --port 8770
```

接口：

```text
http://127.0.0.1:8770/docs
```

文档页使用随项目分发的本地 Swagger UI，不依赖外部 CDN；接口定义仍由
`/openapi.json` 提供。

## 4. Docker

先完成 TFLite 转换，确保 `artifacts/load_mlp_float32.tflite` 存在：

```powershell
docker build -t home-energy-edge:latest .
docker run --rm -p 8000:8000 home-energy-edge:latest
```

Dockerfile 默认使用清华 PyPI 镜像。需要改源时可以覆盖：

```powershell
docker build `
  --build-arg PIP_INDEX_URL=https://pypi.org/simple `
  -t home-energy-edge:latest .
```

## GitHub Actions

工作流位于：

```text
.github/workflows/edge-ci.yml
```

包含两个任务：

- Python 3.12：运行不依赖 TensorFlow 的单元测试。
- Python 3.11：安装 TensorFlow，使用合成数据完成训练、转换、基准和 TFLite 制品冒烟。

当前项目目录还不是 Git 仓库，因此工作流文件已准备好，但只有初始化 Git、提交并推送到 GitHub 后才会实际触发。
