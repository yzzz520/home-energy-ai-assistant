# 家庭能源 AI 助手

## 当前阶段

当前只完成“助手设计与数据契约”，尚未接入 LLM，也未实现工具函数。

已冻结的第一版契约：

- 用户范围：已有 300 户、冷启动用户、超出范围
- 意图体系：`contracts/intents.json`
- 工具输入输出：`contracts/tools.json`
- 评测集结构：`contracts/evaluation_case.schema.json`
- 中文评测种子：`data/eval_seed.csv`
- 设计与边界：`docs/assistant_design.md`

## 设计原则

- 先判断用户范围，再判断用户意图。
- 已有 300 户是正式支持范围。
- 新用户未上传数据时只提供范围说明和上传要求。
- 新用户上传合规数据后，只能获得实验性冷启动分析。
- LLM 不能直接读取 CSV、数据库或文件系统，只能调用工具。
- 预测、电价、SOC、成本和调度结果必须来自工具。
- 规则回退只作为安全层，不能伪装成主优化策略。
- 助手不能直接控制真实设备。

## 第一版不实现

- 意图分类器
- LLM 接入
- RAG 向量库
- 工具执行函数
- 数据上传接口
- FastAPI 服务

本目录当前只冻结边界、接口和评测口径。

## 范围解析与契约校验

第一版运行时代码：

- `scope.py`：已有客户、冷启动和超范围判定
- `contract_validator.py`：意图、槽位、身份要求和工具权限校验
- `test_contracts.py`：范围与契约测试

运行：

```powershell
Set-Location "AI助手"
python -m unittest -v test_contracts.py
```

## 规则版意图与槽位基线

```powershell
Set-Location "AI助手"
python rule_baseline.py
python -m unittest -v test_rule_baseline.py
```

输出：

- `results/rule_baseline_predictions.csv`
- `results/rule_baseline_metrics.json`

该基线使用关键词和正则，不调用 LLM。评测指标包括意图准确率、槽位精确匹配率和槽位 micro-F1。

当前 22 条种子结果：

```text
意图准确率：100%
槽位精确匹配率：100%
槽位 micro-F1：100%
```

结论：

- 22 条样本规模过小，不能据此认为规则基线已经具备真实泛化能力。
- 当前封闭意图体系下，不需要立刻用 LLM 替换规则分类器。
- 在评测集扩展到 100-200 条并覆盖同义表达、多意图、缺槽位和口语噪声后，再决定是否引入 LLM。
- LLM 更适合作为低置信度或未知意图的兜底分类器，不应绕过用户范围和工具契约。

## 只读工具 API

工具实现位于：

- `models.py`：冻结 Pydantic 输入、输出和错误模型
- `tool_service.py`：只读 CSV/Parquet 和知识文件
- `api.py`：FastAPI 路由
- `test_tools.py`：Pytest 契约、错误码和数字一致性测试

启动：

```powershell
Set-Location "AI助手"
python -m uvicorn api:app --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765/docs
```

工具不会重新训练预测模型或重新运行 MILP。`get_forecast` 读取已保存的 LightGBM walk-forward 输出，`get_schedule`、`get_tariff`、`explain_schedule` 和 `get_energy_summary` 读取冻结的 `schedule.parquet`。

当前只实现 LightGBM 负荷预测。PV、净负荷预测和 `simulate_scenario` 会按契约返回未支持错误，不会伪造结果。

冻结错误码：

```text
NO_DATA
DATE_OUT_OF_RANGE
MISSING_PARAMETER
OUT_OF_SCOPE
UNSUPPORTED_TARGET
TOOL_NOT_IMPLEMENTED
```

Pytest：

```powershell
python -m pytest -q test_tools.py
```

当前工具测试共 21 个，覆盖 Pydantic 模型、FastAPI `/docs`、参数校验、所有错误码和 Parquet/CSV 数字一致性。

## DeepSeek 少样本意图识别

实现：

- `llm_intent.py`
- `data/few_shot_examples.json`

配置：

```powershell
$env:DEEPSEEK_API_KEY="你的本地密钥"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

也可以把同样的变量写入本地 `AI助手/.env`。不要提交真实密钥。

运行 22 条种子评测：

```powershell
Set-Location "AI助手"
python llm_intent.py
```

输出：

- `results/llm_intent_predictions.csv`
- `results/llm_intent_metrics.json`

LLM 输出必须经过严格 Pydantic JSON 和冻结意图/槽位契约校验。当前进程尚未检测到 `DEEPSEEK_API_KEY`，因此尚未生成真实 LLM 准确率。

## 无 API 确定性端到端闭环

```powershell
Set-Location "AI助手"
python assistant_pipeline.py
python -m unittest -v test_assistant_pipeline.py
```

流程：

```text
规则意图/槽位
→ 用户范围
→ 契约校验
→ 工具计划
→ 只读工具调用
→ 模板回答
→ 数字一致性检查
```

输出：

- `results/pipeline_predictions.csv`
- `results/pipeline_metrics.json`

该路径不需要任何外部 API。

当前 22 条无 API 端到端结果：

```text
意图准确率：100%
槽位精确匹配率：100%
结果类型准确率：100%
工具计划匹配率：100%
数字一致率：100%
```

注意：22 条种子是手工围绕契约设计的小样本，且规则基线与种子同源。该结果用于验证闭环、错误处理和数字忠实度，不能作为泛化能力结论。

扩大到 100 条时应：

- 保留手写测试集，不把合成数据混入最终手写指标。
- 新增口语化、错别字、多意图、歧义和缺槽位变体，并标记 `synthetic`。
- 不用同一个测试集反向调整 Prompt 或规则后再报告原指标。
- 按失败类型标注：意图错误、槽位错误、工具选择错误、工具执行失败、回答数值错误。
