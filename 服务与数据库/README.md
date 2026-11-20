# 服务与数据库

这一阶段只做两件事：

1. 复用 `AI助手/api.py` 中已经冻结的只读工具接口。
2. 用一个 SQLite 数据库记录每个 HTTP 请求的元数据。

不修改预测、调度结果、工具契约或 `schedule.parquet`。

## 文件职责

- `database.py`：创建 `request_logs` 表，写入和查询请求日志。
- `service_api.py`：包装现有 FastAPI 应用，加入日志中间件和日志查询接口。
- `test_service_api.py`：成功请求、校验失败、工具错误、数据库重开和文档页测试。
- `data/requests.sqlite3`：运行后自动生成，不在测试中复用真实日志。

## 日志字段

| 字段 | 含义 |
| --- | --- |
| `request_id` | 服务端生成的请求 UUID |
| `timestamp_utc` | UTC 请求开始时间 |
| `method` | HTTP 方法 |
| `path` | 请求路径，不保存查询参数和请求正文 |
| `status_code` | HTTP 状态码 |
| `latency_ms` | 从进入中间件到响应生成的耗时 |
| `client_host` | 客户端主机 |
| `error_code` | 工具契约错误码，例如 `MISSING_PARAMETER` |

日志不保存请求正文，避免把家庭数据、查询文本或其他敏感内容写入数据库。

## 启动

```powershell
Set-Location "C:\Users\Y\Desktop\家庭光储荷预测与储能调度 AI 助手\服务与数据库"
python -m uvicorn service_api:app --host 127.0.0.1 --port 8766
```

打开：

```text
http://127.0.0.1:8766/docs
```

原有服务端口 `8765` 不受影响。

## 验证

```powershell
python -m pytest -q test_service_api.py
```

冒烟检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8766/service/health
Invoke-RestMethod "http://127.0.0.1:8766/service/request-logs?limit=5"
```

## 边界

- 这是开发阶段的单机 SQLite 日志，不是生产级审计系统。
- 日志查询接口目前没有鉴权，公开部署前必须加管理员鉴权。
- 当前不保存请求正文、响应正文、API 密钥或用户上传文件。
