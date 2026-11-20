from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = PROJECT_ROOT / "AI助手"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from api import create_app  # noqa: E402
from tool_service import ToolService  # noqa: E402

from database import RequestLogDatabase  # noqa: E402


logger = logging.getLogger(__name__)
DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "requests.sqlite3"
MAX_CAPTURED_BODY_BYTES = 64 * 1024


class RequestLogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    request_id: str
    timestamp_utc: datetime
    method: str
    path: str
    status_code: int
    latency_ms: float
    client_host: str | None
    error_code: str | None


class RequestLogPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[RequestLogEntry]
    count: int = Field(ge=0)


def _extract_error_code(
    response_headers: list[tuple[bytes, bytes]],
    body: bytes,
) -> str | None:
    headers = {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in response_headers
    }
    content_type = headers.get("content-type", "")
    if "json" not in content_type or not body:
        return None
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    error_code = data.get("error_code") if isinstance(data, dict) else None
    return error_code if isinstance(error_code, str) and error_code else None


class RequestLogMiddleware:
    def __init__(self, app: ASGIApp, database: RequestLogDatabase) -> None:
        self.app = app
        self.database = database

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        started_at = time.perf_counter()
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        client = scope.get("client")
        client_host = client[0] if client else None
        status_code = 500
        latency_ms = 0.0
        response_headers: list[tuple[bytes, bytes]] = []
        captured_body = bytearray()

        async def send_with_logging(message: Message) -> None:
            nonlocal status_code, latency_ms, response_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                latency_ms = round(
                    (time.perf_counter() - started_at) * 1000,
                    3,
                )
                response_headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower()
                    not in {b"x-request-id", b"x-process-time-ms"}
                ]
                response_headers.extend(
                    [
                        (b"x-request-id", request_id.encode("ascii")),
                        (
                            b"x-process-time-ms",
                            str(latency_ms).encode("ascii"),
                        ),
                    ]
                )
                message["headers"] = response_headers
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                remaining = MAX_CAPTURED_BODY_BYTES - len(captured_body)
                if remaining > 0:
                    captured_body.extend(body[:remaining])
            await send(message)

        try:
            await self.app(scope, receive, send_with_logging)
        finally:
            if latency_ms == 0.0:
                latency_ms = round(
                    (time.perf_counter() - started_at) * 1000,
                    3,
                )
            error_code = _extract_error_code(
                response_headers,
                bytes(captured_body),
            )
            try:
                self.database.insert_request_log(
                    request_id=request_id,
                    timestamp_utc=timestamp_utc,
                    method=scope.get("method", ""),
                    path=scope.get("path", ""),
                    status_code=status_code,
                    latency_ms=latency_ms,
                    client_host=client_host,
                    error_code=error_code,
                )
            except Exception:
                logger.exception("Failed to write request log")


def create_service_app(
    *,
    database_path: str | Path | None = None,
    service: ToolService | None = None,
) -> FastAPI:
    database = RequestLogDatabase(database_path or DEFAULT_DATABASE_PATH)
    app = create_app(service)
    app.title = "家庭光储荷 AI 助手服务 API"
    app.version = "1.1.0"
    app.description = "只读工具 API，带 SQLite 请求日志。"
    app.state.request_log_database = database
    app.add_middleware(RequestLogMiddleware, database=database)

    @app.get(
        "/service/request-logs",
        response_model=RequestLogPage,
        tags=["service"],
    )
    def list_request_logs(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        status_code: Annotated[int | None, Query(ge=100, le=599)] = None,
        error_code: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
        path: Annotated[str | None, Query(min_length=1, max_length=512)] = None,
    ) -> RequestLogPage:
        rows = database.list_request_logs(
            limit=limit,
            status_code=status_code,
            error_code=error_code,
            path=path,
        )
        items = [RequestLogEntry.model_validate(row) for row in rows]
        return RequestLogPage(items=items, count=len(items))

    @app.get("/service/health", tags=["service"])
    def service_health() -> dict[str, str]:
        return {
            "status": "ok" if database.ping() else "error",
            "database": str(database.path),
        }

    return app


app = create_service_app()
