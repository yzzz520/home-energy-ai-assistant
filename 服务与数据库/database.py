from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class RequestLogDatabase:
    """Small SQLite repository for API request metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    timestamp_utc TEXT NOT NULL,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    client_host TEXT,
                    error_code TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp
                    ON request_logs(timestamp_utc DESC);

                CREATE INDEX IF NOT EXISTS idx_request_logs_path
                    ON request_logs(path, id DESC);

                CREATE INDEX IF NOT EXISTS idx_request_logs_error_code
                    ON request_logs(error_code, id DESC);
                """
            )

    def insert_request_log(
        self,
        *,
        request_id: str,
        timestamp_utc: str,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        client_host: str | None,
        error_code: str | None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO request_logs (
                    request_id,
                    timestamp_utc,
                    method,
                    path,
                    status_code,
                    latency_ms,
                    client_host,
                    error_code
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    timestamp_utc,
                    method,
                    path,
                    status_code,
                    latency_ms,
                    client_host,
                    error_code,
                ),
            )

    def list_request_logs(
        self,
        *,
        limit: int = 50,
        status_code: int | None = None,
        error_code: str | None = None,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        parameters: list[Any] = []

        if status_code is not None:
            where.append("status_code = ?")
            parameters.append(status_code)
        if error_code is not None:
            where.append("error_code = ?")
            parameters.append(error_code)
        if path is not None:
            where.append("path = ?")
            parameters.append(path)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        parameters.append(limit)

        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    request_id,
                    timestamp_utc,
                    method,
                    path,
                    status_code,
                    latency_ms,
                    client_host,
                    error_code
                FROM request_logs
                {where_sql}
                ORDER BY id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def ping(self) -> bool:
        with self.connection() as connection:
            row = connection.execute("SELECT 1 AS ok").fetchone()
        return bool(row and row["ok"] == 1)
