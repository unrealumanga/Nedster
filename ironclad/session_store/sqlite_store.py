"""SQLite-backed session store (v4 contract: ironclad.contracts.SessionStore).

One connection per instance, guarded by a threading.Lock; every write runs inside
``BEGIN IMMEDIATE`` so concurrent processes/instances never fork per-session
counters. The file carries a ``schema_version`` row and forward migrations.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ironclad.contracts import ChatMessage, ToolCall

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project_root TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            created_ns INTEGER NOT NULL,
            updated_ns INTEGER NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_sessions_root ON sessions(project_root, updated_ns DESC)",
        """
        CREATE TABLE IF NOT EXISTS messages (
            session_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tool_calls_json TEXT NOT NULL DEFAULT '[]',
            tool_call_id TEXT,
            name TEXT,
            thinking TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (session_id, seq)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS outputs (
            session_id TEXT NOT NULL,
            handle INTEGER NOT NULL,
            tool TEXT NOT NULL,
            call_id TEXT NOT NULL DEFAULT '',
            full_text TEXT NOT NULL DEFAULT '',
            created_ns INTEGER NOT NULL,
            PRIMARY KEY (session_id, handle)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS state (
            session_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            PRIMARY KEY (session_id, key)
        )
        """,
    ),
}


def normalize_project_root(project_root: str) -> str:
    return os.path.normcase(os.path.realpath(project_root))


def _serialize_tool_calls(tool_calls: list[ToolCall]) -> str:
    payload = [
        {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
        for call in tool_calls
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deserialize_tool_calls(raw: str) -> list[ToolCall]:
    return [
        ToolCall(id=item["id"], name=item["name"], arguments=item.get("arguments") or {})
        for item in json.loads(raw)
    ]


class SqliteSessionStore:
    """SessionStore implementation over a single SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SqliteSessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- schema -------------------------------------------------------------

    def schema_version(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT version FROM schema_version").fetchone()
        return int(row[0]) if row else 0

    def _migrate(self) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
                )
                row = self._connection.execute("SELECT version FROM schema_version").fetchone()
                current = int(row[0]) if row else 0
                for version in range(current + 1, SCHEMA_VERSION + 1):
                    for statement in _MIGRATIONS[version]:
                        self._connection.execute(statement)
                self._connection.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
                if row is None:
                    self._connection.execute(
                        "INSERT INTO schema_version(version) VALUES(?)", (SCHEMA_VERSION,)
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def _write(self, statements: Sequence[tuple[str, tuple[Any, ...]]]) -> None:
        """Run ``statements`` atomically inside BEGIN IMMEDIATE."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for sql, params in statements:
                    self._connection.execute(sql, params)
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    # -- sessions -----------------------------------------------------------

    def create(self, project_root: str, title: str = "") -> str:
        session_id = uuid.uuid4().hex
        now = time.time_ns()
        root = normalize_project_root(project_root)
        self._write(
            [
                (
                    "INSERT INTO sessions(id, project_root, title, created_ns, updated_ns) VALUES(?,?,?,?,?)",
                    (session_id, root, title, now, now),
                )
            ]
        )
        return session_id

    def list_sessions(self, project_root: str, limit: int = 20) -> list[dict[str, Any]]:
        root = normalize_project_root(project_root)
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, project_root, title, created_ns, updated_ns FROM sessions"
                " WHERE project_root = ? ORDER BY updated_ns DESC, id DESC LIMIT ?",
                (root, limit),
            ).fetchall()
        return [
            {
                "id": row[0],
                "project_root": row[1],
                "title": row[2],
                "created_ns": row[3],
                "updated_ns": row[4],
            }
            for row in rows
        ]

    def latest_session(self, project_root: str) -> str | None:
        root = normalize_project_root(project_root)
        with self._lock:
            row = self._connection.execute(
                "SELECT id FROM sessions WHERE project_root = ?"
                " ORDER BY updated_ns DESC, id DESC LIMIT 1",
                (root,),
            ).fetchone()
        return str(row[0]) if row else None

    def _touch(self, session_id: str, now: int) -> None:
        self._connection.execute(
            "UPDATE sessions SET updated_ns = ? WHERE id = ?", (now, session_id)
        )

    # -- messages -----------------------------------------------------------

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        tool_calls_json = _serialize_tool_calls(message.tool_calls)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                seq = int(row[0]) + 1
                self._connection.execute(
                    "INSERT INTO messages(session_id, seq, role, content, tool_calls_json,"
                    " tool_call_id, name, thinking) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        session_id,
                        seq,
                        message.role,
                        message.content,
                        tool_calls_json,
                        message.tool_call_id,
                        message.name,
                        message.thinking,
                    ),
                )
                self._touch(session_id, time.time_ns())
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def messages(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT role, content, tool_calls_json, tool_call_id, name, thinking"
                " FROM messages WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [
            ChatMessage(
                role=row[0],
                content=row[1],
                tool_calls=_deserialize_tool_calls(row[2]),
                tool_call_id=row[3],
                name=row[4],
                thinking=row[5],
            )
            for row in rows
        ]

    def replace_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        """Atomically swap the whole history (compaction). On any failure the old
        messages remain untouched."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "DELETE FROM messages WHERE session_id = ?", (session_id,)
                )
                for index, message in enumerate(messages, start=1):
                    self._connection.execute(
                        "INSERT INTO messages(session_id, seq, role, content, tool_calls_json,"
                        " tool_call_id, name, thinking) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            session_id,
                            index,
                            message.role,
                            message.content,
                            _serialize_tool_calls(message.tool_calls),
                            message.tool_call_id,
                            message.name,
                            message.thinking,
                        ),
                    )
                self._touch(session_id, time.time_ns())
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    # -- outputs (observation masking) --------------------------------------

    def put_output(self, session_id: str, tool: str, call_id: str, text: str) -> int:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(handle), 0) FROM outputs WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                handle = int(row[0]) + 1
                self._connection.execute(
                    "INSERT INTO outputs(session_id, handle, tool, call_id, full_text, created_ns)"
                    " VALUES(?,?,?,?,?,?)",
                    (session_id, handle, tool, call_id, text, time.time_ns()),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return handle

    def get_output(self, session_id: str, handle: int) -> str:
        """Full text stored under ``handle``; ``KeyError`` when there is none (never a silent empty string)."""
        with self._lock:
            row = self._connection.execute(
                "SELECT full_text FROM outputs WHERE session_id = ? AND handle = ?",
                (session_id, handle),
            ).fetchone()
        if row is None:
            raise KeyError(f"no output {handle} in session {session_id}")
        return str(row[0])

    # -- state --------------------------------------------------------------

    def set_state(self, session_id: str, key: str, value: Any) -> None:
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._write(
            [
                (
                    "INSERT INTO state(session_id, key, value_json) VALUES(?,?,?)"
                    " ON CONFLICT(session_id, key) DO UPDATE SET value_json = excluded.value_json",
                    (session_id, key, rendered),
                )
            ]
        )

    def get_state(self, session_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._connection.execute(
                "SELECT value_json FROM state WHERE session_id = ? AND key = ?",
                (session_id, key),
            ).fetchone()
        return json.loads(row[0]) if row else default
