"""SQLite-backed session store (v4 contract: ironclad.contracts.SessionStore).

Three kinds of durable data live side by side and never mix:

* ``messages``   — the *model* history (masked/compacted by ``replace_messages``).
* ``transcript`` — the canonical *display* transcript (append-only, never compacted).
* ``outputs``/``state``/``session_locks`` — handle vault, task state and activity locks.

One connection per instance, guarded by a threading.Lock; every write runs inside
``BEGIN IMMEDIATE`` so concurrent processes/instances never fork per-session
counters. The file carries a ``schema_version`` row and forward migrations.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ironclad.contracts import ChatMessage, ToolCall

SCHEMA_VERSION = 2

#: A session counts as *active* while its lock heartbeat is younger than this.
LOCK_TTL_S = 60.0

#: Synthetic result text for an assistant call that never produced one.
#: Kept byte-identical to ``ironclad.engine.loop.INTERRUPTED``.
INTERRUPTED_TEXT = "[cancelled] not completed: session was interrupted"

_HANDLE_PATTERNS = tuple(re.compile(p) for p in (r"handle=(\d+)", r'"handle":\s*(\d+)', r"#(\d+)"))
_HANDLE_LIST_PATTERN = re.compile(r'"handles"\s*:\s*\[([^\]]*)\]')  # SummaryData.handles in task state
_HANDLE_RANGE_PATTERN = re.compile(r"#(\d+)\s*-\s*#(\d+)")  # "_ranges()" summary spans: #4-#6 keeps 4,5,6
_HANDLE_RANGE_MAX = 100_000

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
    2: (
        """
        CREATE TABLE IF NOT EXISTS transcript (
            session_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL DEFAULT '',
            meta_json TEXT NOT NULL DEFAULT '{}',
            created_ns INTEGER NOT NULL,
            PRIMARY KEY (session_id, seq)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS session_locks (
            session_id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            acquired_ns INTEGER NOT NULL,
            heartbeat_ns INTEGER NOT NULL
        )
        """,
    ),
}


def normalize_project_root(project_root: str) -> str:
    """Canonical binding key for a project: resolved (symlinks, relative parts) and
    case-normalised on Windows, so ``C:/Proj`` and ``c:\\proj`` are one project."""
    return os.path.normcase(os.path.realpath(project_root))


def make_owner(label: str = "") -> str:
    """Opaque lock owner identity: ``host:pid:label`` (+ a nonce) so a crashed holder can be
    detected while a second host stays conservative (heartbeat only)."""
    host = socket.gethostname() or "unknown"
    return f"{host}:{os.getpid()}:{label or 'nedster'}:{uuid.uuid4().hex[:6]}"


def _owner_pid(owner: str) -> int | None:
    parts = owner.split(":")
    if len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1])
    return None


def _owner_host(owner: str) -> str:
    parts = owner.split(":", 1)
    return parts[0] if parts else ""


def _pid_alive(pid: int) -> bool:
    """True/False when provable; True (conservative) when not — the heartbeat TTL still expires."""
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001 - psutil is a hard dependency; fall back per OS
        if os.name == "posix":
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except OSError:
                return True
            return True
        return True  # never call os.kill(pid, 0) on Windows: it terminates the process


def _handle_numbers(text: str) -> set[int]:
    """Output handles referenced by a message/state blob (``show_output(handle=N)``, ``[output #N``,
    ``{"handle":N}``, ``{"handles":[N,...]}``, ``#N-#M`` ranges). Over-matching only ever retains
    an output, never deletes one."""
    found: set[int] = set()
    for pattern in _HANDLE_PATTERNS:
        found.update(int(match.group(1)) for match in pattern.finditer(text))
    for match in _HANDLE_LIST_PATTERN.finditer(text):
        found.update(int(number) for number in re.findall(r"\d+", match.group(1)))
    for match in _HANDLE_RANGE_PATTERN.finditer(text):
        start, end = int(match.group(1)), int(match.group(2))
        if end >= start and end - start <= _HANDLE_RANGE_MAX:
            found.update(range(start, end + 1))
    return found


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


def _repair_pairs(history: Sequence[ChatMessage]) -> tuple[list[ChatMessage], int]:
    """Same placement rules as ``ironclad.engine.loop.repair_history``; also counts what it added."""
    repaired: list[ChatMessage] = []
    pending: list[ToolCall] = []
    added = 0

    def flush() -> None:
        nonlocal added
        for call in pending:
            repaired.append(
                ChatMessage(role="tool", content=INTERRUPTED_TEXT, tool_call_id=call.id, name=call.name)
            )
            added += 1
        pending.clear()

    for message in history:
        if message.role == "tool":
            match = next((c for c in pending if c.id == message.tool_call_id), None)
            if match is not None:
                pending.remove(match)
        else:
            flush()
            if message.role == "assistant":
                pending.extend(message.tool_calls)
        repaired.append(message)
    flush()
    return repaired, added


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

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id, project_root, title, created_ns, updated_ns FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "project_root": row[1],
            "title": row[2],
            "created_ns": row[3],
            "updated_ns": row[4],
        }

    def state_keys(self, session_id: str) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT key FROM state WHERE session_id = ? ORDER BY key", (session_id,)
            ).fetchall()
        return [str(row[0]) for row in rows]

    def delete_session(self, session_id: str, *, force: bool = False, ttl_s: float = LOCK_TTL_S) -> bool:
        """Delete a session and every row that belongs to it (messages, transcript, outputs,
        state, lock). Returns ``False`` and deletes nothing while the session is active
        (live lock) unless ``force`` is set. Unknown ids count as deleted (idempotent)."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT owner, heartbeat_ns FROM session_locks WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if not force and row is not None and self._lock_live(str(row[0]), int(row[1]), ttl_s):
                    self._connection.execute("ROLLBACK")
                    return False
                self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM transcript WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM outputs WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM state WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM session_locks WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return True

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

    # -- transcript (canonical display history) -----------------------------

    def append_transcript(
        self, session_id: str, role: str, text: str, meta: Mapping[str, Any] | None = None
    ) -> int:
        """Append one display entry. Separate from ``append_message``: the transcript is what the
        user sees and replays, the messages table is what the model is sent. Returns the new seq."""
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        rendered = json.dumps(
            dict(meta or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM transcript WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                seq = int(row[0]) + 1
                self._connection.execute(
                    "INSERT INTO transcript(session_id, seq, role, text, meta_json, created_ns)"
                    " VALUES(?,?,?,?,?,?)",
                    (session_id, seq, role, text, rendered, time.time_ns()),
                )
                self._touch(session_id, time.time_ns())
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return seq

    def transcript(self, session_id: str, limit: int = 0) -> list[dict[str, Any]]:
        """Display entries for ``session_id`` in chronological order; ``limit > 0`` keeps only the
        newest ``limit`` entries (still oldest-first). ``limit <= 0`` returns everything."""
        with self._lock:
            if limit and int(limit) > 0:
                rows = self._connection.execute(
                    "SELECT seq, role, text, meta_json, created_ns FROM transcript"
                    " WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
                    (session_id, int(limit)),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = self._connection.execute(
                    "SELECT seq, role, text, meta_json, created_ns FROM transcript"
                    " WHERE session_id = ? ORDER BY seq",
                    (session_id,),
                ).fetchall()
        return [
            {
                "seq": int(row[0]),
                "role": row[1],
                "text": row[2],
                "meta": json.loads(row[3]),
                "created_ns": int(row[4]),
            }
            for row in rows
        ]

    def transcript_count(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM transcript WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def repair_interrupted(self, session_id: str) -> int:
        """Answer every stored assistant tool call that has no tool result (the process died
        mid-run) with one synthetic ``INTERRUPTED_TEXT`` result. Idempotent; returns how many
        results were added. Uses ``replace_messages`` only when a placeholder lands mid-history."""
        history = self.messages(session_id)
        repaired, added = _repair_pairs(history)
        if added == 0:
            return 0
        if repaired[: len(history)] == history:
            for message in repaired[len(history) :]:
                self.append_message(session_id, message)
        else:
            self.replace_messages(session_id, repaired)
        return added

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

    def output_for_call(self, session_id: str, call_id: str) -> str | None:
        """Full text stored for one tool call, or None. Newest handle wins if a call stored more than once."""
        if not call_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT full_text FROM outputs WHERE session_id = ? AND call_id = ?"
                " ORDER BY handle DESC LIMIT 1",
                (session_id, call_id),
            ).fetchone()
        return None if row is None else str(row[0])

    def outputs_report(self, session_id: str) -> dict[str, int]:
        """Visible retention picture for one session: how many handles exist, how many are still
        referenced by a message or task state, and how many are prunable."""
        with self._lock:
            existing = {
                int(row[0])
                for row in self._connection.execute(
                    "SELECT handle FROM outputs WHERE session_id = ?", (session_id,)
                )
            }
            referenced = self._referenced_handles(session_id) & existing
        return {
            "total": len(existing),
            "referenced": len(referenced),
            "unreferenced": len(existing - referenced),
        }

    def prune_outputs(self, project_root: str, *, cutoff_ns: int) -> dict[str, int]:
        """Delete outputs of *this project* that are older than ``cutoff_ns`` AND not referenced
        by any message or task state of their session. Referenced and recent handles always stay."""
        root = normalize_project_root(project_root)
        stats = {
            "sessions": 0,
            "outputs": 0,
            "deleted": 0,
            "retained_referenced": 0,
            "retained_recent": 0,
        }
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._connection.execute(
                    "SELECT id FROM sessions WHERE project_root = ?", (root,)
                ).fetchall()
                for (session_id,) in rows:
                    stats["sessions"] += 1
                    referenced = self._referenced_handles(str(session_id))
                    outputs = self._connection.execute(
                        "SELECT handle, created_ns FROM outputs WHERE session_id = ?", (session_id,)
                    ).fetchall()
                    for handle, created_ns in outputs:
                        stats["outputs"] += 1
                        if int(created_ns) >= cutoff_ns:
                            stats["retained_recent"] += 1
                        elif int(handle) in referenced:
                            stats["retained_referenced"] += 1
                        else:
                            self._connection.execute(
                                "DELETE FROM outputs WHERE session_id = ? AND handle = ?",
                                (session_id, handle),
                            )
                            stats["deleted"] += 1
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return stats

    def _referenced_handles(self, session_id: str) -> set[int]:
        """Handles mentioned by this session's messages (content/thinking/calls/name) or task state."""
        found: set[int] = set()
        blobs = self._connection.execute(
            "SELECT content || ' ' || thinking || ' ' || tool_calls_json || ' ' || COALESCE(name, '')"
            " FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        for (blob,) in blobs:
            found.update(_handle_numbers(str(blob or "")))
        states = self._connection.execute(
            "SELECT value_json FROM state WHERE session_id = ?", (session_id,)
        ).fetchall()
        for (blob,) in states:
            found.update(_handle_numbers(str(blob or "")))
        return found

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

    # -- activity locks ------------------------------------------------------

    def _lock_live(self, owner: str, heartbeat_ns: int, ttl_s: float) -> bool:
        """A lock is live while its heartbeat is fresh and (same host) its pid still exists."""
        if time.time_ns() - heartbeat_ns > int(max(ttl_s, 0.0) * 1_000_000_000):
            return False
        pid = _owner_pid(owner)
        if pid is not None and _owner_host(owner) == (socket.gethostname() or "unknown"):
            return _pid_alive(pid)
        return True

    def acquire_lock(self, session_id: str, owner: str, *, ttl_s: float = LOCK_TTL_S) -> bool:
        """Take or renew the session's activity lock. ``False`` when a live foreign holder has it;
        ``KeyError`` when the session does not exist. Replaces stale locks (expired heartbeat or a
        dead holder pid) so a crashed process never blocks a session for longer than ``ttl_s``."""
        now = time.time_ns()
        acquired = False
        error: KeyError | None = None
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                exists = self._connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if exists is None:
                    error = KeyError(f"no session {session_id}")
                else:
                    row = self._connection.execute(
                        "SELECT owner, heartbeat_ns FROM session_locks WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    held = (
                        row is not None
                        and str(row[0]) != owner
                        and self._lock_live(str(row[0]), int(row[1]), ttl_s)
                    )
                    if not held:
                        self._connection.execute(
                            "INSERT INTO session_locks(session_id, owner, acquired_ns, heartbeat_ns)"
                            " VALUES(?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET"
                            " owner = excluded.owner, heartbeat_ns = excluded.heartbeat_ns",
                            (session_id, owner, now, now),
                        )
                        acquired = True
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        if error is not None:
            raise error
        return acquired

    def refresh_lock(self, session_id: str, owner: str) -> bool:
        """Heartbeat: keeps our own lock live. ``False`` when it is gone or held by someone else."""
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE session_locks SET heartbeat_ns = ? WHERE session_id = ? AND owner = ?",
                (time.time_ns(), session_id, owner),
            )
            return cursor.rowcount == 1

    def release_lock(self, session_id: str, owner: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM session_locks WHERE session_id = ? AND owner = ?",
                (session_id, owner),
            )

    def lock_info(self, session_id: str, *, ttl_s: float = LOCK_TTL_S) -> dict[str, Any] | None:
        """``None`` when nothing is recorded; otherwise owner/timestamps plus ``active``."""
        with self._lock:
            row = self._connection.execute(
                "SELECT owner, acquired_ns, heartbeat_ns FROM session_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        owner = str(row[0])
        return {
            "owner": owner,
            "pid": _owner_pid(owner),
            "acquired_ns": int(row[1]),
            "heartbeat_ns": int(row[2]),
            "active": self._lock_live(owner, int(row[2]), ttl_s),
        }

    def is_active(self, session_id: str, *, ttl_s: float = LOCK_TTL_S) -> bool:
        info = self.lock_info(session_id, ttl_s=ttl_s)
        return bool(info and info["active"])
