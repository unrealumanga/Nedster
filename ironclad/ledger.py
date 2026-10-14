from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_or_create_ledger_key(state_dir: str | Path, env_name: str = "NEDSTER_LEDGER_KEY") -> bytes:
    configured = os.getenv(env_name)
    if configured:
        return hashlib.sha256(configured.encode("utf-8")).digest()
    key_path = Path(state_dir) / "ledger.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        return key_path.read_bytes()
    key = secrets.token_bytes(32)
    temp_path = key_path.with_name(f"{key_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(temp_path, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, key_path)  # atomic create-if-absent; visible file is complete
        except FileExistsError:
            return key_path.read_bytes()  # a concurrent winner created it first
        except OSError:
            os.replace(temp_path, key_path)  # filesystems without hard links
            return key_path.read_bytes()
        return key
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink()


class EventLedger:
    """Append-only SQLite event log with a keyed hash chain."""

    GENESIS = "0" * 64

    def __init__(self, path: str | Path, key: bytes):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._key = key
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                actor TEXT NOT NULL,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            )
            """
        )
        self._connection.commit()

    def _digest(self, ts_ns: int, kind: str, actor: str, payload: str, previous: str) -> str:
        message = f"{ts_ns}\0{kind}\0{actor}\0{payload}\0{previous}".encode()
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def append(self, kind: str, actor: str, payload: dict[str, Any] | None = None) -> int:
        rendered = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                previous = row[0] if row else self.GENESIS
                ts_ns = time.time_ns()
                digest = self._digest(ts_ns, kind, actor, rendered, previous)
                cursor = self._connection.execute(
                    "INSERT INTO events(ts_ns, kind, actor, payload, prev_hash, event_hash) VALUES(?,?,?,?,?,?)",
                    (ts_ns, kind, actor, rendered, previous, digest),
                )
                self._connection.execute("COMMIT")
                return int(cursor.lastrowid)
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def iter_events(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        query = "SELECT seq, ts_ns, kind, actor, payload, prev_hash, event_hash FROM events ORDER BY seq"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        for row in self._connection.execute(query, params):
            yield {
                "seq": row[0],
                "ts_ns": row[1],
                "kind": row[2],
                "actor": row[3],
                "payload": json.loads(row[4]),
                "prev_hash": row[5],
                "event_hash": row[6],
            }

    def verify(self) -> tuple[bool, int, str]:
        with self._lock:
            previous = self.GENESIS
            count = 0
            for event in list(self.iter_events()):
                rendered = json.dumps(
                    event["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                expected = self._digest(
                    event["ts_ns"], event["kind"], event["actor"], rendered, previous
                )
                if event["prev_hash"] != previous or not hmac.compare_digest(
                    event["event_hash"], expected
                ):
                    return False, count, f"chain break at event {event['seq']}"
                previous = event["event_hash"]
                count += 1
            return True, count, previous

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> EventLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
