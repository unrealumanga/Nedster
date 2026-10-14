"""Per-turn file checkpoints and undo.

The file tools journal a path's pre-image (bytes or "absent", plus mode) before its FIRST mutation in a user
turn; ``end_turn`` stores the post-image hash. ``undo_last`` preflights every path of the latest journaled turn
and refuses the whole undo if any path conflicts, then restores atomically per file under a recovery journal
that ``recover`` completes or rolls back after an interrupted undo. Command side effects are never covered.

Tables live in the sessions DB file (own connection, WAL, ``BEGIN IMMEDIATE``).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..tool_runtime.files import DEFAULT_PROTECTED, ToolError, atomic_replace, locate

CHUNK = 64 * 1024
ABSENT = "absent"
DIR = "directory"
UNREADABLE = "unreadable"

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS ckpt_turns (
        turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        root TEXT NOT NULL,
        started_ns INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'open',
        commands INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ckpt_turns_session ON ckpt_turns(session_id, turn_id DESC)",
    """
    CREATE TABLE IF NOT EXISTS ckpt_files (
        turn_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        rel TEXT NOT NULL,
        seq INTEGER NOT NULL,
        pre_exists INTEGER NOT NULL,
        pre_blob BLOB,
        pre_mode INTEGER,
        pre_sha TEXT NOT NULL,
        created_dirs TEXT NOT NULL DEFAULT '[]',
        post_sha TEXT,
        PRIMARY KEY (turn_id, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ckpt_undo (
        undo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'restoring'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ckpt_undo_files (
        undo_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        cur_exists INTEGER NOT NULL,
        cur_blob BLOB,
        cur_mode INTEGER,
        PRIMARY KEY (undo_id, key)
    )
    """,
)


def path_key(rel: str) -> str:
    """Identity of a path inside the journal: case-folded where the filesystem is (Windows)."""
    return os.path.normcase(rel)


def digest_path(path: Path) -> str:
    """sha256 of a regular file, ``absent``, ``directory`` or ``unreadable`` (never raises)."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return ABSENT
    except OSError:
        return UNREADABLE
    if stat.S_ISDIR(info.st_mode):
        return DIR
    if not stat.S_ISREG(info.st_mode):
        return UNREADABLE
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return UNREADABLE
    return digest.hexdigest()


def _missing_dirs(root: Path, path: Path) -> list[str]:
    missing: list[str] = []
    parent = path.parent
    while parent != root and not parent.exists() and root in parent.parents:
        missing.append(parent.relative_to(root).as_posix())
        parent = parent.parent
    return missing  # deepest first


@dataclass(slots=True, frozen=True)
class UndoConflict:
    path: str
    reason: str


@dataclass(slots=True)
class UndoReport:
    ok: bool
    turn_id: int | None = None
    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    conflicts: list[UndoConflict] = field(default_factory=list)
    commands_not_covered: bool = False
    message: str = ""

    @property
    def paths(self) -> list[str]:
        return [*self.restored, *self.removed]

    def __str__(self) -> str:
        return self.message


_PARTIAL = (
    " This turn also ran commands (run_command/run_tests/git); their side effects are NOT covered by undo "
    "and were not reverted."
)


class CheckpointStore:
    """File-mutation journal + undo. One instance per session store; thread-safe."""

    def __init__(self, db_path: str | Path, *, protected: Sequence[str] = DEFAULT_PROTECTED) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.protected = tuple(protected)
        self._lock = threading.RLock()
        self._mutation_lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        for statement in _SCHEMA:
            self._connection.execute(statement)
        self._turn: tuple[int, str, Path] | None = None  # (turn_id, session_id, root) of the open turn
        self._pending: tuple[str, Path] | None = None  # turn declared but not yet journaled (lazy row)
        self.on_undone: list[Callable[[list[str]], None]] = []

    # -- plumbing ----------------------------------------------------------------------

    @property
    def mutation_lock(self) -> threading.RLock:
        """Held by file tools across journal+mutate and by undo across preflight+restore."""
        return self._mutation_lock

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _tx(self):
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")

    # -- journal -----------------------------------------------------------------------

    def begin_turn(self, session_id: str, root: str | Path) -> None:
        """Declare a user turn. The DB row is created lazily by the first mutation or command."""
        with self._lock:
            if self._turn is not None:
                self.end_turn()
            self._pending = (session_id, Path(root).resolve())

    def _ensure_turn(self) -> tuple[int, Path]:
        if self._turn is None:
            if self._pending is None:
                raise OSError("no open turn: file mutations are refused without a checkpoint turn")
            session_id, root = self._pending
            with self._tx() as db:
                cursor = db.execute(
                    "INSERT INTO ckpt_turns(session_id, root, started_ns) VALUES (?, ?, ?)",
                    (session_id, str(root), time.time_ns()),
                )
            self._turn = (int(cursor.lastrowid), session_id, root)
        return self._turn[0], self._turn[2]

    def record_pre_image(self, rel: str, path: Path) -> None:
        """Store the pre-image of ``path`` unless this turn already did. Raises OSError on failure
        so the caller refuses the mutation (fail closed)."""
        with self._lock:
            turn_id, root = self._ensure_turn()
            key = path_key(rel)
            if self._connection.execute(
                "SELECT 1 FROM ckpt_files WHERE turn_id=? AND key=?", (turn_id, key)
            ).fetchone():
                return
            exists = os.path.lexists(path)
            blob: bytes | None = None
            mode: int | None = None
            sha = ABSENT
            if exists:
                if path.is_dir() or path.is_symlink():
                    raise OSError(f"cannot checkpoint a directory or symlink: {rel}")
                blob = path.read_bytes()
                mode = stat.S_IMODE(path.stat().st_mode)
                sha = hashlib.sha256(blob).hexdigest()
            created = json.dumps(_missing_dirs(root, path))
            with self._tx() as db:
                seq = db.execute("SELECT COUNT(*) FROM ckpt_files WHERE turn_id=?", (turn_id,)).fetchone()[0]
                db.execute(
                    "INSERT INTO ckpt_files(turn_id,key,rel,seq,pre_exists,pre_blob,pre_mode,pre_sha,created_dirs)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (turn_id, key, rel, seq, int(exists), blob, mode, sha, created),
                )

    def note_command(self) -> None:
        """A command ran this turn: its side effects are outside undo coverage."""
        with self._lock:
            if self._turn is None and self._pending is None:
                return
            turn_id, _ = self._ensure_turn()
            with self._tx() as db:
                db.execute("UPDATE ckpt_turns SET commands = commands + 1 WHERE turn_id=?", (turn_id,))

    def end_turn(self) -> None:
        """Close the open turn: store the post-image hash of every journaled path."""
        with self._lock:
            self._pending = None
            if self._turn is None:
                return
            turn_id, _, root = self._turn
            self._turn = None
            self._close_turn(turn_id, root)

    def _close_turn(self, turn_id: int, root: Path) -> None:
        rows = self._connection.execute("SELECT key, rel FROM ckpt_files WHERE turn_id=?", (turn_id,)).fetchall()
        posts = [(digest_path(root / rel), turn_id, key) for key, rel in rows]
        with self._tx() as db:
            db.executemany("UPDATE ckpt_files SET post_sha=? WHERE turn_id=? AND key=?", posts)
            db.execute("UPDATE ckpt_turns SET state='closed' WHERE turn_id=? AND state='open'", (turn_id,))

    # -- recovery ----------------------------------------------------------------------

    def recover(self) -> list[str]:
        """Run on start: close turns left open by a crash (post-image = what is on disk now), then
        complete or roll back interrupted undos. Returns human-readable notes."""
        notes: list[str] = []
        with self._lock, self._mutation_lock:
            for turn_id, root in self._connection.execute(
                "SELECT turn_id, root FROM ckpt_turns WHERE state='open'"
            ).fetchall():
                self._close_turn(int(turn_id), Path(root))
                notes.append(f"closed interrupted turn {turn_id}")
            for undo_id, turn_id in self._connection.execute(
                "SELECT undo_id, turn_id FROM ckpt_undo WHERE state='restoring'"
            ).fetchall():
                notes.append(self._recover_undo(int(undo_id), int(turn_id)))
        return notes

    def _recover_undo(self, undo_id: int, turn_id: int) -> str:
        root = Path(self._connection.execute("SELECT root FROM ckpt_turns WHERE turn_id=?", (turn_id,)).fetchone()[0])
        files = self._turn_files(turn_id)
        states = {key: self._classify(root, row) for key, row in files.items()}
        roll_forward = all(state in ("pre", "post") for state in states.values())
        for key, row in files.items():
            target = root / row["rel"]
            if roll_forward and states[key] == "post":
                self._restore_pre(root, row)
            elif not roll_forward and states[key] == "pre" and row["pre_sha"] != row["post_sha"]:
                self._restore_backup(undo_id, key, target)
        final = "done" if roll_forward else "rolled_back"
        with self._tx() as db:
            db.execute("UPDATE ckpt_undo SET state=? WHERE undo_id=?", (final, undo_id))
            if roll_forward:
                db.execute("UPDATE ckpt_turns SET state='undone' WHERE turn_id=?", (turn_id,))
        verb = "completed" if roll_forward else "rolled back"
        return f"{verb} interrupted undo of turn {turn_id}"

    # -- undo --------------------------------------------------------------------------

    def _turn_files(self, turn_id: int) -> dict[str, sqlite3.Row]:
        cursor = self._connection.execute(
            "SELECT key, rel, pre_exists, pre_blob, pre_mode, pre_sha, created_dirs, post_sha"
            " FROM ckpt_files WHERE turn_id=? ORDER BY seq",
            (turn_id,),
        )
        names = [column[0] for column in cursor.description]
        return {row[0]: dict(zip(names, row, strict=True)) for row in cursor.fetchall()}  # type: ignore[misc]

    @staticmethod
    def _classify(root: Path, row: dict) -> str:
        current = digest_path(root / row["rel"])
        if current == row["pre_sha"]:
            return "pre"
        if current == row["post_sha"]:
            return "post"
        return "other"

    def latest_undoable(self, session_id: str) -> int | None:
        row = self._connection.execute(
            "SELECT turn_id FROM ckpt_turns WHERE session_id=? AND state='closed' ORDER BY turn_id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return int(row[0]) if row else None

    def undo_last(self, session_id: str) -> UndoReport:
        """Undo the latest journaled turn of ``session_id``; refuses wholesale on any conflict."""
        with self._mutation_lock, self._lock:
            if self._turn is not None:
                return UndoReport(False, message="Undo refused: a turn is still running.")
            turn_id = self.latest_undoable(session_id)
            if turn_id is None:
                return UndoReport(False, message="Nothing to undo: no journaled file changes in this session.")
            root_text, commands = self._connection.execute(
                "SELECT root, commands FROM ckpt_turns WHERE turn_id=?", (turn_id,)
            ).fetchone()
            root, ran = Path(root_text), bool(commands)
            files = self._turn_files(turn_id)
            conflicts = self._preflight(root, files)
            if conflicts:
                listing = "; ".join(f"{c.path} ({c.reason})" for c in conflicts)
                note = _PARTIAL if ran else ""
                return UndoReport(
                    False, turn_id, conflicts=conflicts, commands_not_covered=ran,
                    message=f"Undo refused, nothing was changed. Conflicts: {listing}.{note}",
                )
            report = self._restore(turn_id, root, files)
            report.commands_not_covered = ran
            return report

    def _preflight(self, root: Path, files: dict[str, dict]) -> list[UndoConflict]:
        conflicts: list[UndoConflict] = []
        for row in files.values():
            rel = row["rel"]
            if row["post_sha"] is None:
                conflicts.append(UndoConflict(rel, "post-image was never recorded"))
                continue
            try:
                located = locate(root, rel, protected=self.protected, for_write=True)
            except ToolError as exc:
                conflicts.append(UndoConflict(rel, str(exc)))
                continue
            if path_key(located.rel) != path_key(rel):
                conflicts.append(UndoConflict(rel, f"path now resolves to {located.rel} (link or case swap)"))
                continue
            if os.path.islink(root / rel):
                conflicts.append(UndoConflict(rel, "path is now a symbolic link"))
                continue
            current = digest_path(root / rel)
            if current != row["post_sha"]:
                conflicts.append(UndoConflict(rel, "changed since the turn ended"))
        return conflicts

    def _restore(self, turn_id: int, root: Path, files: dict[str, dict]) -> UndoReport:
        with self._tx() as db:
            undo_id = int(db.execute("INSERT INTO ckpt_undo(turn_id) VALUES (?)", (turn_id,)).lastrowid)
            for key, row in files.items():
                target = root / row["rel"]
                exists = target.exists()
                blob = target.read_bytes() if exists else None
                mode = stat.S_IMODE(target.stat().st_mode) if exists else None
                db.execute(
                    "INSERT INTO ckpt_undo_files(undo_id,key,cur_exists,cur_blob,cur_mode) VALUES (?,?,?,?,?)",
                    (undo_id, key, int(exists), blob, mode),
                )
        report = UndoReport(True, turn_id)
        try:
            for row in files.values():
                if row["pre_sha"] == row["post_sha"]:
                    continue
                self._restore_pre(root, row)
                (report.restored if row["pre_exists"] else report.removed).append(row["rel"])
        except OSError as exc:
            # Roll back the files already restored to the content they had before this undo.
            for key, row in files.items():
                if digest_path(root / row["rel"]) == row["pre_sha"] != row["post_sha"]:
                    self._restore_backup(undo_id, key, root / row["rel"])
            with self._tx() as db:
                db.execute("UPDATE ckpt_undo SET state='rolled_back' WHERE undo_id=?", (undo_id,))
            return UndoReport(
                False, turn_id, message=f"Undo failed ({exc}); files restored so far were rolled back.",
                commands_not_covered=False,
            )
        with self._tx() as db:
            db.execute("UPDATE ckpt_undo SET state='done' WHERE undo_id=?", (undo_id,))
            db.execute("UPDATE ckpt_turns SET state='undone' WHERE turn_id=?", (turn_id,))
        report.message = self._summary(report, self._commands(turn_id))
        for hook in list(self.on_undone):
            hook(report.paths)
        return report

    def _commands(self, turn_id: int) -> bool:
        row = self._connection.execute("SELECT commands FROM ckpt_turns WHERE turn_id=?", (turn_id,)).fetchone()
        return bool(row and row[0])

    @staticmethod
    def _summary(report: UndoReport, ran: bool) -> str:
        parts = []
        if report.restored:
            parts.append(f"restored {len(report.restored)} file(s): {', '.join(report.restored)}")
        if report.removed:
            parts.append(f"removed {len(report.removed)} created file(s): {', '.join(report.removed)}")
        text = "Undid file-tool changes of the last turn — " + ("; ".join(parts) or "nothing differed") + "."
        return text + (_PARTIAL if ran else "")

    def _restore_pre(self, root: Path, row: dict) -> None:
        target = root / row["rel"]
        if row["pre_exists"]:
            atomic_replace(target, bytes(row["pre_blob"] or b""), row["pre_mode"])
            return
        if target.exists():
            target.unlink()
        for rel in json.loads(row["created_dirs"]):  # deepest first
            directory = root / rel
            try:
                directory.rmdir()
            except OSError:
                break

    def _restore_backup(self, undo_id: int, key: str, target: Path) -> None:
        row = self._connection.execute(
            "SELECT cur_exists, cur_blob, cur_mode FROM ckpt_undo_files WHERE undo_id=? AND key=?", (undo_id, key)
        ).fetchone()
        if row is None:
            return
        if row[0]:
            atomic_replace(target, bytes(row[1] or b""), row[2])
        elif target.exists():
            target.unlink()


__all__ = ["CheckpointStore", "UndoConflict", "UndoReport", "digest_path", "path_key"]
