"""Read-before-edit file tools: read_file / write_file / edit_file / list_files.

Write and edit refuse to touch a file whose sha256 the model has not seen: existing
files need an ``expected_sha`` matching the current bytes ("read it first"). Path
arguments resolve (symlinks followed) inside the project root, or inside an extra
read root for reads only; anything else is INVALID and nothing is executed.
"""

from __future__ import annotations

import bisect
import contextlib
import difflib
import fnmatch
import hashlib
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from ..contracts import ToolCall, ToolResult, ToolSpec, ToolStatus
from ..types import RiskLevel
from .schema import validate

READ_FILE, WRITE_FILE, EDIT_FILE, LIST_FILES = "read_file", "write_file", "edit_file", "list_files"
TOOLS = (READ_FILE, WRITE_FILE, EDIT_FILE, LIST_FILES)
DEFAULT_PROTECTED = (".git", ".nedster")
MAX_LIST_ENTRIES = 2000
MAX_DIFF_LINES = 200
MAX_REPORTED_MATCH_LINES = 3
MAX_EDITS = 50
NEWLINE_CRLF = "\r\n"
WALK_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
MAX_FILE_BYTES = 10 * 1024 * 1024
READ_CHUNK = 64 * 1024
SHORT_NAME = re.compile(r"^(?P<stem>[^~]*)~\d+(?:\..*)?$")
Inventory = Callable[[Path], list[str]]


class MutationJournal(Protocol):
    """Receives each path's pre-image before its first mutation in a turn (see session_store.checkpoints).
    ``record_pre_image`` raises OSError when it cannot journal; the mutation is then refused."""

    @property
    def mutation_lock(self) -> Any: ...

    def record_pre_image(self, rel: str, path: Path) -> None: ...


class ToolError(Exception):
    """Argument/scope problem. Mapped to ToolStatus.INVALID; nothing was executed."""


class Located(NamedTuple):
    path: Path  # resolved absolute path (symlinks followed)
    rel: str  # POSIX path relative to ``root``
    root: Path  # the allowed root that contains ``path``


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SHORT_SHA = 16


def hash_matches(expected: str, digest: str) -> bool:
    value = expected.strip().lower()
    if value.startswith("sha256:"):
        value = value[len("sha256:") :]
    # read_file shows the first 16 hex digits; models often "complete" them into an invented 64-digit value.
    # 16 digits (64 bits) of the current hash are as good evidence of a read as the full hash.
    return value == digest or (len(value) >= SHORT_SHA and value[:SHORT_SHA] == digest[:SHORT_SHA])


def _fold(name: str) -> str:
    return name.rstrip(" .").casefold()


def is_protected(parts: Sequence[str], protected: Sequence[str]) -> bool:
    """True when any path component names (or aliases) a protected directory.

    Case-insensitive on every OS, ignores trailing dots/spaces (Windows), rejects NTFS
    alternate data stream syntax (``:``) and 8.3 short names such as ``NEDSTE~1``.
    """
    names = {_fold(item) for item in protected}
    stems = {name.lstrip(".") for name in names}
    for part in parts:
        folded = _fold(part)
        if folded in names:
            return True
        if ":" in part and (os.name == "nt" or _fold(part.split(":", 1)[0]) in names):
            return True
        short = SHORT_NAME.match(folded)
        if short:
            stem = short.group("stem").lstrip(".")
            if stem and any(name.startswith(stem) for name in stems):
                return True
    return False


def locate(
    root: Path,
    raw: str,
    *,
    extra_roots: Sequence[Path] = (),
    protected: Sequence[str] = (),
    for_write: bool = False,
) -> Located:
    if not raw or "\x00" in raw:
        raise ToolError(f"invalid path {raw!r}")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ToolError(f"cannot resolve path {raw!r}: {exc}") from exc
    bases = (root, *extra_roots) if not for_write else (root,)
    for base in bases:
        try:
            rel = resolved.relative_to(base)
        except ValueError:
            continue
        if for_write and is_protected(rel.parts, protected):
            raise ToolError(f"path is inside a protected directory: {rel.as_posix()}")
        return Located(resolved, rel.as_posix(), base)
    raise ToolError(f"path escapes the project root: {raw}")


def read_regular(path: Path, rel: str, limit: int) -> tuple[bytes, str]:
    """Read a regular file of at most ``limit`` bytes; returns (data, sha256) hashed while streaming."""
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ToolError(f"not a regular file: {rel}")
    if info.st_size > limit:
        raise ToolError(f"file too large ({info.st_size} > {limit} bytes): {rel}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    with open(path, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ToolError(f"not a regular file: {rel}")
        while True:
            chunk = handle.read(min(READ_CHUNK, limit + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ToolError(f"file too large (> {limit} bytes): {rel}")
            digest.update(chunk)
            chunks.append(chunk)
    return b"".join(chunks), digest.hexdigest()


def detect_newline(data: bytes) -> str | None:
    if NEWLINE_CRLF.encode() in data:
        return NEWLINE_CRLF
    return "\n" if b"\n" in data else None


def apply_newline(text: str, newline: str | None) -> str:
    if newline is None:
        return text
    return text.replace("\r\n", "\n").replace("\n", newline)


def walk_inventory(base: Path) -> list[str]:
    """Sorted POSIX file paths under ``base`` (skips VCS/build dirs). The default inventory."""
    entries: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if name not in WALK_SKIP_DIRS)
        for name in filenames:
            try:
                entries.add((Path(dirpath) / name).relative_to(base).as_posix())
            except ValueError:
                continue
    return sorted(entries)


def glob_match(pattern: str, rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    if "/" in pattern:
        return fnmatch.fnmatchcase(rel, pattern)
    return fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(rel, pattern)


def line_numbers(text: str, needle: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    numbers: list[int] = []
    offset = text.find(needle)
    while offset != -1:
        numbers.append(bisect.bisect_right(starts, offset))
        offset = text.find(needle, offset + 1)
    return numbers


def unified_diff(before: str, after: str, rel: str) -> str:
    lines = list(
        difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="")
    )
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"[diff truncated at {MAX_DIFF_LINES} lines]"]
    return "\n".join(lines) or "(no textual difference)"


def atomic_replace(target: Path, payload: bytes, mode: int | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


_PATH_PROP = {"type": "string", "description": "Project-relative or absolute path inside the project."}
READ_SCHEMA = {
    "type": "object",
    "properties": {"path": _PATH_PROP, "start_line": {"type": "integer", "minimum": 1},
                   "end_line": {"type": "integer", "minimum": 1}},  # fmt: skip
    "required": ["path"],
    "additionalProperties": False,
}
WRITE_SCHEMA = {
    "type": "object",
    "properties": {"path": _PATH_PROP, "content": {"type": "string"}, "expected_sha": {"type": "string"}},
    "required": ["path", "content"],
    "additionalProperties": False,
}
_EDIT_ITEM = {
    "type": "object",
    "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
    "required": ["old", "new"],
    "additionalProperties": False,
}
# Either {old, new[, count]} or edits[1..MAX_EDITS]. The bundled validator ignores oneOf/minItems/maxItems, so
# _edit_list enforces the exclusivity and the bounds itself; the keywords document the contract for the model.
EDIT_SCHEMA = {
    "type": "object",
    "properties": {"path": _PATH_PROP, "old": {"type": "string"}, "new": {"type": "string"},
                   "expected_sha": {"type": "string"}, "count": {"type": "integer", "minimum": 1},
                   "edits": {"type": "array", "items": _EDIT_ITEM, "minItems": 1, "maxItems": MAX_EDITS}},  # fmt: skip
    "required": ["path", "expected_sha"],
    "oneOf": [{"required": ["old", "new"]}, {"required": ["edits"]}],
    "additionalProperties": False,
}
LIST_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "glob": {"type": "string"}},
    "additionalProperties": False,
}


class FileTools:
    """Sync ToolRuntime for project files. Roots are resolved once at construction and immutable."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        extra_read_roots: Sequence[str | Path] = (),
        protected: Sequence[str] = DEFAULT_PROTECTED,
        max_read_bytes: int = 200_000,
        max_file_bytes: int = MAX_FILE_BYTES,
        inventory: Inventory | None = None,
        journal: MutationJournal | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.extra_read_roots = tuple(Path(item).resolve() for item in extra_read_roots)
        self.protected = tuple(protected)
        self.max_read_bytes = int(max_read_bytes)
        self.max_file_bytes = int(max_file_bytes)
        self._inventory = inventory or walk_inventory
        self._backend = "inventory" if inventory else "walk"
        self.read_hashes: dict[str, str] = {}
        self._own_writes: dict[str, tuple[str, str]] = {}  # rel -> (sha256 we wrote, HH:MM:SS)
        self.journal = journal
        self.mutation_lock: Any = journal.mutation_lock if journal is not None else threading.RLock()
        self._schemas: dict[str, Mapping[str, Any]] = {
            READ_FILE: READ_SCHEMA,
            WRITE_FILE: WRITE_SCHEMA,
            EDIT_FILE: EDIT_SCHEMA,
            LIST_FILES: LIST_SCHEMA,
        }
        self._handlers = {
            READ_FILE: self._read_file,
            WRITE_FILE: self._write_file,
            EDIT_FILE: self._edit_file,
            LIST_FILES: self._list_files,
        }

    # -- ToolRuntime -------------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        read = ToolSpec(
            READ_FILE,
            "Read a UTF-8 file as numbered lines. Needs no prior state; returns the whole-file sha256.",
            READ_SCHEMA,
            int(RiskLevel.READ),
        )
        write = ToolSpec(
            WRITE_FILE,
            "Create or replace a file. Existing files require expected_sha from a prior read_file (read first). "
            "New files are created exclusively.",
            WRITE_SCHEMA,
            int(RiskLevel.WRITE),
            mutates=True,
        )
        edit = ToolSpec(
            EDIT_FILE,
            "Replace an exact substring (old/new, count times) or apply edits=[{old,new},...] (1..50, in order, one "
            "atomic write; each old must match exactly once; any failure writes nothing). Requires expected_sha "
            "from a prior read_file or from the sha a previous write/edit returned. Returns a unified diff and the "
            "new sha256.",
            EDIT_SCHEMA,
            int(RiskLevel.WRITE),
            mutates=True,
        )
        listing = ToolSpec(
            LIST_FILES,
            "List project files under a directory, optionally filtered by a glob.",
            LIST_SCHEMA,
            int(RiskLevel.READ),
        )
        return [read, write, edit, listing]

    def run(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(call.id, call.name, ToolStatus.INVALID, f"unknown tool {call.name!r}")
        errors = validate(self._schemas[call.name], dict(call.arguments or {}))
        if errors:
            return ToolResult(call.id, call.name, ToolStatus.INVALID, "\n".join(errors))
        try:
            return handler(call)
        except ToolError as exc:
            return ToolResult(call.id, call.name, ToolStatus.INVALID, str(exc))
        except OSError as exc:
            return ToolResult(call.id, call.name, ToolStatus.FAILED, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # a tool bug must not escape as an exception
            return ToolResult(call.id, call.name, ToolStatus.FAILED, f"tool error: {exc}")

    # -- helpers -----------------------------------------------------------------------

    def _locate(self, raw: str, *, for_write: bool) -> Located:
        return locate(
            self.project_root,
            raw,
            extra_roots=self.extra_read_roots,
            protected=self.protected,
            for_write=for_write,
        )

    @staticmethod
    def _decode(data: bytes, rel: str) -> str:
        if b"\x00" in data:
            raise ToolError(f"binary file, refusing: {rel}")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"binary file (not valid UTF-8): {rel}") from exc

    def _journal_before(self, located: Located) -> None:
        if self.journal is not None:
            self.journal.record_pre_image(located.rel, located.path)

    def invalidate_read_hashes(self, paths: Sequence[str] | None = None) -> None:
        """Forget read-before-edit hashes (all, or the given relative paths) so stale reads cannot authorize edits."""
        with self.mutation_lock:
            if paths is None:
                self.read_hashes.clear()
                self._own_writes.clear()
                return
            drop = {os.path.normcase(path) for path in paths}
            for rel in [rel for rel in self.read_hashes if os.path.normcase(rel) in drop]:
                del self.read_hashes[rel]
            for rel in [rel for rel in self._own_writes if os.path.normcase(rel) in drop]:
                del self._own_writes[rel]

    def _stale_error(self, rel: str, digest: str) -> ToolError:
        """Stale/missing expected_sha. Reveals the current sha only when this instance wrote those exact bytes."""
        own = self._own_writes.get(rel)
        if own is not None and own[0] == digest:
            return ToolError(
                f"file changed since your read - by your own edit at {own[1]}; current sha256:{digest[:16]}"
            )
        if self.read_hashes.get(rel) == digest:  # the model read exactly these bytes; it only mis-copied the sha
            return ToolError(
                f"expected_sha does not match {rel}, which is unchanged since your read: "
                f"pass expected_sha \"{digest[:SHORT_SHA]}\" (copy it from the read_file header)"
            )
        return ToolError("file changed since read / read it first")

    def _record_write(self, rel: str, digest: str) -> None:
        self.read_hashes[rel] = digest
        self._own_writes[rel] = (digest, time.strftime("%H:%M:%S"))

    # -- tools -------------------------------------------------------------------------

    def _read_file(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        located = self._locate(str(args["path"]), for_write=False)
        data, digest = read_regular(located.path, located.rel, self.max_file_bytes)
        text = self._decode(data, located.rel)
        lines = text.splitlines()
        total = len(lines)
        start = int(args.get("start_line", 1))
        end = int(args.get("end_line", total or 1))
        if end < start:
            raise ToolError(f"end_line {end} is before start_line {start}")
        self.read_hashes[located.rel] = digest
        selected = lines[start - 1 : end]
        width = max(len(str(total)), 1)
        header = (
            f"{located.rel}: lines {start}-{start - 1 + len(selected)} of {total}, sha256:{digest[:16]}"
            if selected
            else f"{located.rel}: 0 of {total} lines, sha256:{digest[:16]}"
        )
        body = [f"{number:>{width}}| {lines[number - 1]}" for number in range(start, start + len(selected))]
        output = header + ("\n" + "\n".join(body) if body else "")
        truncated_at: int | None = None
        while len(output.encode("utf-8")) > self.max_read_bytes and body:
            body.pop()
            truncated_at = start + len(body)
            output = header + ("\n" + "\n".join(body) if body else "")
        if truncated_at is not None:
            output += f"\n[truncated: use start_line={truncated_at}]"
        data_field = {"path": located.rel, "total_lines": total, "sha256": digest}
        return ToolResult(call.id, call.name, ToolStatus.OK, output, data_field)

    def _write_file(self, call: ToolCall) -> ToolResult:
        with self.mutation_lock:
            return self._write_locked(call)

    def _write_locked(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        located = self._locate(str(args["path"]), for_write=True)
        content = str(args["content"])
        expected = args.get("expected_sha")
        target = located.path
        created = False
        if target.exists():
            if target.is_dir():
                raise ToolError(f"is a directory: {located.rel}")
            current, digest = read_regular(target, located.rel, self.max_file_bytes)
            if not expected or not hash_matches(str(expected), digest):
                raise self._stale_error(located.rel, digest)
            payload = apply_newline(content, detect_newline(current)).encode("utf-8")
            self._journal_before(located)
            atomic_replace(target, payload, stat.S_IMODE(target.stat().st_mode))
        else:
            if expected:
                raise ToolError("file changed since read / read it first")
            payload = content.encode("utf-8")
            self._journal_before(located)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(target, "xb") as handle:
                    handle.write(payload)
            except FileExistsError as exc:
                raise ToolError("file changed since read / read it first") from exc
            created = True
        new_digest = sha256_hex(payload)
        self._record_write(located.rel, new_digest)
        verb = "created" if created else "updated"
        return ToolResult(
            call.id,
            call.name,
            ToolStatus.OK,
            f"{verb} {located.rel} ({len(payload)} bytes), new sha256:{new_digest[:16]}",
            {"sha256": new_digest, "sha": new_digest[:16], "created": created},
            (located.rel,),
        )

    def _edit_file(self, call: ToolCall) -> ToolResult:
        with self.mutation_lock:
            return self._edit_locked(call)

    @staticmethod
    def _edit_list(args: Mapping[str, Any]) -> tuple[list[tuple[str, str]], int, bool]:
        """Normalise the two accepted forms to ([(old, new)], count, is_multi); raises ToolError on misuse."""
        multi = "edits" in args
        if multi and ("old" in args or "new" in args or "count" in args):
            raise ToolError("give either old/new[/count] or edits, not both")
        if not multi:
            if "old" not in args or "new" not in args:
                raise ToolError("edit_file needs 'old' and 'new', or 'edits'")
            return [(str(args["old"]), str(args["new"]))], int(args.get("count", 1)), False
        raw = args["edits"]
        if not 1 <= len(raw) <= MAX_EDITS:
            raise ToolError(f"edits must contain 1..{MAX_EDITS} items, got {len(raw)}")
        return [(str(item["old"]), str(item["new"])) for item in raw], 1, True

    @staticmethod
    def _apply_edits(text: str, pairs: Sequence[tuple[str, str]], count: int, multi: bool) -> str:
        prefix_tail = "; nothing was written" if multi else ""
        for index, (old, new) in enumerate(pairs):
            prefix = f"edits[{index}]: " if multi else ""
            if not old:
                raise ToolError(f"{prefix}old must not be empty{prefix_tail}")
            actual = text.count(old)
            if actual != count:
                hits = line_numbers(text, old)[:MAX_REPORTED_MATCH_LINES]
                where = f" at line(s) {', '.join(str(number) for number in hits)}" if hits else ""
                raise ToolError(f"{prefix}'old' occurs {actual} time(s), expected {count}{where}{prefix_tail}")
            text = text.replace(old, new)
        return text

    def _edit_locked(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        located = self._locate(str(args["path"]), for_write=True)
        pairs, count, multi = self._edit_list(args)
        current, digest = read_regular(located.path, located.rel, self.max_file_bytes)
        if not hash_matches(str(args["expected_sha"]), digest):
            raise self._stale_error(located.rel, digest)
        text = self._decode(current, located.rel)
        edited = apply_newline(self._apply_edits(text, pairs, count, multi), detect_newline(current))
        payload = edited.encode("utf-8")
        if payload == current:
            return ToolResult(
                call.id,
                call.name,
                ToolStatus.OK,
                f"no change: replacement produced identical content, sha256:{digest[:16]}",
                {"sha256": digest, "sha": digest[:16]},
            )
        self._journal_before(located)
        atomic_replace(located.path, payload, stat.S_IMODE(located.path.stat().st_mode))
        new_digest = sha256_hex(payload)
        self._record_write(located.rel, new_digest)
        output = unified_diff(text, edited, located.rel) + f"\nnew sha256:{new_digest[:16]}"
        return ToolResult(
            call.id,
            call.name,
            ToolStatus.OK,
            output,
            {"sha256": new_digest, "sha": new_digest[:16], "edits": len(pairs)},
            (located.rel,),
        )

    def _list_files(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        raw_path = str(args.get("path") or ".")
        located = self._locate(raw_path, for_write=False)
        if not located.path.exists():
            raise ToolError(f"no such directory: {located.rel}")
        if not located.path.is_dir():
            raise ToolError(f"not a directory: {located.rel}")
        entries = sorted(set(self._inventory(located.root)))
        if located.rel != ".":
            prefix = located.rel + "/"
            entries = [entry for entry in entries if entry.startswith(prefix)]
        pattern = args.get("glob")
        if pattern:
            entries = [entry for entry in entries if glob_match(str(pattern), entry)]
        total = len(entries)
        shown = entries[:MAX_LIST_ENTRIES]
        output = "\n".join(shown)
        if total > len(shown):
            output += f"\n[truncated: showing first {len(shown)} of {total} entries]"
        data_field = {
            "count": len(shown),
            "total": total,
            "truncated": total > len(shown),
            "backend": self._backend,
        }
        return ToolResult(call.id, call.name, ToolStatus.OK, output, data_field)


__all__ = ["TOOLS", "FileTools", "ToolError", "locate", "walk_inventory", "is_protected", "sha256_hex", "hash_matches",
           "atomic_replace", "MutationJournal"]  # fmt: skip
