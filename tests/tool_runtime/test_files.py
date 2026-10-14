from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.files import FileTools


def call(name: str, call_id: str = "f1", **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.txt").write_bytes(b"one\ntwo\nthree\n")
    (project / "sub").mkdir()
    (project / "sub" / "b.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return project


@pytest.fixture
def tools(root: Path) -> FileTools:
    return FileTools(root)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_specs_shape(tools: FileTools):
    specs = tools.specs()
    assert {spec.name for spec in specs} == {"read_file", "write_file", "edit_file", "list_files"}
    read = {spec.name: spec for spec in specs}
    assert not read["read_file"].mutates and read["read_file"].risk == 0
    assert read["write_file"].mutates and read["write_file"].risk == 1
    assert read["edit_file"].mutates and read["edit_file"].risk == 1
    assert not read["list_files"].mutates


def test_read_numbers_lines_and_reports_hash(tools: FileTools, root: Path):
    result = tools.run(call("read_file", path="a.txt"))
    assert result.status is ToolStatus.OK, result.output
    assert "of 3" in result.output and "sha256:" + sha(b"one\ntwo\nthree\n")[:16] in result.output
    assert "1| one" in result.output and "3| three" in result.output
    assert tools.read_hashes["a.txt"] == sha(b"one\ntwo\nthree\n")
    assert result.data["total_lines"] == 3


def test_read_line_900_of_1000(tools: FileTools, root: Path):
    body = "\n".join(f"line {index}" for index in range(1, 1001)) + "\n"
    (root / "big.txt").write_text(body, encoding="utf-8", newline="\n")
    result = tools.run(call("read_file", path="big.txt", start_line=900, end_line=900))
    assert result.status is ToolStatus.OK
    assert "900| line 900" in result.output
    assert "line 899" not in result.output and "line 901" not in result.output


def test_read_range_and_empty_file(tools: FileTools, root: Path):
    (root / "empty.txt").write_bytes(b"")
    result = tools.run(call("read_file", path="a.txt", start_line=2, end_line=3))
    assert result.status is ToolStatus.OK
    assert "2| two" in result.output and "3| three" in result.output and "1| one" not in result.output
    empty = tools.run(call("read_file", path="empty.txt"))
    assert empty.status is ToolStatus.OK and "0 lines" in empty.output


def test_read_truncates_with_hint(tools: FileTools, root: Path):
    small = FileTools(root, max_read_bytes=80)
    (root / "many.txt").write_text("\n".join(f"line {index}" for index in range(1, 51)), encoding="utf-8", newline="\n")
    result = small.run(call("read_file", path="many.txt"))
    assert result.status is ToolStatus.OK
    assert "[truncated: use start_line=" in result.output
    assert len(result.output.encode()) <= 80 + 100


def test_read_binary_is_invalid(tools: FileTools, root: Path):
    (root / "blob.bin").write_bytes(b"\x00\x01\x02")
    result = tools.run(call("read_file", path="blob.bin"))
    assert result.status is ToolStatus.INVALID and "binary" in result.output


def test_read_missing_file_is_failed(tools: FileTools):
    result = tools.run(call("read_file", path="nope.txt"))
    assert result.status is ToolStatus.FAILED


def test_missing_content_is_invalid_and_file_unchanged(tools: FileTools, root: Path):
    before = (root / "a.txt").read_bytes()
    result = tools.run(call("write_file", path="a.txt"))
    assert result.status is ToolStatus.INVALID and "missing required field 'content'" in result.output
    assert (root / "a.txt").read_bytes() == before


def test_write_new_file_including_empty_content(tools: FileTools, root: Path):
    result = tools.run(call("write_file", path="new.txt", content=""))
    assert result.status is ToolStatus.OK, result.output
    assert (root / "new.txt").read_bytes() == b""
    assert result.mutated_paths == ("new.txt",)
    assert result.data["sha256"] == sha(b"")
    again = tools.run(call("write_file", path="new.txt", content="x"))
    assert again.status is ToolStatus.INVALID and "your own edit" in again.output and sha(b"")[:16] in again.output


def test_write_existing_requires_matching_hash(tools: FileTools, root: Path):
    stale = tools.run(call("write_file", path="a.txt", content="new\n"))
    assert stale.status is ToolStatus.INVALID and "read it first" in stale.output
    read = tools.run(call("read_file", path="a.txt"))
    digest = read.data["sha256"]
    ok = tools.run(call("write_file", path="a.txt", content="new\n", expected_sha=digest))
    assert ok.status is ToolStatus.OK and ok.data["sha256"] == sha(b"new\n")
    assert (root / "a.txt").read_bytes() == b"new\n"
    again = tools.run(call("write_file", path="a.txt", content="other\n", expected_sha=digest))
    assert again.status is ToolStatus.INVALID and "your own edit" in again.output and sha(b"new\n")[:16] in again.output


def test_write_accepts_sha_prefix_from_read_header(tools: FileTools, root: Path):
    read = tools.run(call("read_file", path="a.txt"))
    prefix = read.output.split("sha256:")[1].split()[0]
    assert len(prefix) == 16
    result = tools.run(call("write_file", path="a.txt", content="x\n", expected_sha=prefix))
    assert result.status is ToolStatus.OK


def test_write_expected_sha_for_missing_file_is_invalid(tools: FileTools, root: Path):
    result = tools.run(call("write_file", path="ghost.txt", content="x", expected_sha="0" * 64))
    assert result.status is ToolStatus.INVALID and "read it first" in result.output
    assert not (root / "ghost.txt").exists()


def test_edit_zero_or_two_matches_is_invalid(tools: FileTools, root: Path):
    read = tools.run(call("read_file", path="a.txt"))
    digest = read.data["sha256"]
    zero = tools.run(call("edit_file", path="a.txt", old="missing", new="x", expected_sha=digest))
    assert zero.status is ToolStatus.INVALID and "occurs 0 time(s)" in zero.output
    (root / "dup.txt").write_text("x\nx\n", encoding="utf-8", newline="\n")
    read_dup = tools.run(call("read_file", path="dup.txt"))
    twice = tools.run(
        call("edit_file", path="dup.txt", old="x", new="y", expected_sha=read_dup.data["sha256"])
    )
    assert twice.status is ToolStatus.INVALID and "occurs 2 time(s)" in twice.output and "line(s) 1, 2" in twice.output
    assert (root / "a.txt").read_bytes() == b"one\ntwo\nthree\n"
    assert (root / "dup.txt").read_bytes() == b"x\nx\n"


def test_edit_requires_stale_hash_and_returns_diff(tools: FileTools, root: Path):
    read = tools.run(call("read_file", path="a.txt"))
    digest = read.data["sha256"]
    stale = tools.run(call("edit_file", path="a.txt", old="two", new="TWO", expected_sha="0" * 64))
    # unchanged since the read: the error names the sha the model already saw
    assert stale.status is ToolStatus.INVALID and "unchanged since your read" in stale.output and digest[:16] in stale.output
    result = tools.run(call("edit_file", path="a.txt", old="two", new="TWO", expected_sha=digest))
    assert result.status is ToolStatus.OK, result.output
    assert result.output.startswith("--- a/a.txt")
    assert "-two" in result.output and "+TWO" in result.output
    assert result.mutated_paths == ("a.txt",)
    assert (root / "a.txt").read_bytes() == b"one\nTWO\nthree\n"
    assert result.data["sha256"] == sha(b"one\nTWO\nthree\n")


def test_edit_count_occurrences(tools: FileTools, root: Path):
    (root / "dup.txt").write_text("x x y\n", encoding="utf-8", newline="\n")
    digest = tools.run(call("read_file", path="dup.txt")).data["sha256"]
    result = tools.run(call("edit_file", path="dup.txt", old="x", new="z", expected_sha=digest, count=2))
    assert result.status is ToolStatus.OK
    assert (root / "dup.txt").read_bytes() == b"z z y\n"


def test_crlf_file_stays_crlf_after_edit(tools: FileTools, root: Path):
    (root / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    digest = tools.run(call("read_file", path="crlf.txt")).data["sha256"]
    result = tools.run(call("edit_file", path="crlf.txt", old="b", new="c", expected_sha=digest))
    assert result.status is ToolStatus.OK
    assert (root / "crlf.txt").read_bytes() == b"a\r\nc\r\n"
    rewritten = tools.run(call("write_file", path="crlf.txt", content="d\ne\n", expected_sha=result.data["sha256"]))
    assert rewritten.status is ToolStatus.OK
    assert (root / "crlf.txt").read_bytes() == b"d\r\ne\r\n"


def test_parent_escape_is_invalid(tools: FileTools, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    for name, arguments in (
        ("read_file", {"path": "../outside.txt"}),
        ("write_file", {"path": "../outside.txt", "content": "x"}),
        ("edit_file", {"path": "../outside.txt", "old": "s", "new": "x", "expected_sha": "0" * 64}),
        ("list_files", {"path": ".."}),
    ):
        result = tools.run(call(name, **arguments))
        assert result.status is ToolStatus.INVALID, (name, result.output)
    assert outside.read_text(encoding="utf-8") == "secret"


def test_symlink_escape_is_invalid(tools: FileTools, root: Path, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported here")
    result = tools.run(call("read_file", path="link.txt"))
    assert result.status is ToolStatus.INVALID and "escapes" in result.output
    write = tools.run(call("write_file", path="link.txt", content="x"))
    assert write.status is ToolStatus.INVALID


def test_write_into_protected_dirs_is_invalid(tools: FileTools, root: Path):
    for raw in (".git/config", ".nedster/state.json", "sub/.git/HEAD"):
        result = tools.run(call("write_file", path=raw, content="x"))
        assert result.status is ToolStatus.INVALID and "protected" in result.output


def test_write_creates_parent_directories(tools: FileTools, root: Path):
    result = tools.run(call("write_file", path="deep/nested/file.txt", content="hi"))
    assert result.status is ToolStatus.OK
    assert (root / "deep" / "nested" / "file.txt").read_bytes() == b"hi"
    assert result.mutated_paths == ("deep/nested/file.txt",)


def test_extra_read_roots_allow_reads_but_not_writes(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "e.txt").write_text("hello", encoding="utf-8")
    tools = FileTools(root, extra_read_roots=[extra])
    read = tools.run(call("read_file", path=str(extra / "e.txt")))
    assert read.status is ToolStatus.OK and "hello" in read.output
    assert "e.txt" in tools.read_hashes
    write = tools.run(call("write_file", path=str(extra / "e.txt"), content="x"))
    assert write.status is ToolStatus.INVALID


def test_list_files_default_walk_inventory(tools: FileTools, root: Path):
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.pyc").write_bytes(b"x")
    result = tools.run(call("list_files"))
    assert result.status is ToolStatus.OK
    assert result.data["backend"] == "walk"
    assert "a.txt" in result.output and "sub/b.py" in result.output
    assert "__pycache__/junk.pyc" not in result.output


def test_list_files_never_spawns_git(tools: FileTools, monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):
        raise AssertionError("files.py must not spawn processes")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    assert tools.run(call("list_files")).status is ToolStatus.OK


def test_list_files_uses_injected_inventory(root: Path):
    seen: list[Path] = []

    def fake(base: Path) -> list[str]:
        seen.append(base)
        return ["z.txt", "a.txt", "a.txt", "sub/b.py"]

    result = FileTools(root, inventory=fake).run(call("list_files"))
    assert seen == [root.resolve()]
    assert result.data["backend"] == "inventory"
    assert result.output.splitlines() == ["a.txt", "sub/b.py", "z.txt"]


def test_list_files_filters_by_path_and_glob(tools: FileTools, root: Path):
    result = tools.run(call("list_files", path="sub", glob="*.py"))
    assert result.status is ToolStatus.OK
    assert result.output.splitlines() == ["sub/b.py"]
    missing = tools.run(call("list_files", path="void"))
    assert missing.status is ToolStatus.INVALID
    not_a_dir = tools.run(call("list_files", path="a.txt"))
    assert not_a_dir.status is ToolStatus.INVALID


def test_list_files_caps_entries(tools: FileTools, root: Path):
    for index in range(2005):
        (root / f"f{index}.txt").write_text("x", encoding="utf-8")
    result = tools.run(call("list_files"))
    assert result.status is ToolStatus.OK
    assert result.data["count"] == 2000 and result.data["truncated"]
    assert "[truncated: showing first 2000 of" in result.output


def test_unknown_tool_and_bad_arguments_are_invalid(tools: FileTools):
    assert tools.run(call("delete_everything")).status is ToolStatus.INVALID
    bad = tools.run(call("read_file", path="a.txt", start_line=0))
    assert bad.status is ToolStatus.INVALID
    extra = tools.run(call("read_file", path="a.txt", nonsense=True))
    assert extra.status is ToolStatus.INVALID


@pytest.mark.skipif(os.name != "posix", reason="file modes are meaningful on POSIX only")
def test_write_preserves_mode(tools: FileTools, root: Path):
    target = root / "script.sh"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o755)
    digest = tools.run(call("read_file", path="script.sh")).data["sha256"]
    result = tools.run(call("write_file", path="script.sh", content="new\n", expected_sha=digest))
    assert result.status is ToolStatus.OK
    assert oct(target.stat().st_mode & 0o777) == "0o755"


@pytest.mark.parametrize(
    "raw",
    [".GIT/hooks/pre-commit", ".Git/config", ".NEDSTER/config.toml", ".nedster./x", "sub/.NedSter/x",
     "NEDSTE~1/config.toml", "GIT~1/config", ".nedster:stream", ".git::$INDEX_ALLOCATION/x"],  # fmt: skip
)
def test_protected_dirs_casefold_aliases_and_streams(tools: FileTools, root: Path, raw: str):
    result = tools.run(call("write_file", path=raw, content="x"))
    assert result.status is ToolStatus.INVALID and "protected" in result.output
    assert not (root / ".NEDSTER").exists() and not (root / ".nedster").exists()


def test_protection_does_not_block_lookalikes(tools: FileTools, root: Path):
    for raw in ("gitignore.txt", "my~1.txt", ".github/ci.yml", "notes.git"):
        assert tools.run(call("write_file", path=raw, content="x")).status is ToolStatus.OK


def test_read_refuses_oversize_without_reading(root: Path):
    big = root / "big.log"
    big.write_bytes(b"a\n" * 100)
    tools = FileTools(root, max_file_bytes=50)
    result = tools.run(call("read_file", path="big.log"))
    assert result.status is ToolStatus.INVALID and "too large" in result.output
    edit = tools.run(call("edit_file", path="big.log", old="a", new="b", expected_sha="0" * 16))
    assert edit.status is ToolStatus.INVALID and "too large" in edit.output
    write = tools.run(call("write_file", path="big.log", content="x", expected_sha="0" * 16))
    assert write.status is ToolStatus.INVALID and "too large" in write.output


def test_read_never_reads_more_than_cap_plus_one(root: Path, monkeypatch: pytest.MonkeyPatch):
    from ironclad.tool_runtime import files as files_module

    target = root / "grow.txt"
    target.write_bytes(b"x" * 10)
    monkeypatch.setattr(files_module.Path, "stat", lambda self, **kw: type("S", (), {"st_mode": 0o100644, "st_size": 1})())
    with pytest.raises(files_module.ToolError, match="too large"):
        files_module.read_regular(target, "grow.txt", 5)  # size lied; streaming cap still holds


def test_read_directory_and_special_files_are_invalid(tools: FileTools, root: Path):
    assert tools.run(call("read_file", path="sub")).status is ToolStatus.INVALID
    fifo = getattr(os, "mkfifo", None)
    if fifo is None:
        pytest.skip("no mkfifo on this platform")
    fifo(root / "pipe")
    result = tools.run(call("read_file", path="pipe"))
    assert result.status is ToolStatus.INVALID and "not a regular file" in result.output


def test_read_hash_is_whole_file_sha(tools: FileTools, root: Path):
    data = b"line\n" * 50_000
    (root / "many.txt").write_bytes(data)
    result = tools.run(call("read_file", path="many.txt"))
    assert result.data["sha256"] == hashlib.sha256(data).hexdigest()
