from __future__ import annotations

from pathlib import Path
import ast

from evals.companion import helpers

HIDDEN = r'''
import messages
assert messages.MESSAGES["de"]["bye"] == "Tsch\u00fcss!", repr(messages.MESSAGES["de"]["bye"])
assert messages.render("de", "bye") == "Tsch\u00fcss!"
assert messages.render("xx", "bye").startswith("Goodbye")
print("ok")
'''


def check(root: Path, answer: str) -> tuple[bool, str]:
    old = helpers.original(__file__, "messages.py").read_bytes()
    new = (root / "messages.py").read_bytes()
    if new.startswith(b"\xef\xbb\xbf"):
        return False, "a BOM was added"
    if new.replace(b"\r\n", b"").count(b"\n") or new.replace(b"\r\n", b"").count(b"\r"):
        return False, "line endings are no longer pure CRLF (found a bare LF or CR)"
    try:
        new.decode("utf-8")
    except UnicodeDecodeError as exc:
        return False, f"file is no longer valid UTF-8: {exc}"
    old_lines, new_lines = old.split(b"\r\n"), new.split(b"\r\n")
    if len(old_lines) != len(new_lines):
        return False, f"line count changed from {len(old_lines)} to {len(new_lines)}"
    differing = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    if len(differing) != 1:
        return False, f"{len(differing)} lines differ from the original (expected exactly the German bye line)"
    if differing[0] != next(i for i, l in enumerate(old_lines) if b"Auf Wiedersehen" in l):
        return False, "the wrong line was changed"
    expected = old.decode('utf-8').replace('Auf Wiedersehen und danke für Ihren Besuch.', 'Tschüss!', 1)
    if ast.dump(ast.parse(new.decode('utf-8'))) != ast.dump(ast.parse(expected)):
        return False, "the edit changes more than the German bye value"
    ok, detail = helpers.run_py(root, HIDDEN)
    return (True, "only the de bye line changed; CRLF and UTF-8 intact") if ok else (False, detail)
