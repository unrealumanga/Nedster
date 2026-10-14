from __future__ import annotations

import ast
from pathlib import Path

from evals.companion import helpers

LOG_TEXT = "normalize_sku failed for"

SMOKE = r'''
import importlib, pkgutil, logging, io
import shop
for m in pkgutil.iter_modules(shop.__path__):
    importlib.import_module("shop." + m.name)
ns = {}
exec("from shop.skus import *", ns)
assert "canonical_sku" in ns, "__all__ must export canonical_sku"
assert "is_valid_sku" in ns
assert ns["canonical_sku"](" ab_123 ") == "AB-123"

stream = io.StringIO()
handler = logging.StreamHandler(stream)
logging.getLogger("shop.skus").addHandler(handler)
try:
    ns["canonical_sku"]("nope")
except ValueError:
    pass
else:
    raise AssertionError("invalid sku must raise")
assert "normalize_sku failed for 'nope'" in stream.getvalue(), "log warning text changed"

from shop import cli
import contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    assert cli.main(["lookup", "ab_123"]) == 0
    assert cli.main(["order", "ab_123:100", "wsh 310:2"]) == 0
out = buf.getvalue()
assert "AB-123 Bolt M6 0.25" in out and "total 23.85" in out, out
print("ok")
'''


def check(root: Path, answer: str) -> tuple[bool, str]:
    stale = []
    for source in helpers.original(__file__, '').rglob('*.py'):
        if not (root / source.relative_to(helpers.original(__file__, ''))).is_file():
            return False, f"original module/test deleted: {source.name}"
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        nodes = ast.walk(ast.parse(text))
        if any((isinstance(n, ast.Name) and n.id == 'normalize_sku') or
               (isinstance(n, ast.Attribute) and n.attr == 'normalize_sku') or
               (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'normalize_sku') or
               (isinstance(n, ast.alias) and 'normalize_sku' in n.name.split('.')) for n in nodes):
            stale.append(rel)
    if stale:
        return False, f"old name still present in: {', '.join(stale)}"
    changelog = (root / "CHANGELOG.md").read_bytes()
    if changelog != helpers.original(__file__, "CHANGELOG.md").read_bytes():
        return False, "CHANGELOG.md was modified"
    ok, detail = helpers.run_py(root, SMOKE)
    if not ok:
        return False, detail
    result, detail = helpers.unittest_result(root)
    return bool(result and result['ok'] and result['ran'] > result['skipped']), detail
