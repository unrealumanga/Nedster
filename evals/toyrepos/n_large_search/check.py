from __future__ import annotations

from pathlib import Path

from evals import checklib

_FIND_AND_CALL = """\
import glob, importlib.util, pathlib
matches = [p for p in sorted(glob.glob('*.py'))
           if 'def parse_date' in pathlib.Path(p).read_text(encoding='utf-8')]
assert len(matches) == 1, f'expected exactly one parse_date module, found {matches}'
spec = importlib.util.spec_from_file_location('bugmod', matches[0])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.parse_date('2024-03-15') == (2024, 3, 15), mod.parse_date('2024-03-15')
assert mod.parse_date('1999-12-31') == (1999, 12, 31)
"""


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_python(root, _FIND_AND_CALL)
    if not ok:
        return False, out
    return True, "parse_date returns (year, month, day)"
