from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    code = (
        'import importlib; m = importlib.import_module("donn\\u00e9es"); '
        'assert hasattr(m, "bienvenue"), "bienvenue missing"; '
        'assert not hasattr(m, "accueil"), "accueil still present"; '
        'assert m.bienvenue("Ada") == "Bonjour, Ada ! \\U0001f389"'
    )
    ok, out = checklib.run_python(root, code)
    if not ok:
        return False, out
    return True, "renamed and still returns the accented greeting"
