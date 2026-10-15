from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

from evals.companion import helpers


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outside(raw: bytes) -> list[bytes]:
    nodes = [n for n in ast.parse(raw.decode("utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name == "shipping_cost"]
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise SyntaxError("expected exactly one undecorated shipping_cost")
    node = nodes[0]
    lines = raw.splitlines(keepends=True)
    return lines[: node.lineno - 1] + lines[node.end_lineno:]


def check(root: Path, answer: str) -> tuple[bool, str]:
    orig_path = helpers.original(__file__, "engine.py")
    new_path = root / "engine.py"
    try:
        old_rest = _outside(orig_path.read_bytes())
        new_rest = _outside(new_path.read_bytes())
    except (StopIteration, SyntaxError, OSError) as exc:
        return False, f"engine.py unreadable or shipping_cost missing: {exc}"
    if old_rest != new_rest:
        return False, "bytes outside shipping_cost() changed"

    old = _load(orig_path, "engine_orig")
    new = _load(new_path, "engine_new")
    for zone in ("LOCAL", "EU", "WORLD"):
        for weight in (0.5, 1, 7.25, 19.99, 20, 20.0001, 20.01, 25, 100):
            want = round(old.shipping_cost(weight, zone) + (15.0 if weight > 20 else 0.0), 2)
            got = new.shipping_cost(weight, zone)
            if not math.isfinite(got) or abs(got - want) > 1e-9:
                return False, f"shipping_cost({weight}, {zone!r}) = {got}, expected {want}"
    for args in ((0, "EU"), (-1, "EU"), (5, "MARS")):
        try:
            new.shipping_cost(*args)
        except ValueError:
            continue
        return False, f"shipping_cost{args} should still raise ValueError"
    quote = new.quote_total(100, 30, "EU")
    if not math.isfinite(quote) or abs(quote - round(100 + old.shipping_cost(30, "EU") + 15, 2)) > 1e-9:
        return False, "quote_total no longer includes the surcharge"
    return True, "surcharge applied; the rest of the file is untouched"
