"""Offline validation for the wave-2 eval fixtures (W3-EVALSPLUS).

For every fixture: copy repo to a temp dir, run its check on the untouched copy
(must FAIL, except k/l which must PASS), then apply the intended fix to a fresh
copy and run the check again (must PASS). Nothing here invokes a model or the
network. Run:  .venv\\Scripts\\python .overhaul/wave2/validate_fixtures.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.score import load_check, load_task  # noqa: E402

TOY = ROOT / "evals" / "toyrepos"
EXPLICIT_ANSWERS = {"k_question_with_edit_verb": "Use functools.lru_cache on fetch()."}
PASS_UNTOUCHED = {"k_question_with_edit_verb", "l_already_done"}


def _read(repo: Path, name: str) -> str:
    return (repo / name).read_text(encoding="utf-8")


def _write(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")


def fix_i(repo: Path) -> None:
    _write(repo, "geometry.py", _read(repo, "geometry.py").replace("return width + height", "return width * height"))


def fix_j(repo: Path) -> None:
    _write(repo, "models.py",
           "from dataclasses import asdict, dataclass\n\n\n@dataclass\nclass Item:\n    name: str\n    count: int\n\n"
           "    def to_dict(self):\n        return asdict(self)\n")
    _write(repo, "format.py",
           "import json\n\n\ndef format_item(item):\n    return f\"{item.name}: {item.count}\"\n\n\n"
           "def format_item_json(item):\n    return json.dumps(item.to_dict())\n")
    _write(repo, "cli.py",
           "import argparse\n\nfrom format import format_item, format_item_json\nfrom models import Item\n\n\n"
           "def build_parser():\n    parser = argparse.ArgumentParser(prog=\"inventory\")\n"
           "    parser.add_argument(\"--name\", default=\"widget\")\n"
           "    parser.add_argument(\"--count\", type=int, default=3)\n"
           "    parser.add_argument(\"--json\", action=\"store_true\")\n    return parser\n\n\n"
           "def main(argv=None):\n    args = build_parser().parse_args(argv)\n    item = Item(args.name, args.count)\n"
           "    print(format_item_json(item) if args.json else format_item(item))\n    return 0\n\n\n"
           "if __name__ == \"__main__\":\n    raise SystemExit(main())\n")


def fix_m(repo: Path) -> None:
    text = _read(repo, "app.py")
    text = text.replace("def greet(name):\n", 'def greet(name):\n    """Return a greeting for name."""\n')
    _write(repo, "app.py", text)


def fix_n(repo: Path) -> None:
    _write(repo, "mod_17.py", _read(repo, "mod_17.py").replace("(int(year), int(day), int(month))",
                                                             "(int(year), int(month), int(day))"))


def fix_o(repo: Path) -> None:
    _write(repo, "données.py", _read(repo, "données.py").replace("def accueil(", "def bienvenue("))


def fix_p(repo: Path) -> None:
    _write(repo, "index.js", _read(repo, "index.js").replace("i < n", "i <= n"))


FIXES = {
    "i_failing_test_fix": fix_i,
    "j_multi_file_feature": fix_j,
    "k_question_with_edit_verb": lambda repo: None,
    "l_already_done": lambda repo: None,
    "m_injection_in_code_comment": fix_m,
    "n_large_search": fix_n,
    "o_unicode_paths": fix_o,
    "p_js_fix": fix_p,
}


def _run_check(directory: Path, work: Path, name: str) -> tuple[bool, str]:
    task = load_task(directory)
    check = load_check(task)
    return check(work, EXPLICIT_ANSWERS.get(name, ""))


def validate(name: str) -> list[str]:
    directory = TOY / name
    problems: list[str] = []

    with tempfile.TemporaryDirectory(prefix=f"val_{name}_") as tmp:
        base = Path(tmp)

        untouched = base / "untouched"
        shutil.copytree(directory / "repo", untouched)
        ok, detail = _run_check(directory, untouched, name)
        want_ok = name in PASS_UNTOUCHED
        if ok != want_ok:
            label = "PASS" if want_ok else "FAIL"
            problems.append(f"untouched check should {label}, got ok={ok}: {detail}")
        print(f"{name:28} untouched ok={ok!s:5} (expected {want_ok!s:5}) : {detail[:80]}")

        fixed = base / "fixed"
        shutil.copytree(directory / "repo", fixed)
        FIXES[name](fixed)
        ok2, detail2 = _run_check(directory, fixed, name)
        if not ok2:
            problems.append(f"fixed check should PASS, got ok=False: {detail2}")
        print(f"{name:28} fixed     ok={ok2!s:5} (expected True ) : {detail2[:80]}")

    return problems


def main() -> int:
    failures: list[str] = []
    for name in FIXES:
        failures.extend(validate(name))
    print("\n" + ("ALL FIXTURES VALIDATED" if not failures else "FAILURES:"))
    for problem in failures:
        print(f"  - {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
