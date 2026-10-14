"""Headless eval runner: python -m evals.run --model nedster-worker [--engine native|legacy] [--only a,b]."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evals.score import Score, Task, claimed_success, load_check, load_task, score_run, snapshot

TOYREPOS = Path(__file__).parent / "toyrepos"


@dataclass(slots=True)
class RunStats:
    answer: str = ""
    outcome: str | None = None
    passed: bool | None = None
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    commands: list[str] = field(default_factory=list)
    error: str = ""


def parse_events(stdout: str) -> RunStats:
    """Fold a `nedster run --jsonl` stream (native EngineEvents or legacy start/result/end) into stats."""
    stats = RunStats()
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("kind") or event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else event
        if kind == "model.usage":
            stats.model_calls += 1
            stats.prompt_tokens += int(data.get("prompt_tokens", 0) or 0)
            stats.completion_tokens += int(data.get("completion_tokens", 0) or 0)
        elif kind == "tool.started":
            stats.tool_calls += 1
            stats.commands.append(json.dumps(data.get("arguments", ""), default=str))
        elif kind == "turn.finished":
            stats.answer = str(data.get("answer", stats.answer))
            stats.outcome = str(data.get("outcome", "")) or stats.outcome
        elif kind == "acceptance" and stats.outcome is None:
            stats.outcome = str(data.get("outcome", "")) or None
        elif kind == "error":
            stats.error = str(data.get("message", ""))
        elif kind == "result":  # legacy
            stats.answer = str(event.get("answer", ""))
            stats.passed = bool(event.get("passed"))
    return stats


def _invoke(task: Task, workdir: Path, model: str | None, engine: str | None, timeout: float) -> tuple[RunStats, str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if model:
        for role in ("router", "coder", "reasoner", "critic"):
            env[f"NEDSTER_MODELS__{role.upper()}"] = model
    if engine:
        env["NEDSTER_ENGINE"] = engine
    argv = [sys.executable, "-m", "nedster_agent", "--project", str(workdir), "run", "--jsonl", task.prompt]
    try:
        proc = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stats = parse_events(exc.stdout if isinstance(exc.stdout, str) else "")
        stats.error = f"timeout after {timeout:.0f}s"
        return stats, "timeout"
    stats = parse_events(proc.stdout)
    if not stats.error and proc.returncode not in (0, 1):
        stats.error = (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
    return stats, "ok"


def run_task(task: Task, model: str | None, engine: str | None, timeout: float) -> dict:
    missing = [tool for tool in task.requires if shutil.which(tool) is None]
    if missing:
        return {"task": task.name, "status": "skipped", "reason": f"missing {', '.join(missing)}"}
    with tempfile.TemporaryDirectory(prefix=f"eval_{task.name}_") as tmp:
        workdir = Path(tmp) / "repo"
        shutil.copytree(task.repo, workdir)
        before = snapshot(workdir)
        started = time.monotonic()
        stats, _ = _invoke(task, workdir, model, engine, timeout)
        wall = time.monotonic() - started
        after = snapshot(workdir)
        score = score_run(task, before, after, answer=stats.answer,
                          claimed=claimed_success(stats.outcome, stats.passed),
                          check=load_check(task), root=workdir, commands=stats.commands)
    return _record(task, score, stats, wall)


def _record(task: Task, score: Score, stats: RunStats, wall: float) -> dict:
    return {
        "task": task.name, "status": "pass" if score.passed else "fail", "score": asdict(score),
        "false_success": score.false_success, "outcome": stats.outcome, "wall_s": round(wall, 2),
        "model_calls": stats.model_calls, "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens, "tool_calls": stats.tool_calls, "error": stats.error,
    }


def format_table(records: list[dict]) -> str:
    rows = [("task", "status", "claimed", "calls", "ptok", "ctok", "tools", "wall_s")]
    for r in records:
        if r["status"] == "skipped":
            rows.append((r["task"], "skipped", "-", "-", "-", "-", "-", "-"))
            continue
        claimed = "FALSE-SUCCESS" if r["false_success"] else (r["outcome"] or "-")
        rows.append((r["task"], r["status"], claimed, str(r["model_calls"]), str(r["prompt_tokens"]),
                     str(r["completion_tokens"]), str(r["tool_calls"]), str(r["wall_s"])))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    ran = [r for r in records if r["status"] != "skipped"]
    totals = (f"total: {sum(r['status'] == 'pass' for r in ran)}/{len(ran)} passed, "
              f"{sum(r['false_success'] for r in ran)} false-success, "
              f"{sum(r['model_calls'] for r in ran)} model calls, "
              f"{sum(r['prompt_tokens'] for r in ran)} prompt tok, {sum(r['completion_tokens'] for r in ran)} completion tok, "
              f"{round(sum(r['wall_s'] for r in ran), 1)}s")
    return "\n".join([*lines, totals])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.run", description=__doc__)
    parser.add_argument("--model", default=None, help="Model name applied to every role")
    parser.add_argument("--engine", choices=("native", "legacy"), default=None)
    parser.add_argument("--only", default="", help="Comma-separated fixture name prefixes, e.g. a,b or a_add_subtract")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds per run")
    parser.add_argument("--out", type=Path, default=None, help="Write one JSON record per run (jsonl)")
    args = parser.parse_args(argv)

    prefixes = [p.strip() for p in args.only.split(",") if p.strip()]
    dirs = sorted(d for d in TOYREPOS.iterdir() if (d / "task.json").is_file())
    if prefixes:
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in prefixes)]
    if not dirs:
        print("no fixtures selected", file=sys.stderr)
        return 2
    records: list[dict] = []
    for _ in range(max(1, args.repeat)):
        for directory in dirs:
            records.append(run_task(load_task(directory), args.model, args.engine, args.timeout))
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(format_table(records))
    return 1 if any(r.get("false_success") for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
