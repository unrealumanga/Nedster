from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Heavy modules (httpx, rich, textual, asyncio, the engine) are imported inside command
# functions so `nedster --help` and `nedster config show` stay fast.
from . import __version__

EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, EXIT_NOT_READY = 0, 1, 2, 3


class UsageError(Exception):
    """Bad invocation (exit 2)."""


# Shared help strings for global flags, used in both root parser and subcommand parsers.
GLOBAL_HELP: dict[str, str] = {
    "--project": "Project folder (default: current folder)",
    "--config": "Nedster TOML configuration path",
    "--json": "Emit one machine-readable JSON document",
    "--plain": "Emit stable, one-record-per-line text",
    "--no-tui": "Never open the full-screen interface",
    "--color": "Force color when the destination supports it",
    "--no-color": "Disable color output",
    "--ascii": "Use ASCII-safe symbols",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nedster",
        description="Nedster — a local-first AI coding and automation agent",
    )
    parser.add_argument("--project", default=".", help=GLOBAL_HELP["--project"])
    parser.add_argument("--config", default=None, help=GLOBAL_HELP["--config"])
    for flag in list(GLOBAL_HELP)[2:]:
        parser.add_argument(flag, action="store_true", help=GLOBAL_HELP[flag])
    parser.add_argument("--version", action="version", version=f"Nedster {__version__}")
    subcommands = parser.add_subparsers(dest="command")

    def accept_global_options(command_parser: argparse.ArgumentParser) -> None:
        for flag, help_text in GLOBAL_HELP.items():
            if flag == "--project" or flag == "--config":
                command_parser.add_argument(flag, default=argparse.SUPPRESS, help=help_text)
            else:
                command_parser.add_argument(
                    flag, action="store_true", default=argparse.SUPPRESS, help=help_text
                )

    setup = subcommands.add_parser("setup", help="Install/adopt Ollama and configure recommended models")
    accept_global_options(setup)
    setup.add_argument("--profile", choices=("fast", "balanced", "maximum"))
    setup.add_argument("--ollama-url", default=None, help="Ollama endpoint (default: resolved configuration)")
    setup.add_argument("--skip-ollama-install", action="store_true")
    setup.add_argument("--with-embeddings", action="store_true", help="Also download the embedding model")
    setup.add_argument("--yes", action="store_true", help="Accept the recommended configuration")
    setup.add_argument("--no-input", action="store_true", help="Never prompt; use flags and defaults")
    setup.add_argument("--dry-run", action="store_true", help="Show actions without downloading or writing")
    setup.add_argument("--allow-execution", action="store_true",
                       help="Let Nedster run tests and commands automatically, inside an isolated sandbox only")
    setup.add_argument("--allow-host-execution", action="store_true",
                       help="Let Nedster run commands directly on this machine, with NO isolation (implies execute)")

    doctor = subcommands.add_parser("doctor", help="Check inference, read-only and execution readiness")
    accept_global_options(doctor)
    models = subcommands.add_parser("models", help="Inspect or install the local model fleet")
    accept_global_options(models)
    models.add_argument("action", nargs="?", choices=("list", "recommend", "install"), default="list")
    models.add_argument("--profile", choices=("fast", "balanced", "maximum"), default="balanced")
    models.add_argument("--dry-run", action="store_true")

    run = subcommands.add_parser("run", help="Run one objective and exit")
    accept_global_options(run)
    run.add_argument("objective", nargs="?", help="Objective (default: read from piped stdin)")
    run.add_argument("--jsonl", action="store_true", help="Emit an event stream, one JSON object per line")
    run.add_argument(
        "--engine",
        choices=("native",),
        default=None,
        help="DEPRECATED (kept for one release): the native engine is the only engine; this flag does nothing",
    )
    run.add_argument("--model", default=None, help="Worker model for the native engine (default: configuration)")
    run.add_argument(
        "--expect",
        choices=("change", "answer"),
        default=None,
        help="What a successful run must do: change files, or only answer (default: inferred from the objective)",
    )
    run.add_argument("--continue", dest="resume_latest", action="store_true", help="Continue the latest session here")
    run.add_argument("--session", default=None, help="Continue a specific session id")
    tui = subcommands.add_parser("tui", help="Open the interactive terminal interface")
    accept_global_options(tui)
    ledger = subcommands.add_parser("verify-ledger", help="Verify the local event chain")
    accept_global_options(ledger)
    permissions = subcommands.add_parser("permissions", help="Show the active tool permission policy")
    accept_global_options(permissions)
    config = subcommands.add_parser("config", help="Inspect the effective configuration")
    accept_global_options(config)
    config.add_argument("action", nargs="?", choices=("show", "get", "path"), default="show")
    config.add_argument("key", nargs="?", default=None, help="Dotted key for get, e.g. models.coder")

    sessions = subcommands.add_parser(
        "sessions", help="List, show, delete or prune the saved sessions of this project"
    )
    accept_global_options(sessions)
    session_actions = sessions.add_subparsers(dest="action")
    sessions.set_defaults(action="list")  # `nedster sessions` alone lists (argparse has no subparser default)
    sessions_list = session_actions.add_parser("list", help="Saved sessions, newest first (default)")
    accept_global_options(sessions_list)
    sessions_show = session_actions.add_parser("show", help="One session: metadata + stored transcript")
    accept_global_options(sessions_show)
    sessions_show.add_argument("session_id", help="Session id or unique prefix")
    sessions_delete = session_actions.add_parser(
        "delete", help="Delete a session; refused while the session is active (locked)"
    )
    accept_global_options(sessions_delete)
    sessions_delete.add_argument("session_id", help="Session id or unique prefix")
    sessions_prune = session_actions.add_parser(
        "prune", help="Delete tool outputs older than a cutoff that no message or task state references"
    )
    accept_global_options(sessions_prune)
    sessions_prune.add_argument(
        "--older-than",
        dest="older_than",
        type=float,
        required=True,
        metavar="DAYS",
        help="Only outputs older than DAYS are candidates (required)",
    )
    sessions_prune.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted; change nothing",
    )
    return parser


# --- helpers -----------------------------------------------------------------------------------


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _emit_json(value: Any) -> None:
    print(json.dumps(value, default=str))


def _console(args: argparse.Namespace) -> Any:
    from rich.console import Console

    if args.color and args.no_color:
        raise UsageError("Choose either --color or --no-color, not both.")
    no_color: bool | None = False if args.color else (True if args.no_color else None)
    return Console(no_color=no_color, force_terminal=False if args.no_color else None)


def _stderr_console(args: argparse.Namespace) -> Any:
    from rich.console import Console

    no_color: bool | None = False if args.color else (True if args.no_color else None)
    return Console(stderr=True, no_color=no_color)


def _ascii(value: str, enabled: bool) -> str:
    if not enabled:
        return value
    return value.replace("…", "...").replace("·", "|").replace("—", "-")


def _launcher_path() -> str | None:
    names = ("nedster.exe", "nedster") if sys.platform == "win32" else ("nedster",)
    for raw_directory in os.getenv("PATH", "").split(os.pathsep):
        directory = Path(raw_directory.strip('"'))
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def _resolve(args: argparse.Namespace, overrides: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
    from ironclad.config import resolve_config

    try:
        return resolve_config({"config": args.config, **(overrides or {})}, args.project)
    except (FileNotFoundError, ValueError) as exc:
        raise UsageError(str(exc)) from exc


def _config_dict(config: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), default=str))


# --- commands ----------------------------------------------------------------------------------


def _config_show(args: argparse.Namespace) -> int:
    config, provenance = _resolve(args)
    data = _config_dict(config)
    if args.json:
        _emit_json({"config": data, "provenance": provenance})
        return EXIT_OK
    from ironclad.config import config_dir

    print(f"# user config dir: {config_dir()}")
    for key in sorted(provenance):
        node: Any = data
        for part in key.split("."):
            node = node[part]
        value = json.dumps(node) if not isinstance(node, str) else node
        print(f"{key} = {value}  [{provenance[key]}]")
    for warning in config.warnings:
        print(f"# warning: {warning}")
    return EXIT_OK


def _config(args: argparse.Namespace) -> int:
    action = getattr(args, "action", None) or "show"
    if action == "show":
        return _config_show(args)
    if action == "get":
        return _config_get(args)
    if action == "path":
        return _config_path(args)
    raise UsageError(f"unknown config action: {action}")


def _config_get(args: argparse.Namespace) -> int:
    key = getattr(args, "key", None)
    if not key:
        raise UsageError("Usage: nedster config get <section.field>, e.g. models.coder")
    config, provenance = _resolve(args)
    data = _config_dict(config)
    node: Any = data
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise UsageError(f"Unknown config key {key!r}. See: nedster config show")
    if args.json:
        source = provenance.get(key, "default")
        _emit_json({"key": key, "value": node, "source": source})
    else:
        if isinstance(node, str):
            print(node)
        else:
            print(json.dumps(node))
    return EXIT_OK


def _config_path(args: argparse.Namespace) -> int:
    from ironclad.config import config_dir, data_dir

    user_cfg = config_dir() / "config.toml"
    state_d = data_dir()
    project_root = Path(args.project or ".").resolve()
    proj_cfg = project_root / ".nedster" / "config.toml"

    if args.json:
        _emit_json({
            "user_config": str(user_cfg),
            "project_config": str(proj_cfg),
            "data_dir": str(state_d),
        })
        return EXIT_OK
    user_suffix = " (exists)" if user_cfg.exists() else " (not created yet)"
    proj_suffix = " (exists)" if proj_cfg.exists() else " (not created yet)"
    print(f"user config: {user_cfg}{user_suffix}")
    print(f"project config: {proj_cfg}{proj_suffix}")
    print(f"data dir: {state_d}")
    return EXIT_OK


def _read_line(prompt: str) -> str:
    return input(prompt)


def _ask_yes_no(question: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = _read_line(f"{question} {suffix} ").strip().lower()
    except EOFError:
        return default
    return default if not answer else answer in ("y", "yes")


def _isolated(probe: dict[str, Any]) -> bool:
    return bool(probe["ready"]) and probe["isolation"] == "full"


def _decide_execution(args: argparse.Namespace, probe: dict[str, Any], interactive: bool, say: Any) -> Any:
    """Turn flags / answers into an ExecutionChoice (None = keep what the config already says)."""
    from .model_manager import ExecutionChoice

    if args.allow_host_execution:
        say("WARNING: commands will run directly on this machine with NO isolation. "
            "Undo with: sandbox.allow_host_execution = false in config.toml")
        return ExecutionChoice("execute", True)
    isolated = _isolated(probe)
    if args.allow_execution or (args.yes and not interactive):
        if isolated:
            return ExecutionChoice("execute")
        if args.allow_execution:
            say(f"Execution NOT enabled: no isolated sandbox is ready ({probe['detail']}). "
                "Start Docker/install a sandbox, or use --allow-host-execution to accept no isolation.")
        return None
    if not interactive:
        return None
    if probe["ready"]:
        question = (f"Let Nedster run tests and commands automatically inside {probe['backend']} "
                    f"({probe['isolation']} isolation)?")
        if _ask_yes_no(question, default=isolated):
            return ExecutionChoice("execute")
    else:
        say(f"No execution backend is ready: {probe['detail']}")
    if not isolated:
        say("Without a sandbox, commands would run with your full user permissions and could change or "
            "delete anything you can. Change it later: sandbox.allow_host_execution in config.toml.")
        if _ask_yes_no("No sandbox is available. Allow commands to run directly on this machine (no isolation)?",
                       default=False):
            return ExecutionChoice("execute", True)
    return ExecutionChoice("write", False)


def _setup(args: argparse.Namespace) -> int:
    import httpx  # noqa: F401  (surface import problems early, and for error mapping)
    from rich import box
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    from ironclad.hardware import choose_profile, probe_hardware

    from .model_manager import SetupError, describe_permissions, install_plan, probe_execution, recommend_plan

    console = _console(args)
    config, _ = _resolve(args, {"models.ollama_url": args.ollama_url})
    snapshot = probe_hardware()
    tier = choose_profile(snapshot).name
    interactive = sys.stdin.isatty() and not args.no_input and not args.yes and not args.json
    preference = args.profile or "balanced"
    recommended = recommend_plan(snapshot, preference)
    human = not args.json and not args.plain

    if human:
        console.print(Panel.fit("[bold]Nedster setup[/bold]\nLocal models, selected for this computer",
                                border_style="cyan", box=box.ASCII if args.ascii else box.ROUNDED))
        console.print(_ascii(f"Step 1/4  Hardware  {snapshot.accelerator} · {snapshot.vram_total_gb:.1f} GB VRAM · "
                             f"{snapshot.ram_total_gb:.1f} GB RAM", args.ascii))
        console.print(f"Step 2/4  Recommendation  {tier} / {recommended.label.lower()}")
        table = Table(show_header=False, box=None)
        table.add_row("Worker", recommended.worker)
        table.add_row("Router", recommended.router)
        table.add_row("Memory", recommended.embedding + ("" if args.with_embeddings else " (skipped; --with-embeddings)"))
        table.add_row("Endpoint", config.models.ollama_url)
        table.add_row("Why", recommended.notes)
        console.print(table)

    if interactive:
        preference = Prompt.ask("Choose a profile", choices=["fast", "balanced", "maximum"], default=preference)
        recommended = recommend_plan(snapshot, preference)
        if not Confirm.ask("Download and configure these models?", default=True):
            console.print("Setup cancelled; nothing was changed.")
            return EXIT_OK

    probe = probe_execution(config)
    execution = _decide_execution(args, probe, interactive, lambda message: _err(_ascii(message, args.ascii)))

    if human:
        console.print("Step 3/4  Ollama and models")
    progress_console = _stderr_console(args)
    try:
        result = install_plan(
            recommended,
            ollama_url=config.models.ollama_url,
            install_ollama=not args.skip_ollama_install,
            dry_run=args.dry_run,
            with_embeddings=args.with_embeddings,
            progress=lambda message: progress_console.print(f"  {_ascii(message, args.ascii)}", markup=False),
            execution=execution,
        )
    except SetupError as exc:
        _err(f"[FAIL] Setup stopped: {exc}")
        if exc.remedy:
            _err(f"Remedy: {exc.remedy}")
        _err("Your previous configuration was left untouched.")
        return EXIT_RUNTIME
    risk = execution.max_auto_risk if execution else config.policy.max_auto_risk
    if execution and execution.allow_host_execution is not None:
        host = execution.allow_host_execution
    else:
        host = config.sandbox.allow_host_execution
    permissions = describe_permissions(risk, host)
    result["permissions"] = {"max_auto_risk": risk, "allow_host_execution": bool(host), "summary": permissions}
    if args.json:
        _emit_json(result)
    elif args.plain:
        for key in ("config", "backup", "aliases"):
            print(f"{key}\t{result[key]}")
        print(f"permissions\t{permissions}")
    else:
        console.print("Step 4/4  Ready")
        if result.get("backup"):
            console.print(f"Previous config saved to {result['backup']}")
        launcher = _launcher_path()
        console.print(permissions, markup=False, highlight=False)
        console.print(f"Global command: {'ready at ' + launcher if launcher else 'not visible on PATH; restart the terminal'}")
        console.print("Run [bold]nedster[/bold] from any project folder to begin.")
    return EXIT_OK


def _doctor(args: argparse.Namespace) -> int:
    from ironclad.hardware import doctor_report

    from .model_manager import OllamaManager, context_report, readiness

    config, _ = _resolve(args)
    report = doctor_report()
    manager = OllamaManager(config.models.ollama_url)
    tiers = readiness(config, manager)
    context = context_report(config, manager) if tiers["inference"]["ready"] else None
    report["context"] = context
    from ironclad.config import config_dir

    config_path = config_dir() / "config.toml"
    report["nedster"] = {
        "version": __version__,
        "launcher": _launcher_path(),
        "config": str(config_path),
        "configured": config_path.exists(),
    }
    report["tiers"] = tiers
    report["config_warnings"] = list(config.warnings)
    ready = bool(tiers["inference"]["ready"])
    if args.json:
        _emit_json(report)
        return EXIT_OK if ready else EXIT_NOT_READY
    snapshot = report["hardware"]  # type: ignore[index]
    labels = {"inference": "inference-ready", "read_only": "read-only-ready", "execution": "execution-ready"}
    if args.plain:
        print(f"version\t{__version__}")
        print(f"accelerator\t{snapshot['accelerator']}")  # type: ignore[index]
        print(f"configured\t{str(config_path.exists()).lower()}")
        for key, label in labels.items():
            print(f"{label}\t{'yes' if tiers[key]['ready'] else 'no'}")
    else:
        console = _console(args)
        console.print(f"Nedster {__version__} · {report['profile']['name']}", highlight=False)  # type: ignore[index]
        console.print(_ascii(
            f"Hardware  {snapshot['accelerator']} · {snapshot['vram_total_gb']} GB VRAM · "  # type: ignore[index]
            f"{snapshot['ram_total_gb']} GB RAM", args.ascii), highlight=False)  # type: ignore[index]
        for key, label in labels.items():
            tier = tiers[key]
            mark = "ready" if tier["ready"] else "NOT READY"
            console.print(_ascii(f"{label:<16}{mark} · {tier.get('detail', '')}", args.ascii), highlight=False, markup=False)
        if context:
            kv = f"KV {context['kv_cache_type']} / flash attention {'on' if context['flash_attention'] else 'off'} (as seen by this shell; server unknown)"
            line = f"{'context':<16}model {context['num_ctx'] or '?'} tokens · engine budget {context['engine_budget']} · {kv}"
            console.print(_ascii(line, args.ascii), highlight=False, markup=False)
            for tip in context["advice"]:  # type: ignore[union-attr]
                console.print(_ascii(f"  tip: {tip}", args.ascii), highlight=False, markup=False)
    for warning in config.warnings:
        _err(f"config warning: {warning}")
    for key, label in labels.items():
        if not tiers[key]["ready"]:
            _err(f"{label}: {tiers[key].get('remedy', 'see detail above')}")
    return EXIT_OK if ready else EXIT_NOT_READY


def _models(args: argparse.Namespace) -> int:
    from .model_manager import OllamaManager, install_plan, recommend_plan

    config, _ = _resolve(args)
    manager = OllamaManager(config.models.ollama_url)
    plan = recommend_plan(preference=args.profile)
    if args.action == "recommend":
        if args.json:
            _emit_json(asdict(plan))
        else:
            for key, value in asdict(plan).items():
                print(f"{key}\t{value}")
        return EXIT_OK
    if args.action == "install":
        install_plan(plan, ollama_url=config.models.ollama_url, dry_run=args.dry_run, progress=_err)
        return EXIT_OK
    if not manager.api_available():
        _err(f"Ollama is not reachable at {manager.url}. Run `nedster setup`.")
        return EXIT_NOT_READY
    models = manager.list_models()
    if args.json:
        _emit_json({"models": models})
    else:
        for model in models:
            print(model)
    return EXIT_OK


def _read_objective(args: argparse.Namespace) -> str:
    objective = args.objective
    if not objective and not sys.stdin.isatty():
        objective = sys.stdin.read()
    objective = (objective or "").strip()
    if not objective:
        raise UsageError("No objective given. Pass it as an argument or pipe it on stdin.")
    return objective


async def _run_objective(args: argparse.Namespace, objective: str) -> int:
    config, _ = _resolve(args)
    return await _run_native(args, config, objective)


class _SessionHeartbeat:
    """Refreshes the activity lock of the session a `nedster run` is using, then releases it.

    Keeps `sessions delete` from (and against) a long CLI run in another terminal: while the lock is
    live the store refuses to delete the session, and the run notices a takeover and re-acquires.
    """

    def __init__(self, store: Any, session_id: str, owner: str, interval_s: float = 10.0) -> None:
        self._store, self._session_id, self._owner, self._interval = store, session_id, owner, interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="nedster-session-lock", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                renewed = bool(self._store.refresh_lock(self._session_id, self._owner))
                if not renewed:
                    renewed = bool(self._store.acquire_lock(self._session_id, self._owner))
            except Exception:  # noqa: BLE001 - a dead store ends the heartbeat, never the run
                return
            if not renewed:
                return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        with suppress(Exception):
            self._store.release_lock(self._session_id, self._owner)


def _lock_run_session(session: Any) -> _SessionHeartbeat | None:
    """Take the activity lock for a CLI run; None (with a stderr note) when it cannot be held."""
    from ironclad.session_store.sqlite_store import make_owner

    store = getattr(session, "store", None)
    acquire = getattr(store, "acquire_lock", None)
    session_id = str(getattr(session, "session_id", ""))
    if not callable(acquire) or not session_id:
        return None
    owner = make_owner("run")
    try:
        acquired = bool(acquire(session_id, owner))
    except Exception:  # noqa: BLE001 - never block a run on the lock layer
        return None
    if not acquired:
        _err(f"[warning] session {session_id[:8]} is active in another Nedster; continuing anyway")
        return None
    return _SessionHeartbeat(store, session_id, owner)


async def _run_native(args: argparse.Namespace, config: Any, objective: str) -> int:
    from dataclasses import asdict

    from ironclad.native import NativeSession

    def emit(event: Any) -> None:
        if args.jsonl:
            _emit_json({"kind": event.kind, "session_id": event.session_id, "run_id": event.run_id, "data": dict(event.data)})
        elif event.kind == "tool.finished" and not args.json:
            data = event.data
            _err(f"  {data.get('tool')} -> {data.get('status')} ({data.get('elapsed_ms', 0):.0f} ms)")

    session = NativeSession(
        config, emit=emit, model=args.model, session_id=args.session, resume_latest=args.resume_latest
    )
    heartbeat = _lock_run_session(session)
    try:
        from ironclad.engine.intent import parse_expect

        result = await session.run(objective, intent=parse_expect(getattr(args, "expect", None)))
    finally:
        if heartbeat is not None:
            heartbeat.close()  # releases the lock; before aclose() closes the store
        await session.aclose()
    if args.json:
        payload = asdict(result)
        payload["outcome"] = result.outcome.value
        payload["validated_at"] = result.validated_at.counter if result.validated_at else None
        payload["session_id"] = session.session_id
        _emit_json(payload)
    elif not args.jsonl:
        print(result.answer)
        _err(f"[{result.outcome.value} · session {session.session_id[:8]}]")
    return EXIT_OK if result.passed else EXIT_RUNTIME


def _verify_ledger(args: argparse.Namespace) -> int:
    from ironclad.ledger import EventLedger, load_or_create_ledger_key

    config, _ = _resolve(args)
    key = load_or_create_ledger_key(config.runtime.state_dir, config.runtime.ledger_key_env)
    with EventLedger(config.runtime.state_dir / "events.sqlite3", key) as ledger:
        valid, count, digest = ledger.verify()
    if args.json:
        _emit_json({"valid": valid, "events": count, "head": digest})
    else:
        print(f"valid\t{str(valid).lower()}\nevents\t{count}\nhead\t{digest}")
    return EXIT_OK if valid else EXIT_RUNTIME


def _permissions(args: argparse.Namespace) -> int:
    config, _ = _resolve(args)
    policy = config.policy
    value = {
        "autonomous": policy.autonomous,
        "max_auto_risk": policy.max_auto_risk,
        "allowed_commands": policy.allowed_commands,
    }
    if args.json:
        _emit_json(value)
    elif args.plain:
        print(f"autonomous\t{str(policy.autonomous).lower()}")
        print(f"max_auto_risk\t{policy.max_auto_risk}")
        for executable in policy.allowed_commands:
            print(f"allowed_command\t{executable}")
    else:
        print(f"Autonomous: {policy.autonomous}")
        print(f"Automatic risk ceiling: {policy.max_auto_risk}")
        print("Allowed commands: " + ", ".join(policy.allowed_commands))
    return EXIT_OK


# --- sessions ------------------------------------------------------------------------------

SHOW_TRANSCRIPT_LIMIT = 100  # entries `sessions show` prints without --json
SHOW_ENTRY_CHARS = 400


def _stamp(ns: int) -> str:
    return datetime.fromtimestamp(int(ns) / 1e9).strftime("%Y-%m-%d %H:%M") if ns else ""


def _clip(text: str, limit: int = SHOW_ENTRY_CHARS) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _lock_label(lock: Any) -> str:
    if not lock:
        return "none"
    owner = str(lock.get("owner") or "")
    if lock.get("active"):
        pid = lock.get("pid")
        return f"active (pid {pid}) held by {owner}"
    return f"idle, stale lock by {owner}"


def _find_session(store: Any, project_root: Path, wanted: str) -> dict[str, Any]:
    """Project-bound lookup: only sessions of *this* canonical root can be shown or deleted."""
    sessions = store.list_sessions(str(project_root), limit=10_000)
    matches = [entry for entry in sessions if str(entry["id"]).startswith(wanted)]
    if not matches:
        raise UsageError(f"no session matches {wanted!r} in this project ({project_root})")
    if len(matches) > 1:
        raise UsageError(f"{wanted!r} matches {len(matches)} sessions; give a longer id")
    return matches[0]


def _sessions_list(args: argparse.Namespace, store: Any, project_root: Path) -> int:
    sessions = store.list_sessions(str(project_root), limit=50)
    rows = [{**entry, "active": bool(store.is_active(entry["id"]))} for entry in sessions]
    if args.json:
        _emit_json({"project_root": str(project_root), "sessions": rows})
        return EXIT_OK
    if not rows:
        if not args.plain:
            print("No saved sessions for this project.")
        return EXIT_OK
    if args.plain:
        for row in rows:
            flag = "active" if row["active"] else "-"
            print(f"{row['id']}\t{_stamp(row['updated_ns'])}\t{flag}\t{row['title']}")
        return EXIT_OK
    print(f"{'id':<34}{'updated':<18}{'state':<9}title")
    for row in rows:
        flag = "active" if row["active"] else ""
        print(f"{row['id']:<34}{_stamp(row['updated_ns']):<18}{flag:<9}{row['title']}")
    print(f"{len(rows)} session(s). Actions: nedster sessions show|delete <id>, nedster sessions prune --older-than DAYS")
    return EXIT_OK


def _sessions_show(args: argparse.Namespace, store: Any, project_root: Path) -> int:
    session = _find_session(store, project_root, args.session_id)
    session_id = str(session["id"])
    entries = store.transcript(session_id, limit=SHOW_TRANSCRIPT_LIMIT)
    report = store.outputs_report(session_id)
    lock = store.lock_info(session_id)
    payload = {
        "session": session,
        "messages": len(store.messages(session_id)),
        "transcript_total": store.transcript_count(session_id),
        "transcript_limit": SHOW_TRANSCRIPT_LIMIT,
        "entries": entries,
        "outputs": report,
        "state_keys": store.state_keys(session_id),
        "lock": lock,
    }
    if args.json:
        _emit_json(payload)
        return EXIT_OK
    print(f"session     {session_id}")
    print(f"title       {session['title']}")
    print(f"project     {session['project_root']}")
    print(f"created     {_stamp(session['created_ns'])}")
    print(f"updated     {_stamp(session['updated_ns'])}")
    print(f"messages    {payload['messages']}")
    print(f"transcript  {payload['transcript_total']} entries (showing {len(entries)})")
    print(
        f"outputs     {report['total']} total, {report['referenced']} referenced, "
        f"{report['unreferenced']} prunable"
    )
    print(f"state       {', '.join(payload['state_keys']) or '-'}")
    print(f"lock        {_lock_label(lock)}")
    print("--- transcript ---")
    if not entries:
        print("(no transcript stored for this session)")
    for entry in entries:
        body = _clip(entry["text"])
        print(f"[{entry['role']}] {body}" if body else f"[{entry['role']}]")
    return EXIT_OK


def _sessions_delete(args: argparse.Namespace, store: Any, project_root: Path) -> int:
    session = _find_session(store, project_root, args.session_id)
    session_id = str(session["id"])
    if not store.delete_session(session_id):  # atomically refused while a live lock exists
        _err(f"nedster: session {session_id} is active (locked); close it in the other Nedster and retry")
        return EXIT_RUNTIME
    if args.json:
        _emit_json({"deleted": True, "id": session_id})
    else:
        print(f"Deleted session {session_id}.")
    return EXIT_OK


def _sessions_prune(args: argparse.Namespace, store: Any, project_root: Path) -> int:
    days = float(args.older_than)
    if not 0 < days < 1_000_000:  # positive, finite, and never older than the store itself
        raise UsageError("--older-than must be a positive number of days")
    cutoff_ns = time.time_ns() - int(days * 86_400 * 1e9)
    stats = store.prune_outputs(str(project_root), cutoff_ns=cutoff_ns, dry_run=args.dry_run)
    removable = stats["deleted"]
    payload = {
        "project_root": str(project_root),
        "older_than_days": days,
        "cutoff_ns": cutoff_ns,
        **stats,
    }
    if args.dry_run:
        payload["dry_run"] = True
    if args.json:
        _emit_json(payload)
        return EXIT_OK
    if args.plain:
        for key in ("sessions", "outputs", "deleted", "retained_referenced", "retained_recent"):
            print(f"{key}\t{stats[key]}")
        print(f"dry_run\t{1 if args.dry_run else 0}")
        return EXIT_OK
    # human output – fix the summary line to report removable count
    print(
        f"Scanned {stats['outputs']} output(s) in {stats['sessions']} session(s): "
        f"{removable} removable, {stats['retained_referenced']} referenced, "
        f"{stats['retained_recent']} newer than {days:g} day(s)."
    )
    if removable == 0:
        print("Nothing was prunable.")
    elif args.dry_run:
        print(
            f"Dry run: nothing was deleted. Run again without --dry-run to delete "
            f"{removable} output(s)."
        )
    else:
        print(f"Deleted {removable} unreferenced output(s).")
    print("This prunes stored tool output only; conversations are kept.")
    return EXIT_OK


def _sessions(args: argparse.Namespace) -> int:
    from ironclad.session_store.sqlite_store import SqliteSessionStore

    config, _ = _resolve(args)
    project_root = Path(config.runtime.project_root)
    store = SqliteSessionStore(Path(config.runtime.state_dir) / "sessions.sqlite3")
    try:
        action = str(getattr(args, "action", None) or "list")
        if action == "show":
            return _sessions_show(args, store, project_root)
        if action == "delete":
            return _sessions_delete(args, store, project_root)
        if action == "prune":
            return _sessions_prune(args, store, project_root)
        if action == "list":
            return _sessions_list(args, store, project_root)
        raise UsageError(f"unknown sessions action: {action}")
    finally:
        store.close()


def _check_flags(args: argparse.Namespace) -> None:
    # Check output mode conflicts (more than one of --json, --plain, --jsonl).
    active_modes = [args.json, args.plain]
    if hasattr(args, "jsonl") and args.jsonl:
        active_modes.append(args.jsonl)
    if sum(active_modes) > 1:
        raise UsageError("Choose one output mode: --plain, --json, or --jsonl.")

    # Check --color / --no-color conflict.
    if getattr(args, "color", False) and getattr(args, "no_color", False):
        raise UsageError("Choose either --color or --no-color, not both.")

    # Check --session / --continue conflict.
    session = getattr(args, "session", None)
    continue_ = getattr(args, "resume_latest", False)  # dest of --continue
    if session and continue_:
        raise UsageError("Use either --session or --continue, not both.")


def _tui_or_help(args: argparse.Namespace) -> int:
    machine = args.json or args.plain or args.no_tui
    interactive_tui = not machine and sys.stdout.isatty()
    if not interactive_tui:
        if args.command == "tui":
            _err(
                "The TUI needs an interactive terminal. For piped output use: "
                "nedster run \"...\""
            )
            return EXIT_USAGE
        build_parser().print_help()
        return EXIT_OK
    from .model_manager import state_dir

    if args.command is None and not (state_dir() / "config.toml").exists() and sys.stdin.isatty():
        setup_args = build_parser().parse_args(["--project", args.project, "setup"])
        if _setup(setup_args):
            return EXIT_RUNTIME
    from .tui import run_tui

    config, _ = _resolve(args)
    run_tui(config, ascii_only=args.ascii,
            no_color=args.no_color or (not args.color and bool(os.getenv("NO_COLOR"))))
    return EXIT_OK


def _dispatch(args: argparse.Namespace) -> int:
    _check_flags(args)
    command = args.command
    if command == "config":
        return _config(args)
    if command == "setup":
        return _setup(args)
    if command == "doctor":
        return _doctor(args)
    if command == "models":
        return _models(args)
    if command == "verify-ledger":
        return _verify_ledger(args)
    if command == "permissions":
        return _permissions(args)
    if command == "sessions":
        return _sessions(args)
    if command == "run":
        import asyncio

        return asyncio.run(_run_objective(args, _read_objective(args)))
    return _tui_or_help(args)


def _map_error(exc: Exception) -> tuple[str, int]:
    if isinstance(exc, UsageError):
        return str(exc), EXIT_USAGE
    name = type(exc).__module__.split(".")[0]
    if name == "httpx":
        return f"Cannot reach the model server: {exc}. Run `nedster doctor`.", EXIT_NOT_READY
    return f"{type(exc).__name__}: {exc}" if not str(exc) else str(exc), EXIT_RUNTIME


def _utf8_streams() -> None:
    """Pipes on Windows default to the ANSI code page, which turns every non-ASCII glyph into mojibake."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            with suppress(OSError, ValueError):
                stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_streams()
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        _err("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 — the CLI boundary maps every failure to one line + exit code
        message, code = _map_error(exc)
        _err(f"nedster: error: {message}")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
