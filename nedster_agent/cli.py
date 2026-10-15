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

#: Characters of a session id shown in human output (one length on every surface).
SHORT_ID = 8


class UsageError(Exception):
    """Bad invocation (exit 2)."""


class SessionActive(RuntimeError):
    """Another live Nedster holds the session's activity lock (exit 1)."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nedster",
        description="Nedster — a local-first AI coding and automation agent",
    )
    parser.add_argument("--project", default=".", help="Project folder (default: current folder)")
    parser.add_argument("--config", default=None, help="Nedster TOML configuration path")
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable JSON document")
    parser.add_argument("--plain", action="store_true", help="Emit stable, one-record-per-line text")
    parser.add_argument("--no-tui", action="store_true", help="Never open the full-screen interface")
    parser.add_argument("--color", action="store_true", help="Force color when the destination supports it")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--ascii", action="store_true", help="Use ASCII-safe symbols")
    parser.add_argument("--version", action="version", version=f"Nedster {__version__}")
    subcommands = parser.add_subparsers(dest="command")

    def accept_global_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--project", default=argparse.SUPPRESS)
        command_parser.add_argument("--config", default=argparse.SUPPRESS)
        command_parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--plain", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--no-tui", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--color", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--ascii", action="store_true", default=argparse.SUPPRESS)

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
        "--name",
        default=None,
        help="Name the session (sets the title at creation; renames it when resuming)",
    )
    run.add_argument(
        "--expect",
        choices=("change", "answer"),
        default=None,
        help="What a successful run must do: change files, or only answer (default: inferred from the objective)",
    )
    resume = run.add_mutually_exclusive_group()
    resume.add_argument("--continue", dest="resume_latest", action="store_true", help="Continue the latest session here")
    resume.add_argument(
        "--session", default=None, help="Continue a session by id, unique id prefix or exact title"
    )
    tui = subcommands.add_parser("tui", help="Open the interactive terminal interface")
    accept_global_options(tui)
    tui_open = tui.add_mutually_exclusive_group()
    tui_open.add_argument("--session", default=None, help="Open on a session by id, unique id prefix or exact title")
    tui_open.add_argument(
        "--continue", dest="resume_latest", action="store_true", help="Open the most recent conversation"
    )
    resume_command = subcommands.add_parser(
        "resume", help="Open the interface on a saved conversation (picker when no reference is given)"
    )
    accept_global_options(resume_command)
    resume_choice = resume_command.add_mutually_exclusive_group()
    resume_choice.add_argument("ref", nargs="?", default=None, help="Session id, unique id prefix or exact title")
    resume_choice.add_argument(
        "--last", action="store_true", help="Resume the most recent non-empty conversation here"
    )
    ledger = subcommands.add_parser("verify-ledger", help="Verify the local event chain")
    accept_global_options(ledger)
    permissions = subcommands.add_parser("permissions", help="Show the active tool permission policy")
    accept_global_options(permissions)
    config = subcommands.add_parser("config", help="Inspect the effective configuration")
    accept_global_options(config)
    config.add_argument("action", nargs="?", choices=("show",), default="show")

    sessions = subcommands.add_parser(
        "sessions", help="List, show, rename, delete or prune the saved sessions of this project"
    )
    accept_global_options(sessions)
    session_actions = sessions.add_subparsers(dest="action")
    sessions.set_defaults(action="list")  # `nedster sessions` alone lists (argparse has no subparser default)
    sessions_list = session_actions.add_parser("list", help="Saved sessions, newest first (default)")
    accept_global_options(sessions_list)
    sessions_show = session_actions.add_parser("show", help="One session: metadata + stored transcript")
    accept_global_options(sessions_show)
    sessions_show.add_argument("session_id", help="Session id or unique prefix")
    sessions_rename = session_actions.add_parser("rename", help="Give a session a new name")
    accept_global_options(sessions_rename)
    sessions_rename.add_argument("session_id", help="Session id, unique prefix or exact title")
    sessions_rename.add_argument("name", help="New name (1-80 characters)")
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
            kv = f"KV {context['kv_cache_type']}, flash attention {'on' if context['flash_attention'] else 'off'}"
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


def _resolve_run_session(config: Any, ref: str) -> str:
    """Resolve ``--session`` to one project session before any engine/store write.

    Unknown, ambiguous and foreign references raise :class:`UsageError` (exit 2); a session
    another live Nedster is using raises :class:`SessionActive` (exit 1). Nothing is written.
    """
    from ironclad.session_store.sqlite_store import SessionResolutionError, SqliteSessionStore

    store = SqliteSessionStore(Path(config.runtime.state_dir) / "sessions.sqlite3")
    try:
        try:
            session_id = str(store.resolve_session(str(config.runtime.project_root), ref)["id"])
        except SessionResolutionError as exc:
            raise UsageError(str(exc)) from exc
        if store.is_active(session_id):
            raise SessionActive(
                f"Session {session_id[:SHORT_ID]} is active in another Nedster. Close it there and retry."
            )
        return session_id
    finally:
        store.close()


async def _run_objective(args: argparse.Namespace, objective: str) -> int:
    from ironclad.session_store.titles import normalize_title

    config, _ = _resolve(args)
    if getattr(args, "name", None) is not None:
        name = normalize_title(args.name)
        if not name:
            raise UsageError("--name needs at least one visible character.")
        args.name = name
    if args.session is not None:
        args.session = _resolve_run_session(config, args.session)
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
    """Take the activity lock for a CLI run; ``SessionActive`` (exit 1) when another live
    Nedster holds it, so a lock conflict is refused instead of silently writing (F03)."""
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
        raise SessionActive(
            f"Session {session_id[:SHORT_ID]} is active in another Nedster. Close it there and retry."
        )
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
        config,
        emit=emit,
        model=args.model,
        session_id=args.session,
        resume_latest=args.resume_latest,
        title=getattr(args, "name", "") or "",
    )
    if args.resume_latest and getattr(session, "started_new", False):
        _err("No previous conversation here; starting a new one.")  # F06: never silently continue an empty launch
    try:
        heartbeat = _lock_run_session(session)
    except SessionActive:
        await session.aclose()
        raise
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
        _err(f"[{result.outcome.value} · session {session.session_id[:SHORT_ID]}]")
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


def _short(session_id: str) -> str:
    return str(session_id)[:SHORT_ID]


def _lock_label(lock: Any) -> str:
    if not lock:
        return "none"
    owner = str(lock.get("owner") or "")
    if lock.get("active"):
        pid = lock.get("pid")
        return f"active (pid {pid}) held by {owner}"
    return f"idle, stale lock by {owner}"


def _find_session(store: Any, project_root: Path, wanted: str) -> dict[str, Any]:
    """Project-bound lookup via the store's one resolver (id, prefix or title); a resolution
    failure is a usage error (exit 2) before anything is shown, deleted or written."""
    from ironclad.session_store.sqlite_store import SessionResolutionError

    try:
        return store.resolve_session(str(project_root), wanted)
    except SessionResolutionError as exc:
        raise UsageError(str(exc)) from exc


def _display_title(store: Any, session_id: str, title: str) -> str:
    """Name for display: the stored title, else the first user message, else "Untitled conversation"."""
    from ironclad.session_store.titles import display_title

    if str(title or "").strip():
        return str(title).strip()
    first = getattr(store, "first_user_text", None)
    fallback = first(session_id) if callable(first) else ""
    return display_title("", fallback)


def _sessions_list(args: argparse.Namespace, store: Any, project_root: Path) -> int:
    sessions = store.list_sessions(str(project_root), limit=50)
    rows = [{**entry, "active": bool(store.is_active(entry["id"]))} for entry in sessions]
    if args.json:
        _emit_json({"project_root": str(project_root), "sessions": rows})
        return EXIT_OK
    if not rows:
        if not args.plain:
            print('No saved conversations here. Start one: nedster run "Explain this project"')
        return EXIT_OK
    if args.plain:
        for row in rows:
            flag = "active" if row["active"] else "-"
            print(f"{row['id']}\t{_stamp(row['updated_ns'])}\t{flag}\t{row['title']}")
        return EXIT_OK
    print(f"{'id':<10}{'updated':<12}{'messages':<10}title")
    from ironclad.session_store.titles import relative_age

    for row in rows:
        sid = str(row["id"])
        flag = "  [active]" if row["active"] else ""
        title = _display_title(store, sid, row.get("title", ""))
        print(f"{_short(sid):<10}{relative_age(int(row['updated_ns'])):<12}{int(row.get('message_count', 0)):<10}{title}{flag}")
    print(f"{len(rows)} session(s).")
    print('Resume: nedster run --session <id or title> "..."')
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
    print(f"title       {_display_title(store, session_id, session['title'])}")
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


def _sessions_rename(args: argparse.Namespace, store: Any, project_root: Path) -> int:
    from ironclad.session_store.titles import normalize_title

    session = _find_session(store, project_root, args.session_id)
    session_id = str(session["id"])
    name = normalize_title(args.name)
    if not name:
        raise UsageError("The new name needs at least one visible character.")
    store.set_title(session_id, name)
    if args.json:
        _emit_json({"renamed": True, "id": session_id, "title": name})
    else:
        print(f"Renamed session {session_id[:SHORT_ID]} to {name}.")
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
    stats = store.prune_outputs(str(project_root), cutoff_ns=cutoff_ns)
    payload = {"project_root": str(project_root), "older_than_days": days, "cutoff_ns": cutoff_ns, **stats}
    if args.json:
        _emit_json(payload)
        return EXIT_OK
    if args.plain:
        for key in ("sessions", "outputs", "deleted", "retained_referenced", "retained_recent"):
            print(f"{key}\t{stats[key]}")
        return EXIT_OK
    print(f"Scanned {stats['sessions']} session(s), {stats['outputs']} output(s) older than {days:g} day(s).")
    print(f"Deleted {stats['deleted']} unreferenced output(s).")
    print(f"Kept {stats['retained_referenced']} referenced by a message or task state.")
    print(f"Kept {stats['retained_recent']} newer than the cutoff.")
    if stats["outputs"] == stats["retained_referenced"] + stats["retained_recent"]:
        print("Nothing was prunable.")
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
        if action == "rename":
            return _sessions_rename(args, store, project_root)
        if action == "delete":
            return _sessions_delete(args, store, project_root)
        if action == "prune":
            return _sessions_prune(args, store, project_root)
        if action == "list":
            return _sessions_list(args, store, project_root)
        raise UsageError(f"unknown sessions action: {action}")
    finally:
        store.close()


def _tui_session_ref(config: Any, ref: str) -> str:
    """Resolve ``tui --session`` / ``resume <ref>`` up front so a bad reference is a usage error (exit 2)."""
    from ironclad.session_store.sqlite_store import SqliteSessionStore

    store = SqliteSessionStore(Path(config.runtime.state_dir) / "sessions.sqlite3")
    try:
        return str(_find_session(store, Path(config.runtime.project_root), ref)["id"])
    finally:
        store.close()


def _latest_non_empty(config: Any) -> str | None:
    from ironclad.session_store.sqlite_store import SqliteSessionStore

    store = SqliteSessionStore(Path(config.runtime.state_dir) / "sessions.sqlite3")
    try:
        return store.latest_non_empty(str(config.runtime.project_root))
    finally:
        store.close()


def _tui_or_help(args: argparse.Namespace) -> int:
    machine = args.json or args.plain or args.no_tui
    interactive_tui = not machine and sys.stdout.isatty()
    if not interactive_tui:
        build_parser().print_help()
        return EXIT_OK
    from .model_manager import state_dir

    if args.command is None and not (state_dir() / "config.toml").exists() and sys.stdin.isatty():
        setup_args = build_parser().parse_args(["--project", args.project, "setup"])
        if _setup(setup_args):
            return EXIT_RUNTIME
    from .tui import run_tui

    config, _ = _resolve(args)
    ref = getattr(args, "session", None)
    if ref is not None:
        ref = _tui_session_ref(config, ref)
    run_tui(config, ascii_only=args.ascii,
            no_color=args.no_color or (not args.color and bool(os.getenv("NO_COLOR"))),
            session_ref=ref,
            resume_latest=bool(getattr(args, "resume_latest", False)))
    return EXIT_OK


def _resume(args: argparse.Namespace) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        _err('nedster resume needs an interactive terminal; for scripts use: nedster run --session <ref> "..."')
        return EXIT_USAGE
    from .tui import run_tui

    config, _ = _resolve(args)
    no_color = args.no_color or (not args.color and bool(os.getenv("NO_COLOR")))
    ref = getattr(args, "ref", None)
    if ref is not None:
        run_tui(config, ascii_only=args.ascii, no_color=no_color, session_ref=_tui_session_ref(config, ref))
        return EXIT_OK
    if getattr(args, "last", False):
        latest = _latest_non_empty(config)
        if latest is not None:
            run_tui(config, ascii_only=args.ascii, no_color=no_color, session_ref=latest)
            return EXIT_OK
    run_tui(config, ascii_only=args.ascii, no_color=no_color, open_picker_on_start=True)
    return EXIT_OK


def _dispatch(args: argparse.Namespace) -> int:
    command = args.command
    if command == "config":
        return _config_show(args)
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
    if command == "resume":
        return _resume(args)
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
