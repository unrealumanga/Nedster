from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Heavy modules (httpx, rich, textual, asyncio, the orchestrator) are imported inside command
# functions so `nedster --help` and `nedster config show` stay fast.
from . import __version__

EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, EXIT_NOT_READY = 0, 1, 2, 3


class UsageError(Exception):
    """Bad invocation (exit 2)."""


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
    run.add_argument("--engine", choices=("native", "legacy"), default=None, help="Agent engine (default: configuration)")
    run.add_argument("--model", default=None, help="Worker model for the native engine (default: configuration)")
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
    config.add_argument("action", nargs="?", choices=("show",), default="show")
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


def _setup(args: argparse.Namespace) -> int:
    import httpx  # noqa: F401  (surface import problems early, and for error mapping)
    from rich import box
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    from ironclad.hardware import choose_profile, probe_hardware

    from .model_manager import SetupError, install_plan, recommend_plan

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
        )
    except SetupError as exc:
        _err(f"[FAIL] Setup stopped: {exc}")
        if exc.remedy:
            _err(f"Remedy: {exc.remedy}")
        _err("Your previous configuration was left untouched.")
        return EXIT_RUNTIME
    if args.json:
        _emit_json(result)
    elif args.plain:
        for key in ("config", "backup", "aliases"):
            print(f"{key}\t{result[key]}")
    else:
        console.print("Step 4/4  Ready")
        if result.get("backup"):
            console.print(f"Previous config saved to {result['backup']}")
        launcher = _launcher_path()
        console.print(f"Global command: {'ready at ' + launcher if launcher else 'not visible on PATH; restart the terminal'}")
        console.print("Run [bold]nedster[/bold] from any project folder to begin.")
    return EXIT_OK


def _doctor(args: argparse.Namespace) -> int:
    from ironclad.hardware import doctor_report

    from .model_manager import OllamaManager, readiness

    config, _ = _resolve(args)
    report = doctor_report()
    manager = OllamaManager(config.models.ollama_url)
    tiers = readiness(config, manager)
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
    if (getattr(args, "engine", None) or config.engine) == "native":
        return await _run_native(args, config, objective)
    from ironclad.orchestrator import NedsterRuntime

    runtime = NedsterRuntime(config)
    jsonl = args.jsonl
    if jsonl:
        _emit_json({"type": "start", "objective": objective})
    try:
        await runtime.initialize()
        result = await runtime.respond(objective)
    finally:
        runtime.close()
    passed = bool(result.get("passed"))
    if jsonl:
        _emit_json({"type": "result", **result})
        _emit_json({"type": "end", "passed": passed})
    elif args.json:
        _emit_json(result)
    else:
        print(result.get("answer", ""))
        steps = f"{result.get('completed_steps', 0)}/{result.get('total_steps', 0)} steps"
        _err(f"[{'passed' if passed else 'failed'} · {result.get('mode', 'ask')} · {steps}]")
    return EXIT_OK if passed else EXIT_RUNTIME


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
    try:
        result = await session.run(objective)
    finally:
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
    run_tui(config, ascii_only=args.ascii,
            no_color=args.no_color or (not args.color and bool(os.getenv("NO_COLOR"))))
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


def main(argv: list[str] | None = None) -> int:
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
