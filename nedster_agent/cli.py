from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ironclad.config import load_config
from ironclad.hardware import choose_profile, doctor_report, probe_hardware
from ironclad.ledger import EventLedger, load_or_create_ledger_key
from ironclad.orchestrator import NedsterRuntime

from . import __version__
from .model_manager import OllamaManager, install_plan, recommend_plan, state_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nedster",
        description="Nedster — a local-first AI coding and automation agent",
    )
    parser.add_argument("--project", default=".", help="Project folder (default: current folder)")
    parser.add_argument("--config", default=None, help="Nedster TOML configuration path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
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
    setup.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    setup.add_argument("--skip-ollama-install", action="store_true")
    setup.add_argument("--yes", action="store_true", help="Accept the recommended configuration")
    setup.add_argument("--no-input", action="store_true", help="Never prompt; use flags and defaults")
    setup.add_argument("--dry-run", action="store_true", help="Show actions without downloading or writing")

    doctor = subcommands.add_parser("doctor", help="Check hardware, Ollama, launcher, and configuration")
    accept_global_options(doctor)
    models = subcommands.add_parser("models", help="Inspect or install the local model fleet")
    accept_global_options(models)
    models.add_argument("action", nargs="?", choices=("list", "recommend", "install"), default="list")
    models.add_argument("--profile", choices=("fast", "balanced", "maximum"), default="balanced")
    models.add_argument("--dry-run", action="store_true")

    run = subcommands.add_parser("run", help="Run one objective and exit")
    accept_global_options(run)
    run.add_argument("objective")
    tui = subcommands.add_parser("tui", help="Open the interactive terminal interface")
    accept_global_options(tui)
    ledger = subcommands.add_parser("verify-ledger", help="Verify the local event chain")
    accept_global_options(ledger)
    permissions = subcommands.add_parser("permissions", help="Show the active tool permission policy")
    accept_global_options(permissions)
    return parser


def _console(args: argparse.Namespace) -> Console:
    if args.color and args.no_color:
        raise SystemExit("Choose either --color or --no-color, not both.")
    no_color: bool | None = False if args.color else (True if args.no_color else None)
    return Console(
        no_color=no_color,
        force_terminal=False if args.no_color else None,
    )


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


def _setup(args: argparse.Namespace, console: Console) -> int:
    snapshot = probe_hardware()
    tier = choose_profile(snapshot).name
    interactive = sys.stdin.isatty() and not args.no_input and not args.yes
    preference = args.profile or "balanced"
    recommended = recommend_plan(snapshot, preference)

    if not args.json:
        console.print(Panel.fit("[bold]Nedster setup[/bold]\nLocal models, selected for this computer", border_style="cyan", box=box.ASCII if args.ascii else box.ROUNDED))
        console.print(_ascii(f"Step 1/4  Hardware  {snapshot.accelerator} · {snapshot.vram_total_gb:.1f} GB VRAM · {snapshot.ram_total_gb:.1f} GB RAM", args.ascii))
        console.print(f"Step 2/4  Recommendation  {tier} / {recommended.label.lower()}")
    table = Table(show_header=False, box=None)
    table.add_row("Worker", recommended.worker)
    table.add_row("Router", recommended.router)
    table.add_row("Memory", recommended.embedding)
    table.add_row("Download", f"about {recommended.estimated_download_gb:.1f} GB")
    table.add_row("Why", recommended.notes)
    if not args.json:
        console.print(table)

    if interactive:
        preference = Prompt.ask(
            "Choose a profile",
            choices=["fast", "balanced", "maximum"],
            default=preference,
        )
        recommended = recommend_plan(snapshot, preference)
        if not Confirm.ask("Download and configure these models?", default=True):
            console.print("Setup cancelled; nothing was changed.")
            return 0

    if not args.json:
        console.print("Step 3/4  Ollama and models")
    progress_no_color: bool | None = False if args.color else (True if args.no_color else None)
    progress_console = Console(stderr=True, no_color=progress_no_color)
    try:
        result = install_plan(
            recommended,
            ollama_url=args.ollama_url,
            install_ollama=not args.skip_ollama_install,
            dry_run=args.dry_run,
            progress=lambda message: progress_console.print(f"  {_ascii(message, args.ascii)}"),
        )
    except (RuntimeError, OSError, subprocess.CalledProcessError, httpx.HTTPError) as exc:
        progress_console.print(f"[red][FAIL] Setup stopped:[/red] {exc}")
        return 1
    if args.json:
        console.print_json(json.dumps(result))
    else:
        console.print("Step 4/4  Ready")
        launcher = _launcher_path()
        console.print(f"Global command: {'ready at ' + launcher if launcher else 'not visible on PATH; restart the terminal'}")
        console.print("Run [bold]nedster[/bold] from any project folder to begin.")
    return 0


def _doctor(args: argparse.Namespace, console: Console) -> int:
    report = doctor_report()
    manager = OllamaManager()
    report["nedster"] = {
        "version": __version__,
        "launcher": _launcher_path(),
        "config": str(state_dir() / "config.toml"),
        "configured": (state_dir() / "config.toml").exists(),
    }
    report["ollama"] = {
        "executable": manager.executable,
        "api": manager.api_available(),
        "models": manager.list_models() if manager.api_available() else [],
    }
    if args.json:
        console.print_json(json.dumps(report))
    elif args.plain:
        snapshot = report["hardware"]
        for key, value in (
            ("version", report["nedster"]["version"]),
            ("profile", report["profile"]["name"]),
            ("accelerator", snapshot["accelerator"]),
            ("vram_gb", snapshot["vram_total_gb"]),
            ("ram_gb", snapshot["ram_total_gb"]),
            ("launcher", report["nedster"]["launcher"] or "missing"),
            ("ollama", "ready" if report["ollama"]["api"] else "unavailable"),
            ("configured", str(report["nedster"]["configured"]).lower()),
        ):
            print(f"{key}\t{value}")
    else:
        snapshot = report["hardware"]
        console.print(f"Nedster {__version__} · {report['profile']['name']}")
        console.print(f"Hardware  {snapshot['accelerator']} · {snapshot['vram_total_gb']} GB VRAM · {snapshot['ram_total_gb']} GB RAM")
        console.print(f"Launcher  {report['nedster']['launcher'] or 'not on PATH'}")
        console.print(f"Ollama    {'ready' if report['ollama']['api'] else 'not running'}")
        console.print(f"Config    {'ready' if report['nedster']['configured'] else 'run: nedster setup'}")
    return 0 if report["ollama"]["api"] and report["nedster"]["configured"] else 1


def _models(args: argparse.Namespace, console: Console) -> int:
    manager = OllamaManager()
    plan = recommend_plan(preference=args.profile)
    if args.action == "recommend":
        if args.json:
            console.print_json(json.dumps(asdict(plan)))
        elif args.plain:
            for key, value in asdict(plan).items():
                print(f"{key}\t{value}")
        else:
            console.print(Panel(json.dumps(asdict(plan), indent=2), title="Recommended model plan"))
        return 0
    if args.action == "install":
        install_plan(plan, dry_run=args.dry_run, progress=console.print)
        return 0
    if not manager.api_available():
        console.print("Ollama is not running. Run `nedster setup`.")
        return 1
    models = manager.list_models()
    if args.json:
        console.print_json(json.dumps({"models": models}))
    elif args.plain:
        for model in models:
            print(model)
    else:
        table = Table("Installed Ollama models")
        for model in models:
            table.add_row(model)
        console.print(table)
    return 0


async def _run_objective(args: argparse.Namespace, console: Console) -> int:
    config = load_config(args.config, args.project)
    runtime = NedsterRuntime(config)
    try:
        await runtime.initialize()
        result = await runtime.run(args.objective)
        console.print_json(json.dumps(result, default=str))
        return 0 if result.get("passed") else 2
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = _console(args)
    command = args.command
    if command == "setup":
        return _setup(args, console)
    if command == "doctor":
        return _doctor(args, console)
    if command == "models":
        return _models(args, console)
    if command == "verify-ledger":
        config = load_config(args.config, args.project)
        key = load_or_create_ledger_key(config.runtime.state_dir, config.runtime.ledger_key_env)
        with EventLedger(config.runtime.state_dir / "events.sqlite3", key) as ledger:
            valid, count, digest = ledger.verify()
        console.print({"valid": valid, "events": count, "head": digest})
        return 0 if valid else 1
    if command == "permissions":
        policy = load_config(args.config, args.project).policy
        value = {
            "autonomous": policy.autonomous,
            "max_auto_risk": policy.max_auto_risk,
            "allowed_commands": policy.allowed_commands,
        }
        if args.json:
            console.print_json(json.dumps(value))
        elif args.plain:
            print(f"autonomous\t{str(policy.autonomous).lower()}")
            print(f"max_auto_risk\t{policy.max_auto_risk}")
            for executable in policy.allowed_commands:
                print(f"allowed_command\t{executable}")
        else:
            console.print(f"Autonomous: {policy.autonomous}")
            console.print(f"Automatic risk ceiling: {policy.max_auto_risk}")
            console.print("Allowed commands: " + ", ".join(policy.allowed_commands))
        return 0
    if command == "run":
        return asyncio.run(_run_objective(args, console))

    interactive_tui = not args.no_tui and sys.stdout.isatty()
    config_path = state_dir() / "config.toml"
    if command is None and interactive_tui and not config_path.exists() and sys.stdin.isatty():
        setup_args = build_parser().parse_args(["--project", args.project, "setup"])
        if _setup(setup_args, console):
            return 1
    if not interactive_tui:
        build_parser().print_help()
        return 0
    from .tui import run_tui

    run_tui(
        load_config(args.config, args.project),
        ascii_only=args.ascii,
        no_color=args.no_color or (not args.color and bool(os.getenv("NO_COLOR"))),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
