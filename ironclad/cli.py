from __future__ import annotations

import argparse
import asyncio
import json
import shlex

from rich.console import Console
from rich.table import Table

from .config import load_config
from .hardware import choose_profile, doctor_report, probe_hardware
from .inference import llama_cpp_plan
from .ledger import EventLedger, load_or_create_ledger_key
from .orchestrator import MythosOrchestrator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nedster-ironclad",
        description="Nedster Ironclad: hardware-aware local router/worker agent runtime",
    )
    parser.add_argument("--project", default=".", help="Project root exposed to the safe-zone")
    parser.add_argument("--config", default=None, help="Path to ironclad TOML configuration")
    parser.add_argument("--oracle", action="store_true", help="Enable the optional GPT-5.6 Sol oracle")
    parser.add_argument("--pro", action="store_true", help="Use GPT-5.6 Pro mode for oracle calls")
    parser.add_argument("--autonomous", action="store_true", help="Allow configured write/execute risk ceiling")
    parser.add_argument("--unsafe-process", action="store_true", help="Allow unsandboxed subprocess execution")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("tui", help="Launch the full-screen Ironclad console")
    run = subcommands.add_parser("run", help="Run one objective")
    run.add_argument("objective")
    subcommands.add_parser("doctor", help="Probe hardware and isolation backends")
    subcommands.add_parser("models", help="Show the selected local model fleet")
    subcommands.add_parser("verify-ledger", help="Verify the local event hash chain")
    llama = subcommands.add_parser("llama-command", help="Print a tuned, loopback-only llama.cpp command")
    llama.add_argument("--model", required=True, help="GGUF worker model path")
    llama.add_argument("--draft-model", help="Optional small GGUF draft model")
    llama.add_argument("--port", type=int, default=8080)
    return parser


def _configure(args: argparse.Namespace):
    config = load_config(args.config, args.project)
    if args.oracle:
        config.models.cloud_oracle_enabled = True
    if args.pro:
        config.models.cloud_oracle_enabled = True
        config.models.cloud_reasoning_mode = "pro"
    if args.autonomous:
        config.policy.autonomous = True
    if args.unsafe_process:
        config.sandbox.allow_unsafe_fallback = True
    return config


async def _models(config) -> None:
    runtime = MythosOrchestrator(config)
    try:
        fleet = await runtime.initialize()
        table = Table(title=f"Nedster Ironclad · {runtime.profile.name}", border_style="bright_magenta")
        table.add_column("Role", style="bright_green")
        table.add_column("Model")
        table.add_column("Backend", style="cyan")
        for row in fleet.as_rows():
            table.add_row(*row)
        Console().print(table)
    finally:
        runtime.close()


async def _run(config, objective: str) -> None:
    console = Console()

    def event(value):
        detail = value.get("message") or value.get("reason") or ""
        console.print(f"[dim]{value['seq']:04d}[/] [bright_green]{value['kind']}[/] {detail}")

    runtime = MythosOrchestrator(config, on_event=event)
    try:
        await runtime.initialize()
        result = await runtime.run(objective)
        console.print_json(json.dumps(result, default=str))
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _configure(args)
    command = args.command or "tui"
    if command == "doctor":
        Console().print_json(json.dumps(doctor_report()))
        return 0
    if command == "verify-ledger":
        key = load_or_create_ledger_key(config.runtime.state_dir, config.runtime.ledger_key_env)
        with EventLedger(config.runtime.state_dir / "events.sqlite3", key) as ledger:
            valid, count, digest = ledger.verify()
        Console().print({"valid": valid, "events": count, "head": digest})
        return 0 if valid else 1
    if command == "models":
        asyncio.run(_models(config))
        return 0
    if command == "llama-command":
        plan = llama_cpp_plan(
            choose_profile(probe_hardware()),
            args.model,
            draft_model=args.draft_model,
            port=args.port,
        )
        Console().print(f"[bright_green]{plan.profile}[/] · speculative={plan.speculative_mode}")
        Console().print(shlex.join(plan.command))
        return 0
    if command == "run":
        asyncio.run(_run(config, args.objective))
        return 0
    from .tui import run_tui

    run_tui(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
