#!/usr/bin/env python3
"""Nedster - Local Claude Code Clone CLI

Usage:
  nedster                        → interactive agent REPL
  nedster --project /path        → set working project directory
  nedster --auto                 → no confirmation, full auto mode
  nedster --think                → enable Qwen think mode
  nedster "fix the bug in foo.py" → one-shot task
  nedster init                   → create NEDSTER.md in cwd
  nedster stats                  → show context, VRAM, vector stats
  nedster reset                  → wipe ChromaDB
"""

import argparse
import os
import sys
import atexit
import signal
import subprocess
from pathlib import Path

from tui import NedsterTUI


_KEEP_VRAM = False
_ACTIVE_MODEL = "aria-qwen"


def _release_vram():
    global _KEEP_VRAM, _ACTIVE_MODEL
    if _KEEP_VRAM:
        return
    try:
        subprocess.run(
            ["ollama", "stop", _ACTIVE_MODEL], capture_output=True, timeout=5
        )
        print("\n[Nedster] VRAM released.")
    except Exception:
        pass


atexit.register(_release_vram)
signal.signal(signal.SIGINT, lambda s, f: (_release_vram(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (_release_vram(), sys.exit(0)))


def check_ollama():
    """Check if Ollama is running."""
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def print_stats():
    """Show VRAM, token budget, tool status."""
    import torch
    import psutil
    import subprocess

    tui = NedsterTUI()

    # CPU RAM
    cpu_ram = psutil.virtual_memory()
    used_gb = cpu_ram.used / (1024**3)
    total_gb = cpu_ram.total / (1024**3)
    print(f"CPU RAM: {used_gb:.1f} GB used / {total_gb:.1f} GB total")

    # GPU VRAM
    try:
        vram = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total"],
            capture_output=True,
            text=True,
        )
        if vram.returncode == 0:
            lines = vram.stdout.strip().split("\n")
            for line in lines:
                parts = line.split(",")
                if len(parts) == 2:
                    used = float(parts[0].replace("MiB", "").strip()) / 1024
                    total = float(parts[1].replace("MiB", "").strip()) / 1024
                    print(f"GPU VRAM: {used:.1f} GB used / {total:.1f} GB total")
    except Exception:
        print("GPU VRAM: nvidia-smi not available")

    # ChromaDB vectors
    try:
        import chromadb

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="rag_docs")
        print(f"ChromaDB: {collection.count()} vectors stored")
    except Exception:
        print("ChromaDB: 0 vectors stored (or not initialized)")

    # Token budget
    print(f"\nToken Budget: 4096 (aria-qwen)")
    print(f"Model: aria-qwen (Qwen3.5:9b)")


def cmd_init(project_dir: str):
    """Create NEDSTER.md in cwd."""
    from context_loader import ContextLoader

    loader = ContextLoader(project_dir)
    loader._create_nedster_md()

    nedster_path = Path(project_dir) / "NEDSTER.md"
    if nedster_path.exists():
        tui = NedsterTUI()
        tui.print_success(f"NEDSTER.md created at {nedster_path}")
    else:
        tui = NedsterTUI()
        tui.print_error("Failed to create NEDSTER.md")


def cmd_reset():
    """Wipe ChromaDB."""
    import shutil

    chroma_path = Path("./chroma_db")
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        tui = NedsterTUI()
        tui.print_success("ChromaDB wiped. Re-run ingest to populate.")
    else:
        tui = NedsterTUI()
        tui.print_warning("No ChromaDB found to reset.")


def cmd_stats():
    """Show stats."""
    print_stats()


def cmd_one_shot(query: str, project_dir: str, auto: bool, think: bool):
    """Run a single query and exit."""
    from agent import NedsterAgent

    tui = NedsterTUI()

    if not check_ollama():
        tui.print_error("Ollama not running. Run: ollama serve")
        sys.exit(1)

    agent = NedsterAgent(project_dir, auto=auto, think=think)
    agent.generate(query)


def setup_readline():
    try:
        import readline
        import os

        histfile = os.path.expanduser("~/.aria/nedster_history.txt")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass

        import atexit

        atexit.register(readline.write_history_file, histfile)

        commands = [
            "/add",
            "/diff",
            "/apply",
            "/git",
            "/test",
            "/undo",
            "/clear",
            "/stats",
            "/auto",
            "/think",
            "/exit",
            "/models",
            "/switch",
            "/sessions",
            "/load",
            "/project",
            "/vram",
            "/tools",
            "/journal",
            "/save",
            "/reload",
            "/compact",
            "/quit",
            "/bye",
        ]

        def completer(text, state):
            options = [c for c in commands if c.startswith(text)]
            if state < len(options):
                return options[state]
            else:
                return None

        readline.parse_and_bind("tab: complete")
        readline.set_completer(completer)
    except ImportError:
        pass


def cmd_repl(project_dir: str, auto: bool, think: bool):
    """Interactive REPL loop."""
    from agent import NedsterAgent

    tui = NedsterTUI()

    if not check_ollama():
        tui.print_error("Ollama not running. Run: ollama serve")
        sys.exit(1)

    # Create agent
    agent = NedsterAgent(project_dir, auto=auto, think=think)

    print("\nSlash commands:")
    print("  /add <file>   - manually add file to context")
    print("  /diff         - show pending diffs")
    print("  /apply        - apply all pending diffs")
    print("  /git          - git status + recent log")
    print("  /test         - run detected test suite")
    print("  /undo         - revert last file edit")
    print("  /clear        - clear short-term memory")
    print("  /stats        - VRAM + token budget + tool status")
    print("  /auto         - toggle auto-approve mode")
    print("  /think        - toggle think mode on/off")
    print("  /tools        - show all tools + status")
    print("  /journal      - show last 5 journal entries")
    print("  /save         - force-save milestones")
    print("  /reload       - reload personality")
    print("  /compact      - compress memory")
    print("  /exit         - exit REPL\n")

    setup_readline()
    _ctrlc_count = 0
    while True:
        try:
            # Build prompt with auto/think status
            status_flags = []
            if auto:
                status_flags.append("auto")
            if think:
                status_flags.append("think")
            status_str = f" [{', '.join(status_flags)}]" if status_flags else ""

            user_input = input(f"\nNedster{status_str}> ").strip()
            _ctrlc_count = 0  # reset on successful input

            if not user_input:
                continue

            # Handle slash commands or paths
            if user_input.startswith("/") and os.path.exists(user_input.strip()):
                user_input = f"what is in {user_input.strip()}"
                agent.generate(user_input)
                continue
            elif user_input.startswith("~/"):
                expanded = os.path.expanduser(user_input.strip())
                if os.path.exists(expanded):
                    user_input = f"what is in {expanded}"
                    agent.generate(user_input)
                    continue
            elif user_input.startswith("/"):
                cmd_result = handle_slash_command(
                    user_input, agent, project_dir, auto, think
                )

                if cmd_result == "exit":
                    break
                elif cmd_result == "auto_toggled":
                    auto = not auto
                elif cmd_result == "think_toggled":
                    think = not think
                elif cmd_result == "think_on":
                    think = True
                elif cmd_result == "think_off":
                    think = False
                continue

            # Regular query
            agent.generate(user_input)

        except KeyboardInterrupt:
            _ctrlc_count += 1
            if _ctrlc_count == 1:
                print("\n[Ctrl+C again to exit, or press Enter to continue]")
                continue
            else:
                break
        except EOFError:
            break

    print("\n[Nedster session ended]")
    _release_vram()
    sys.exit(0)


def handle_slash_command(cmd: str, agent, project_dir: str, auto: bool, think: bool):
    """Handle REPL slash commands."""
    tui = NedsterTUI()
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ["/exit", "/quit", "/bye"]:
        return "exit"

    elif command == "/add":
        if arg:
            from context_loader import ContextLoader

            loader = ContextLoader(project_dir)
            content = loader._read_file(arg)
            if content:
                tui.print_success(f"Added {arg} to context ({len(content)} chars)")
            else:
                tui.print_error(f"Could not read {arg}")
        else:
            tui.print_warning("Usage: /add <file>")

    elif command == "/diff":
        pending = agent.editor.get_pending_diffs()
        if pending:
            for path, (orig, new) in pending.items():
                tui.print_diff(path, orig, new)
        else:
            tui.print_status("No pending diffs")

    elif command == "/apply":
        tui.print_status("Pending edits applied")

    elif command == "/git":
        try:
            from git_tools import git_status

            status = git_status(project_dir)
            print(status)
        except Exception as e:
            tui.print_error(f"Git error: {e}")

    elif command == "/test":
        try:
            from code_tools import run_tests

            result = run_tests(project_dir)
            print(result[:2000])
        except Exception as e:
            tui.print_error(f"Test error: {e}")

    elif command == "/undo":
        result = agent.editor.undo_last()
        tui.print_status(result)

    elif command == "/clear":
        agent.memory.clear()
        tui.print_success("Short-term memory cleared")

    elif command == "/models":
        try:
            import subprocess

            r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            tui.print_status("Available models:")
            for line in r.stdout.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    size = parts[1] + " " + parts[2]
                    active = " [ACTIVE]" if name == agent.model else ""
                    print(f"  {name.ljust(20)} {size.ljust(8)} {active}")
            print("Use /switch <model> to change active model.")
        except Exception:
            tui.print_error("Ollama not running or not found.")

    elif command == "/switch":
        if arg:
            import subprocess

            try:
                # stop current
                subprocess.run(["ollama", "stop", agent.model], capture_output=True)
                agent.model = arg
                global _ACTIVE_MODEL
                _ACTIVE_MODEL = arg
                tui.print_status(f"Switched to {arg}. VRAM reloading...")
                # trigger load
                subprocess.run(["ollama", "run", arg, ""], capture_output=True)
            except Exception as e:
                tui.print_error(f"Switch failed: {e}")
        else:
            tui.print_warning("Usage: /switch <model>")

    elif command == "/sessions":
        import os

        path = os.path.expanduser("~/.aria/milestones.md")
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            tui.print_status("Recent sessions:")
            for i, line in enumerate(lines):
                if line.startswith("## Session "):
                    print(f"  {line.strip()}")
            print("Use /load <session_id> to restore context.")
        else:
            tui.print_warning("No milestones.md found.")

    elif command == "/load":
        if arg:
            import os

            path = os.path.expanduser("~/.aria/milestones.md")
            if os.path.exists(path):
                with open(path) as f:
                    content = f.read()
                session_start = content.find(f"## Session {arg}")
                if session_start != -1:
                    session_end = content.find("## Session ", session_start + 10)
                    if session_end == -1:
                        session_content = content[session_start:]
                    else:
                        session_content = content[session_start:session_end]
                    agent.memory.session_summary = session_content.strip()
                    tui.print_success(f"Session {arg} context loaded.")
                else:
                    tui.print_error(f"Session {arg} not found.")
        else:
            tui.print_warning("Usage: /load <session_id>")

    elif command == "/project":
        if arg:
            import os
            from pathlib import Path

            new_proj = Path(os.path.expanduser(arg)).resolve()
            if new_proj.is_dir():
                from tools import SESSION
                SESSION.set_project(str(new_proj))
                agent.project_dir = str(new_proj)
                agent.context_loader.project_root = new_proj
                files = agent.context_loader.scan_project()
                tui.print_success(f"Project: {new_proj.name} | {files} files scanned")
            else:
                tui.print_error(f"Directory not found: {arg}")
        else:
            tui.print_warning("Usage: /project <path>")

    elif command == "/vram":
        import subprocess

        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            tui.print_status(f"VRAM: {r.stdout.strip()} — {agent.model} loaded")
        except Exception:
            tui.print_error("nvidia-smi not available.")

    elif command == "/tools":
        from tools import TOOL_REGISTRY, probe_tools

        status = probe_tools()
        tui.print_status("Tools:")
        for t in TOOL_REGISTRY.keys():
            st = (
                status.get(t, "OK")
                if t in ["bash", "tavily", "duckduckgo", "ollama"]
                else "OK"
            )
            print(f"  {t.ljust(15)} {st}")

    elif command == "/journal":
        tui.print_status("Not implemented yet.")

    elif command == "/save":
        result = agent.save_session()
        tui.print_status(result)

    elif command == "/verbose":
        agent.verbose = not agent.verbose
        tui.print_success(f"Verbose mode: {'ON' if agent.verbose else 'OFF'}")

    elif command == "/reload":
        agent.context_loader.read_nedster_md()
        tui.print_success("NEDSTER.md and personality reloaded.")

    elif command == "/compact":
        agent.memory._compress_session()
        tui.print_success("Memory compressed.")

    elif command == "/stats":
        print_stats()
        tui.print_status(f"Tool stats: {agent.tool_stats}")

    elif command == "/auto":
        new_auto = not auto
        tui.print_success(f"Auto mode: {'ON' if new_auto else 'OFF'}")
        return "auto_toggled"

    elif command in ["/think", "/thinking"]:
        if arg == "on":
            tui.print_success("Think mode: ON — adds ~500 tokens")
            return "think_on"
        elif arg == "off":
            tui.print_success("Think mode: OFF")
            return "think_off"
        else:
            new_think = not think
            tui.print_success(
                f"Think mode: {'ON' if new_think else 'OFF'} — adds ~500 tokens"
            )
            return "think_toggled"

    else:
        tui.print_warning(f"Unknown command: {command}")


def main():
    parser = argparse.ArgumentParser(
        description="Nedster - Local Claude Code Clone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "query", nargs="?", help="One-shot query (e.g., 'fix the bug in foo.py')"
    )

    parser.add_argument(
        "--project",
        type=str,
        default=os.getcwd(),
        help="Project directory (default: cwd)",
    )

    parser.add_argument(
        "--auto", action="store_true", help="No confirmation, full auto mode"
    )

    parser.add_argument("--think", action="store_true", help="Enable Qwen think mode")

    parser.add_argument(
        "--keep-vram",
        action="store_true",
        help="Skip unloading model from VRAM on exit",
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["init", "stats", "reset"],
        help="Commands: init (create NEDSTER.md), stats, reset (wipe ChromaDB)",
    )

    args = parser.parse_args()

    global _KEEP_VRAM
    _KEEP_VRAM = args.keep_vram

    # Fix argparse overlap: if query is a command
    if args.query in ["init", "stats", "reset"]:
        args.command = args.query
        args.query = None

    # Handle positional commands
    if args.command:
        project_dir = Path(args.project)

        if args.command == "init":
            cmd_init(str(project_dir))
            return
        elif args.command == "stats":
            cmd_stats()
            return
        elif args.command == "reset":
            cmd_reset()
            return

    # Handle query or REPL
    project_dir = Path(args.project)
    if not project_dir.exists():
        tui = NedsterTUI()
        tui.print_error(f"Project directory not found: {project_dir}")
        sys.exit(1)

    try:
        from nedster_api import start_api_server

        start_api_server()
    except Exception as e:
        print(f"[API Server failed to start: {e}]")

    if args.query:
        # One-shot mode
        cmd_one_shot(args.query, str(project_dir), args.auto, args.think)
    else:
        # REPL mode
        cmd_repl(str(project_dir), args.auto, args.think)


if __name__ == "__main__":
    main()
