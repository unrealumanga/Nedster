import os
import sys
import atexit
import signal
import subprocess
import json
import time
import tempfile
import random
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter

from tui import NedsterTUI
from swarm_coordinator import main_swarm_entry
from skill_manager import install_skill

# --- Globals & Utils ---
_KEEP_VRAM = False
_ACTIVE_MODEL_NAME = "aria-local"
_AVAILABLE_MODELS_CACHE = []
_AVAILABLE_SESSIONS_CACHE = []
TIPS = [
    "Tip: Use /editor to open a full-screen editor for large prompts.",
    "Tip: /switch <number> is a fast way to change models.",
    "Tip: Use /think to toggle the visibility of the agent's reasoning.",
    "Tip: /swarm <prompt> can break down large tasks for parallel agents.",
    "Tip: Use /sessions to list and /resume <number> to continue past conversations.",
]


def _release_vram():
    if not _KEEP_VRAM:
        try:
            subprocess.run(
                ["ollama", "stop", _ACTIVE_MODEL_NAME], capture_output=True, timeout=5
            )
        except Exception:
            pass


atexit.register(_release_vram)


def check_ollama():
    try:
        import ollama

        ollama.Client(host="127.0.0.1").list()
        return True
    except Exception:
        return False


def get_editor_input():
    editor = os.environ.get("EDITOR", "nano" if sys.platform != "win32" else "notepad")
    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    ) as tf:
        tf.write(
            "# Enter your prompt. When you save and close, Nedster will process it.\n"
        )
        temp_path = tf.name
    subprocess.call([editor, temp_path])
    with open(temp_path, "r", encoding="utf-8") as tf:
        content = tf.read().strip()
    os.remove(temp_path)
    if content.startswith("# Enter your prompt"):
        return ""
    return content


def _get_model_capability(model_name: str) -> str:
    name_lower = model_name.lower()
    if name_lower.startswith("aria-") or "instruct" in name_lower:
        return "full"
    if "embed" in name_lower:
        return "embed"
    if any(s in name_lower for s in ["1.5b", "2b", "3b"]):
        return "chat"
    return "tools"


# --- REPL and Slash Commands ---
def handle_slash_command(cmd: str, agent, project_dir: str):
    global _AVAILABLE_MODELS_CACHE, _ACTIVE_MODEL_NAME
    tui = NedsterTUI()
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ["/exit", "/quit", "/bye"]:
        return "exit"
    elif command == "/think":
        return "think_toggled"

    elif command in ["/model", "/models"]:
        try:
            import ollama

            client = ollama.Client(host="127.0.0.1")
            models_list = client.list().get("models", [])
            _AVAILABLE_MODELS_CACHE = [m["model"] for m in models_list]
            tui.print_status("Available models:")
            for i, model in enumerate(models_list):
                name, cap = model["model"], _get_model_capability(model["model"])
                cap_icon = {
                    "full": "★★★",
                    "tools": "★★☆",
                    "chat": "★☆☆",
                    "embed": "---",
                }[cap]
                active = " ← ACTIVE" if name.startswith(_ACTIVE_MODEL_NAME) else ""
                print(f"  [{i + 1}] {name.ljust(40)} {cap_icon}{active}")
            print("\nUse /switch <name_or_number> to change.")
        except Exception as e:
            tui.print_error(f"Ollama connection failed: {e}")

    elif command == "/switch":
        if not arg:
            tui.print_warning("Usage: /switch <model_name_or_number>")
            return
        model_to_switch = ""
        if arg.isdigit() and _AVAILABLE_MODELS_CACHE:
            try:
                index = int(arg) - 1
                if 0 <= index < len(_AVAILABLE_MODELS_CACHE):
                    model_to_switch = _AVAILABLE_MODELS_CACHE[index]
                else:
                    tui.print_error("Invalid model number.")
                    return
            except (ValueError, IndexError):
                tui.print_error("Invalid model number.")
                return
        else:
            if not any(m.startswith(arg) for m in _AVAILABLE_MODELS_CACHE):
                tui.print_error(f"Model '{arg}' not found.")
                return
            model_to_switch = arg

        _ACTIVE_MODEL_NAME = model_to_switch
        agent.model = model_to_switch
        tui.print_success(f"Switched active model to {model_to_switch}")

    elif command in ["/session", "/sessions", "/resume"]:
        log_dir = Path.home() / ".aria" / "session_logs"
        if not log_dir.exists():
            tui.print_warning("No session log directory found.")
            return

        global _AVAILABLE_SESSIONS_CACHE
        _AVAILABLE_SESSIONS_CACHE = sorted(
            log_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True
        )

        if command == "/resume":
            if not arg:
                tui.print_warning("Usage: /resume <session_number>")
                return

            if agent.memory.turn_count > 0:
                tui.print_status("Current session state has been auto-saved.")

            try:
                index = int(arg) - 1
                if 0 <= index < len(_AVAILABLE_SESSIONS_CACHE):
                    session_to_load = _AVAILABLE_SESSIONS_CACHE[index]
                    agent.memory.load_session_from_log(session_to_load)
                    tui.print_success(f"Resumed session {session_to_load.stem}")
                else:
                    tui.print_error("Invalid session number.")
            except (ValueError, IndexError):
                tui.print_error("Invalid session number.")

        else:  # /sessions or /session
            tui.print_status("Recent sessions:")
            if not _AVAILABLE_SESSIONS_CACHE:
                print("  No sessions found.")
            for i, log_file in enumerate(_AVAILABLE_SESSIONS_CACHE[:15]):
                try:
                    with open(log_file, "r") as f:
                        first_line = f.readline()
                        if first_line:
                            first_event = json.loads(first_line)
                            topic = first_event.get("data", {}).get(
                                "text", "Session start"
                            )[:50]
                            timestamp = first_event.get("ts", "No date")
                            print(f"  [{i + 1}] {timestamp.split('T')[0]} | {topic}...")
                        else:
                            print(f"  [{i + 1}] {log_file.stem} | (Empty session)")
                except Exception:
                    print(f"  [{i + 1}] {log_file.stem} | (Could not parse)")
            print("\nUse /resume <number> to load a session.")

    elif command == "/editor":
        tui.print_status("Opening editor...")
        user_input = get_editor_input()
        if not user_input:
            tui.print_warning("Editor closed with no input.")
            return

        if len(user_input) > 500:
            if click.confirm("This is a large task. Route to Swarm Coordinator?"):
                main_swarm_entry(user_input, project_dir)
            else:
                agent.generate(user_input)
        else:
            agent.generate(user_input)
        return "input_handled"
    else:
        tui.print_warning(f"Unknown command: {command}")
    return None


def cmd_repl(project_dir: str, auto: bool, think: bool):
    tui = NedsterTUI()
    if not check_ollama():
        tui.print_error("Ollama not running. Run: ollama serve")
        sys.exit(1)

    from agent import NedsterAgent

    agent = NedsterAgent(project_dir, auto=auto, think=think)
    agent.model = _ACTIVE_MODEL_NAME

    history_path = os.path.expanduser("~/.aria/nedster_history.txt")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    history = FileHistory(history_path)
    completer = WordCompleter(
        ["/think", "/exit", "/models", "/switch", "/editor", "/swarm"], ignore_case=True
    )
    session = PromptSession(history=history, completer=completer)

    tui.print_boot_logo()

    while True:
        try:
            ctx_pct = agent.memory.get_context_percentage()
            status_text = (
                f"Context: {ctx_pct}% | Memory: ∞ | Model: {_ACTIVE_MODEL_NAME}"
            )
            tui.print_status_bar(status_text)
            tui.console.print(f"[dim italic]{random.choice(TIPS)}[/]", justify="center")

            user_input = session.prompt("\nNedster> ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                result = handle_slash_command(user_input, agent, project_dir)
                if result == "exit":
                    break
                elif result == "think_toggled":
                    agent.think_visible = not agent.think_visible
            else:
                agent.generate(user_input)
        except (KeyboardInterrupt, EOFError):
            break
    print("\n[Nedster session ended]")


# --- Click CLI Definition ---
@click.group()
@click.option("--project", default=os.getcwd(), help="Project directory.")
@click.option("--auto", is_flag=True, help="Enable auto-confirmation mode.")
@click.option("--think", is_flag=True, help="Enable think mode for complex queries.")
@click.option(
    "--keep-vram", is_flag=True, help="Do not unload model from VRAM on exit."
)
@click.pass_context
def cli(ctx, project, auto, think, keep_vram):
    ctx.ensure_object(dict)
    ctx.obj.update({"PROJECT": project, "AUTO": auto, "THINK": think})
    global _KEEP_VRAM
    _KEEP_VRAM = keep_vram
    if not Path(project).exists():
        NedsterTUI().print_error(f"Project directory not found: {project}")
        sys.exit(1)


@cli.command()
@click.pass_context
def repl(ctx):
    cmd_repl(ctx.obj["PROJECT"], ctx.obj["AUTO"], ctx.obj["THINK"])


@cli.command()
@click.argument("prompt")
@click.pass_context
def oneshot(ctx, prompt):
    from agent import NedsterAgent

    agent = NedsterAgent(
        ctx.obj["PROJECT"], auto=ctx.obj["AUTO"], think=ctx.obj["THINK"]
    )
    agent.generate(prompt)


@cli.command()
@click.argument("prompt")
@click.pass_context
def swarm(ctx, prompt):
    main_swarm_entry(prompt, ctx.obj["PROJECT"])


@cli.command(hidden=True)
@click.option("--project-dir", required=True)
@click.option("--task", required=True)
@click.option("--job-id", required=True, type=int)
@click.option("--scoped-dirs", required=True)
def work(project_dir, task, job_id, scoped_dirs):
    from agent import NedsterAgent

    agent = NedsterAgent(
        project_dir,
        auto=True,
        think=True,
        job_id=job_id,
        scoped_dirs=scoped_dirs.split(","),
    )
    agent.generate(task)


@cli.command("install-skill")
@click.argument("skill_url")
def install_skill_cmd(skill_url):
    install_skill(skill_url)


if __name__ == "__main__":
    cli()
