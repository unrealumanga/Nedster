"""Backward-compatible module entry point. Prefer the global `nedster` command."""

from nedster_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
