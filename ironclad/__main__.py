"""``python -m ironclad`` stays a valid entry point; it runs the Nedster CLI."""

from nedster_agent.cli import main

raise SystemExit(main())
