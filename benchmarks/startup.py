"""Startup benchmark: median of 7 runs per command plus the top-10 import-time offenders.

    python -m benchmarks.startup [--runs 7]
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

COMMANDS = (
    ("nedster --help", ["--help"]),
    ("nedster config show", ["config", "show"]),
    ("nedster doctor --plain", ["doctor", "--plain"]),
)


def median_ms(argv: list[str], runs: int) -> float:
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        subprocess.run(argv, capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def import_top(n: int = 10) -> list[tuple[int, str]]:
    """Return (cumulative_us, module) for the slowest imports of nedster_agent.cli."""
    proc = subprocess.run([sys.executable, "-X", "importtime", "-c", "import nedster_agent.cli"],
                          capture_output=True, text=True, timeout=120)
    rows = []
    for line in proc.stderr.splitlines():
        parts = line.split("|")
        if len(parts) == 3 and parts[1].strip().isdigit():
            rows.append((int(parts[1]), parts[2].strip()))
    return sorted(rows, reverse=True)[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.startup")
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as state:  # keep doctor/config off the user's real state
        os.environ["NEDSTER_STATE_DIR"] = state
        print(f"{'command':28} median of {args.runs} (ms)")
        for label, extra in COMMANDS:
            print(f"{label:28} {median_ms([sys.executable, '-m', 'nedster_agent', *extra], args.runs):9.0f}")
    print("\nTop 10 imports by cumulative time (us):")
    for micros, module in import_top():
        print(f"{micros:>10}  {module}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
