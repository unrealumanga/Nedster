"""Bill splitting for the group-expenses tool."""

from __future__ import annotations

import sys
from decimal import ROUND_HALF_EVEN, Decimal

CENT = Decimal("0.01")


def to_money(value) -> Decimal:
    """Parse str/int/float/Decimal into a Decimal rounded to cents."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_EVEN)


def split_evenly(total, people: int) -> list[Decimal]:
    """Split `total` between `people`, returning one share per person.

    Every share is a whole number of cents.
    """
    if people < 1:
        raise ValueError("need at least one person")
    amount = to_money(total)
    share = (amount / people).quantize(CENT, rounding=ROUND_HALF_EVEN)
    return [share for _ in range(people)]


def split_weighted(total, weights: list[int]) -> list[Decimal]:
    """Split proportionally to integer weights; the remainder goes to the heaviest."""
    if not weights or any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")
    amount = to_money(total)
    cents = int(amount * 100)
    whole = sum(weights)
    parts = [cents * w // whole for w in weights]
    parts[weights.index(max(weights))] += cents - sum(parts)
    return [Decimal(p) / 100 for p in parts]


def format_shares(shares: list[Decimal]) -> str:
    return ", ".join(f"{s:.2f}" for s in shares)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: splitter.py TOTAL PEOPLE")
        return 2
    print(format_shares(split_evenly(args[0], int(args[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
