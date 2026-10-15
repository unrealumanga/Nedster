"""Orders: a list of (sku, qty) lines."""

from __future__ import annotations

from decimal import Decimal

import shop.skus as sk
from shop import inventory, pricing


class Order:
    def __init__(self):
        self.lines: list[tuple[str, int]] = []

    def add(self, raw: str, qty: int = 1) -> None:
        sku = sk.normalize_sku(raw)
        inventory.reserve(sku, qty)
        self.lines.append((sku, qty))

    def total(self) -> Decimal:
        return sum((pricing.line_price(sku, qty) for sku, qty in self.lines), Decimal("0"))
