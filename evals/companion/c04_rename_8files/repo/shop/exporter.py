"""Export an order for the warehouse system."""

from __future__ import annotations

from shop.skus import normalize_sku


def export_order(order) -> str:
    return "\n".join(f"{normalize_sku(sku)};{qty}" for sku, qty in order.lines)
