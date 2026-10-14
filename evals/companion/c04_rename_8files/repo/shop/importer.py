"""Bulk import of order lines from CSV-ish rows."""

from __future__ import annotations

from shop.orders import Order
from shop.skus import is_valid_sku, normalize_sku


def import_rows(rows: list[str]) -> tuple[Order, list[str]]:
    """Each row is 'sku,qty'. Returns the order and the rows that were rejected."""
    order = Order()
    rejected = []
    for row in rows:
        sku, _, qty = row.partition(",")
        if not is_valid_sku(sku):
            rejected.append(row)
            continue
        order.add(normalize_sku(sku), int(qty or 1))
    return order, rejected
