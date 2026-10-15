"""The product catalog."""

from __future__ import annotations

from decimal import Decimal

from shop.skus import normalize_sku

CATALOG = {
    "AB-123": {"name": "Bolt M6", "price": Decimal("0.25"), "stock": 400},
    "AB-124": {"name": "Bolt M8", "price": Decimal("0.35"), "stock": 250},
    "NUT-2001": {"name": "Nut M6", "price": Decimal("0.10"), "stock": 900},
    "WSH-310": {"name": "Washer 6mm", "price": Decimal("0.05"), "stock": 1200},
}


def find(raw: str) -> dict:
    sku = normalize_sku(raw)
    try:
        return CATALOG[sku]
    except KeyError:
        raise KeyError(f"unknown SKU {sku}") from None


def names() -> list[str]:
    return sorted(entry["name"] for entry in CATALOG.values())
