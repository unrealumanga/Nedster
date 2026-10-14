"""Stock levels."""

from __future__ import annotations

from shop import catalog
from shop.skus import normalize_sku

_reserved: dict[str, int] = {}


def available(raw: str) -> int:
    sku = normalize_sku(raw)
    return catalog.find(sku)["stock"] - _reserved.get(sku, 0)


def reserve(raw: str, qty: int) -> None:
    sku = normalize_sku(raw)
    if qty > available(sku):
        raise ValueError(f"only {available(sku)} of {sku} left")
    _reserved[sku] = _reserved.get(sku, 0) + qty


def reset() -> None:
    _reserved.clear()
