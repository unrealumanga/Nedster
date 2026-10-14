"""Prices and bulk discounts."""

from __future__ import annotations

from decimal import Decimal

from . import skus
from .catalog import find


def unit_price(raw: str) -> Decimal:
    return find(skus.normalize_sku(raw))["price"]


def line_price(raw: str, qty: int) -> Decimal:
    price = unit_price(raw) * qty
    if qty >= 100:
        price *= Decimal("0.95")
    return price.quantize(Decimal("0.01"))
