"""SKU handling: every SKU entering the system goes through here."""

from __future__ import annotations

import logging
import re

log = logging.getLogger("shop.skus")

__all__ = ["normalize_sku", "is_valid_sku"]

_SKU = re.compile(r"[A-Z]{2,4}-\d{3,6}")


def normalize_sku(raw: str) -> str:
    """Return the canonical form of a SKU: upper case, dashes only."""
    cleaned = str(raw).strip().upper().replace("_", "-").replace(" ", "-")
    if not _SKU.fullmatch(cleaned):
        log.warning("normalize_sku failed for %r", raw)
        raise ValueError(f"invalid SKU: {raw!r}")
    return cleaned


def is_valid_sku(raw: str) -> bool:
    try:
        normalize_sku(raw)
    except ValueError:
        return False
    return True
