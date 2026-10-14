"""Order rules engine: pricing, eligibility and routing rules for the storefront.

Rules are small pure functions registered in RULES at the bottom of the file."""

from __future__ import annotations

import re
from datetime import date, timedelta

ZONE_BASE = {"LOCAL": 4.0, "EU": 9.5, "WORLD": 22.0}
PER_KG = 0.8


def tier_discount_000(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 0: 3% above 382, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 382:
        rate += 3 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_000 = {"white": 49, "black": 46, "grey": 49, "teal": 17}


def colour_surcharge_000(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 0; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_000.get(key, 0) * max(qty, 0)


def clean_code_000(raw: str) -> str:
    """Normalise a promo code of group 0: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_000(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 0; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_000(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 0: members over 751, or anyone over 1502 with 3+ items."""
    if is_member and subtotal >= 751:
        return True
    return subtotal >= 1502 and items >= 3


def tier_discount_001(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 1: 6% above 222, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 222:
        rate += 6 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_001 = {"pink": 46, "black": 22, "blue": 34, "white": 5}


def colour_surcharge_001(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 1; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_001.get(key, 0) * max(qty, 0)


def clean_code_001(raw: str) -> str:
    """Normalise a promo code of group 1: letters and digits only, max 12 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 12:
        cleaned = cleaned[:12]
    return cleaned


def delivery_estimate_001(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 1; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_001(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 1: members over 826, or anyone over 1652 with 3+ items."""
    if is_member and subtotal >= 826:
        return True
    return subtotal >= 1652 and items >= 3


def tier_discount_002(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 2: 3% above 282, plus 8 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 282:
        rate += 3 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 8 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_002 = {"black": 41, "navy": 27, "red": 30, "blue": 8}


def colour_surcharge_002(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 2; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_002.get(key, 0) * max(qty, 0)


def clean_code_002(raw: str) -> str:
    """Normalise a promo code of group 2: letters and digits only, max 10 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 10:
        cleaned = cleaned[:10]
    return cleaned


def delivery_estimate_002(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 2; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_002(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 2: members over 499, or anyone over 998 with 3+ items."""
    if is_member and subtotal >= 499:
        return True
    return subtotal >= 998 and items >= 3


def tier_discount_003(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 3: 6% above 112, plus 4 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 112:
        rate += 6 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 4 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_003 = {"grey": 37, "navy": 14, "black": 27, "blue": 15}


def colour_surcharge_003(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 3; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_003.get(key, 0) * max(qty, 0)


def clean_code_003(raw: str) -> str:
    """Normalise a promo code of group 3: letters and digits only, max 12 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 12:
        cleaned = cleaned[:12]
    return cleaned


def delivery_estimate_003(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 3; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_003(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 3: members over 867, or anyone over 1734 with 3+ items."""
    if is_member and subtotal >= 867:
        return True
    return subtotal >= 1734 and items >= 3


def tier_discount_004(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 4: 1% above 142, plus 3 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 142:
        rate += 1 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 3 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_004 = {"pink": 46, "grey": 27, "navy": 40, "white": 8}


def colour_surcharge_004(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 4; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_004.get(key, 0) * max(qty, 0)


def clean_code_004(raw: str) -> str:
    """Normalise a promo code of group 4: letters and digits only, max 16 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 16:
        cleaned = cleaned[:16]
    return cleaned


def delivery_estimate_004(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 4; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_004(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 4: members over 899, or anyone over 1798 with 3+ items."""
    if is_member and subtotal >= 899:
        return True
    return subtotal >= 1798 and items >= 3


def tier_discount_005(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 5: 4% above 247, plus 3 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 247:
        rate += 4 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 3 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_005 = {"grey": 32, "blue": 39, "green": 41, "teal": 10}


def colour_surcharge_005(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 5; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_005.get(key, 0) * max(qty, 0)


def clean_code_005(raw: str) -> str:
    """Normalise a promo code of group 5: letters and digits only, max 14 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 14:
        cleaned = cleaned[:14]
    return cleaned


def delivery_estimate_005(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 5; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_005(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 5: members over 738, or anyone over 1476 with 3+ items."""
    if is_member and subtotal >= 738:
        return True
    return subtotal >= 1476 and items >= 3


def tier_discount_006(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 6: 6% above 372, plus 4 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 372:
        rate += 6 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 4 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_006 = {"teal": 50, "blue": 9, "grey": 22, "navy": 14}


def colour_surcharge_006(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 6; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_006.get(key, 0) * max(qty, 0)


def clean_code_006(raw: str) -> str:
    """Normalise a promo code of group 6: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_006(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 6; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_006(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 6: members over 200, or anyone over 400 with 3+ items."""
    if is_member and subtotal >= 200:
        return True
    return subtotal >= 400 and items >= 3


def tier_discount_007(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 7: 1% above 87, plus 2 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 87:
        rate += 1 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 2 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_007 = {"teal": 6, "black": 13, "blue": 19, "green": 16}


def colour_surcharge_007(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 7; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_007.get(key, 0) * max(qty, 0)


def clean_code_007(raw: str) -> str:
    """Normalise a promo code of group 7: letters and digits only, max 20 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 20:
        cleaned = cleaned[:20]
    return cleaned


def delivery_estimate_007(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 7; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_007(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 7: members over 369, or anyone over 738 with 3+ items."""
    if is_member and subtotal >= 369:
        return True
    return subtotal >= 738 and items >= 3


def tier_discount_008(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 8: 9% above 233, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 233:
        rate += 9 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_008 = {"navy": 20, "black": 20, "pink": 18, "red": 9}


def colour_surcharge_008(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 8; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_008.get(key, 0) * max(qty, 0)


def clean_code_008(raw: str) -> str:
    """Normalise a promo code of group 8: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_008(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 8; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_008(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 8: members over 639, or anyone over 1278 with 3+ items."""
    if is_member and subtotal >= 639:
        return True
    return subtotal >= 1278 and items >= 3


def tier_discount_009(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 9: 8% above 121, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 121:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_009 = {"black": 15, "navy": 21, "green": 23, "white": 49}


def colour_surcharge_009(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 9; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_009.get(key, 0) * max(qty, 0)


def clean_code_009(raw: str) -> str:
    """Normalise a promo code of group 9: letters and digits only, max 6 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 6:
        cleaned = cleaned[:6]
    return cleaned


def delivery_estimate_009(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 9; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_009(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 9: members over 794, or anyone over 1588 with 3+ items."""
    if is_member and subtotal >= 794:
        return True
    return subtotal >= 1588 and items >= 3


def tier_discount_010(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 10: 7% above 389, plus 7 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 389:
        rate += 7 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 7 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_010 = {"green": 34, "pink": 9, "teal": 14, "red": 41}


def colour_surcharge_010(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 10; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_010.get(key, 0) * max(qty, 0)


def clean_code_010(raw: str) -> str:
    """Normalise a promo code of group 10: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_010(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 10; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_010(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 10: members over 616, or anyone over 1232 with 3+ items."""
    if is_member and subtotal >= 616:
        return True
    return subtotal >= 1232 and items >= 3


def tier_discount_011(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 11: 4% above 259, plus 2 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 259:
        rate += 4 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 2 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_011 = {"blue": 40, "pink": 7, "red": 39, "green": 36}


def colour_surcharge_011(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 11; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_011.get(key, 0) * max(qty, 0)


def clean_code_011(raw: str) -> str:
    """Normalise a promo code of group 11: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_011(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 11; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_011(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 11: members over 709, or anyone over 1418 with 3+ items."""
    if is_member and subtotal >= 709:
        return True
    return subtotal >= 1418 and items >= 3


def tier_discount_012(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 12: 9% above 102, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 102:
        rate += 9 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_012 = {"pink": 18, "grey": 2, "red": 12, "navy": 4}


def colour_surcharge_012(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 12; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_012.get(key, 0) * max(qty, 0)


def clean_code_012(raw: str) -> str:
    """Normalise a promo code of group 12: letters and digits only, max 8 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 8:
        cleaned = cleaned[:8]
    return cleaned


def delivery_estimate_012(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 12; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_012(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 12: members over 789, or anyone over 1578 with 3+ items."""
    if is_member and subtotal >= 789:
        return True
    return subtotal >= 1578 and items >= 3


def tier_discount_013(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 13: 4% above 306, plus 8 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 306:
        rate += 4 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 8 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_013 = {"black": 31, "pink": 21, "teal": 18, "green": 47}


def colour_surcharge_013(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 13; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_013.get(key, 0) * max(qty, 0)


def clean_code_013(raw: str) -> str:
    """Normalise a promo code of group 13: letters and digits only, max 20 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 20:
        cleaned = cleaned[:20]
    return cleaned


def delivery_estimate_013(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 13; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_013(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 13: members over 833, or anyone over 1666 with 3+ items."""
    if is_member and subtotal >= 833:
        return True
    return subtotal >= 1666 and items >= 3


def tier_discount_014(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 14: 7% above 310, plus 3 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 310:
        rate += 7 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 3 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_014 = {"teal": 9, "navy": 14, "white": 11, "grey": 20}


def colour_surcharge_014(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 14; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_014.get(key, 0) * max(qty, 0)


def clean_code_014(raw: str) -> str:
    """Normalise a promo code of group 14: letters and digits only, max 18 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 18:
        cleaned = cleaned[:18]
    return cleaned


def delivery_estimate_014(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 14; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_014(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 14: members over 870, or anyone over 1740 with 3+ items."""
    if is_member and subtotal >= 870:
        return True
    return subtotal >= 1740 and items >= 3


def tier_discount_015(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 15: 6% above 243, plus 2 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 243:
        rate += 6 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 2 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_015 = {"red": 1, "green": 50, "blue": 15, "grey": 49}


def colour_surcharge_015(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 15; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_015.get(key, 0) * max(qty, 0)


def clean_code_015(raw: str) -> str:
    """Normalise a promo code of group 15: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_015(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 15; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_015(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 15: members over 774, or anyone over 1548 with 3+ items."""
    if is_member and subtotal >= 774:
        return True
    return subtotal >= 1548 and items >= 3


def tier_discount_016(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 16: 4% above 321, plus 0 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 321:
        rate += 4 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 0 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_016 = {"teal": 28, "pink": 22, "white": 1, "red": 33}


def colour_surcharge_016(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 16; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_016.get(key, 0) * max(qty, 0)


def clean_code_016(raw: str) -> str:
    """Normalise a promo code of group 16: letters and digits only, max 16 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 16:
        cleaned = cleaned[:16]
    return cleaned


def delivery_estimate_016(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 16; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_016(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 16: members over 361, or anyone over 722 with 3+ items."""
    if is_member and subtotal >= 361:
        return True
    return subtotal >= 722 and items >= 3


def tier_discount_017(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 17: 3% above 125, plus 9 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 125:
        rate += 3 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 9 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_017 = {"green": 12, "navy": 2, "red": 22, "pink": 20}


def colour_surcharge_017(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 17; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_017.get(key, 0) * max(qty, 0)


def clean_code_017(raw: str) -> str:
    """Normalise a promo code of group 17: letters and digits only, max 14 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 14:
        cleaned = cleaned[:14]
    return cleaned


def delivery_estimate_017(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 17; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_017(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 17: members over 265, or anyone over 530 with 3+ items."""
    if is_member and subtotal >= 265:
        return True
    return subtotal >= 530 and items >= 3


def tier_discount_018(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 18: 8% above 232, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 232:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_018 = {"teal": 50, "grey": 8, "green": 8, "red": 24}


def colour_surcharge_018(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 18; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_018.get(key, 0) * max(qty, 0)


def clean_code_018(raw: str) -> str:
    """Normalise a promo code of group 18: letters and digits only, max 18 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 18:
        cleaned = cleaned[:18]
    return cleaned


def delivery_estimate_018(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 18; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_018(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 18: members over 617, or anyone over 1234 with 3+ items."""
    if is_member and subtotal >= 617:
        return True
    return subtotal >= 1234 and items >= 3


def tier_discount_019(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 19: 1% above 347, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 347:
        rate += 1 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_019 = {"navy": 15, "red": 8, "pink": 21, "blue": 35}


def colour_surcharge_019(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 19; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_019.get(key, 0) * max(qty, 0)


def clean_code_019(raw: str) -> str:
    """Normalise a promo code of group 19: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_019(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 19; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_019(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 19: members over 317, or anyone over 634 with 3+ items."""
    if is_member and subtotal >= 317:
        return True
    return subtotal >= 634 and items >= 3


def tier_discount_020(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 20: 2% above 230, plus 7 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 230:
        rate += 2 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 7 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_020 = {"green": 11, "white": 49, "black": 15, "teal": 1}


def colour_surcharge_020(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 20; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_020.get(key, 0) * max(qty, 0)


def clean_code_020(raw: str) -> str:
    """Normalise a promo code of group 20: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_020(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 20; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_020(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 20: members over 303, or anyone over 606 with 3+ items."""
    if is_member and subtotal >= 303:
        return True
    return subtotal >= 606 and items >= 3


def tier_discount_021(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 21: 9% above 303, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 303:
        rate += 9 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_021 = {"navy": 42, "teal": 16, "white": 36, "green": 49}


def colour_surcharge_021(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 21; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_021.get(key, 0) * max(qty, 0)


def clean_code_021(raw: str) -> str:
    """Normalise a promo code of group 21: letters and digits only, max 14 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 14:
        cleaned = cleaned[:14]
    return cleaned


def delivery_estimate_021(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 21; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_021(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 21: members over 705, or anyone over 1410 with 3+ items."""
    if is_member and subtotal >= 705:
        return True
    return subtotal >= 1410 and items >= 3


def tier_discount_022(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 22: 8% above 336, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 336:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_022 = {"green": 21, "navy": 33, "black": 34, "blue": 15}


def colour_surcharge_022(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 22; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_022.get(key, 0) * max(qty, 0)


def clean_code_022(raw: str) -> str:
    """Normalise a promo code of group 22: letters and digits only, max 12 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 12:
        cleaned = cleaned[:12]
    return cleaned


def delivery_estimate_022(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 22; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_022(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 22: members over 332, or anyone over 664 with 3+ items."""
    if is_member and subtotal >= 332:
        return True
    return subtotal >= 664 and items >= 3


def tier_discount_023(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 23: 8% above 303, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 303:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_023 = {"white": 13, "pink": 30, "grey": 42, "navy": 35}


def colour_surcharge_023(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 23; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_023.get(key, 0) * max(qty, 0)


def clean_code_023(raw: str) -> str:
    """Normalise a promo code of group 23: letters and digits only, max 6 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 6:
        cleaned = cleaned[:6]
    return cleaned


def delivery_estimate_023(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 23; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_023(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 23: members over 489, or anyone over 978 with 3+ items."""
    if is_member and subtotal >= 489:
        return True
    return subtotal >= 978 and items >= 3


def tier_discount_024(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 24: 4% above 144, plus 8 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 144:
        rate += 4 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 8 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_024 = {"navy": 19, "blue": 45, "grey": 35, "teal": 9}


def colour_surcharge_024(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 24; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_024.get(key, 0) * max(qty, 0)


def clean_code_024(raw: str) -> str:
    """Normalise a promo code of group 24: letters and digits only, max 8 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 8:
        cleaned = cleaned[:8]
    return cleaned


def delivery_estimate_024(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 24; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_024(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 24: members over 325, or anyone over 650 with 3+ items."""
    if is_member and subtotal >= 325:
        return True
    return subtotal >= 650 and items >= 3


def tier_discount_025(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 25: 5% above 161, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 161:
        rate += 5 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_025 = {"black": 12, "navy": 31, "grey": 24, "blue": 2}


def colour_surcharge_025(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 25; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_025.get(key, 0) * max(qty, 0)


def clean_code_025(raw: str) -> str:
    """Normalise a promo code of group 25: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_025(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 25; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_025(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 25: members over 811, or anyone over 1622 with 3+ items."""
    if is_member and subtotal >= 811:
        return True
    return subtotal >= 1622 and items >= 3


def shipping_cost(weight_kg: float, zone: str) -> float:
    """Shipping price for a parcel: zone base rate plus a per-kilogram charge."""
    if weight_kg <= 0:
        raise ValueError("weight must be positive")
    try:
        base = ZONE_BASE[zone]
    except KeyError:
        raise ValueError(f"unknown zone {zone!r}") from None
    return round(base + PER_KG * weight_kg, 2)


def tier_discount_026(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 26: 3% above 61, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 61:
        rate += 3 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_026 = {"navy": 49, "grey": 38, "green": 17, "red": 3}


def colour_surcharge_026(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 26; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_026.get(key, 0) * max(qty, 0)


def clean_code_026(raw: str) -> str:
    """Normalise a promo code of group 26: letters and digits only, max 14 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 14:
        cleaned = cleaned[:14]
    return cleaned


def delivery_estimate_026(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 26; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_026(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 26: members over 293, or anyone over 586 with 3+ items."""
    if is_member and subtotal >= 293:
        return True
    return subtotal >= 586 and items >= 3


def tier_discount_027(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 27: 3% above 108, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 108:
        rate += 3 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_027 = {"teal": 8, "white": 7, "red": 38, "black": 18}


def colour_surcharge_027(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 27; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_027.get(key, 0) * max(qty, 0)


def clean_code_027(raw: str) -> str:
    """Normalise a promo code of group 27: letters and digits only, max 10 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 10:
        cleaned = cleaned[:10]
    return cleaned


def delivery_estimate_027(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 27; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_027(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 27: members over 703, or anyone over 1406 with 3+ items."""
    if is_member and subtotal >= 703:
        return True
    return subtotal >= 1406 and items >= 3


def tier_discount_028(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 28: 2% above 124, plus 8 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 124:
        rate += 2 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 8 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_028 = {"black": 9, "blue": 45, "teal": 25, "grey": 44}


def colour_surcharge_028(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 28; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_028.get(key, 0) * max(qty, 0)


def clean_code_028(raw: str) -> str:
    """Normalise a promo code of group 28: letters and digits only, max 8 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 8:
        cleaned = cleaned[:8]
    return cleaned


def delivery_estimate_028(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 28; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_028(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 28: members over 446, or anyone over 892 with 3+ items."""
    if is_member and subtotal >= 446:
        return True
    return subtotal >= 892 and items >= 3


def tier_discount_029(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 29: 6% above 332, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 332:
        rate += 6 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_029 = {"grey": 18, "black": 25, "red": 9, "white": 37}


def colour_surcharge_029(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 29; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_029.get(key, 0) * max(qty, 0)


def clean_code_029(raw: str) -> str:
    """Normalise a promo code of group 29: letters and digits only, max 14 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 14:
        cleaned = cleaned[:14]
    return cleaned


def delivery_estimate_029(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 29; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_029(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 29: members over 618, or anyone over 1236 with 3+ items."""
    if is_member and subtotal >= 618:
        return True
    return subtotal >= 1236 and items >= 3


def tier_discount_030(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 30: 4% above 290, plus 3 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 290:
        rate += 4 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 3 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_030 = {"blue": 13, "navy": 9, "grey": 9, "white": 44}


def colour_surcharge_030(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 30; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_030.get(key, 0) * max(qty, 0)


def clean_code_030(raw: str) -> str:
    """Normalise a promo code of group 30: letters and digits only, max 12 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 12:
        cleaned = cleaned[:12]
    return cleaned


def delivery_estimate_030(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 30; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_030(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 30: members over 125, or anyone over 250 with 3+ items."""
    if is_member and subtotal >= 125:
        return True
    return subtotal >= 250 and items >= 3


def tier_discount_031(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 31: 2% above 226, plus 8 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 226:
        rate += 2 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 8 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_031 = {"pink": 10, "navy": 18, "red": 46, "black": 24}


def colour_surcharge_031(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 31; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_031.get(key, 0) * max(qty, 0)


def clean_code_031(raw: str) -> str:
    """Normalise a promo code of group 31: letters and digits only, max 20 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 20:
        cleaned = cleaned[:20]
    return cleaned


def delivery_estimate_031(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 31; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_031(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 31: members over 398, or anyone over 796 with 3+ items."""
    if is_member and subtotal >= 398:
        return True
    return subtotal >= 796 and items >= 3


def tier_discount_032(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 32: 8% above 292, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 292:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_032 = {"red": 42, "white": 32, "green": 41, "black": 24}


def colour_surcharge_032(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 32; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_032.get(key, 0) * max(qty, 0)


def clean_code_032(raw: str) -> str:
    """Normalise a promo code of group 32: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_032(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 32; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_032(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 32: members over 472, or anyone over 944 with 3+ items."""
    if is_member and subtotal >= 472:
        return True
    return subtotal >= 944 and items >= 3


def tier_discount_033(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 33: 2% above 269, plus 2 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 269:
        rate += 2 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 2 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_033 = {"pink": 29, "white": 30, "green": 21, "navy": 40}


def colour_surcharge_033(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 33; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_033.get(key, 0) * max(qty, 0)


def clean_code_033(raw: str) -> str:
    """Normalise a promo code of group 33: letters and digits only, max 6 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 6:
        cleaned = cleaned[:6]
    return cleaned


def delivery_estimate_033(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 33; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_033(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 33: members over 852, or anyone over 1704 with 3+ items."""
    if is_member and subtotal >= 852:
        return True
    return subtotal >= 1704 and items >= 3


def tier_discount_034(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 34: 7% above 35, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 35:
        rate += 7 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_034 = {"white": 43, "green": 31, "grey": 30, "red": 14}


def colour_surcharge_034(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 34; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_034.get(key, 0) * max(qty, 0)


def clean_code_034(raw: str) -> str:
    """Normalise a promo code of group 34: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_034(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 34; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_034(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 34: members over 790, or anyone over 1580 with 3+ items."""
    if is_member and subtotal >= 790:
        return True
    return subtotal >= 1580 and items >= 3


def tier_discount_035(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 35: 6% above 345, plus 3 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 345:
        rate += 6 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 3 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_035 = {"pink": 38, "white": 4, "black": 44, "blue": 14}


def colour_surcharge_035(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 35; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_035.get(key, 0) * max(qty, 0)


def clean_code_035(raw: str) -> str:
    """Normalise a promo code of group 35: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_035(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 35; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_035(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 35: members over 829, or anyone over 1658 with 3+ items."""
    if is_member and subtotal >= 829:
        return True
    return subtotal >= 1658 and items >= 3


def tier_discount_036(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 36: 5% above 342, plus 3 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 342:
        rate += 5 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 3 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_036 = {"grey": 9, "white": 40, "blue": 35, "red": 35}


def colour_surcharge_036(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 36; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_036.get(key, 0) * max(qty, 0)


def clean_code_036(raw: str) -> str:
    """Normalise a promo code of group 36: letters and digits only, max 14 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 14:
        cleaned = cleaned[:14]
    return cleaned


def delivery_estimate_036(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 36; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_036(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 36: members over 453, or anyone over 906 with 3+ items."""
    if is_member and subtotal >= 453:
        return True
    return subtotal >= 906 and items >= 3


def tier_discount_037(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 37: 1% above 188, plus 2 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 188:
        rate += 1 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 2 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_037 = {"red": 13, "grey": 12, "navy": 19, "blue": 30}


def colour_surcharge_037(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 37; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_037.get(key, 0) * max(qty, 0)


def clean_code_037(raw: str) -> str:
    """Normalise a promo code of group 37: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_037(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 37; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_037(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 37: members over 263, or anyone over 526 with 3+ items."""
    if is_member and subtotal >= 263:
        return True
    return subtotal >= 526 and items >= 3


def tier_discount_038(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 38: 8% above 336, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 336:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_038 = {"white": 13, "navy": 44, "red": 9, "black": 9}


def colour_surcharge_038(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 38; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_038.get(key, 0) * max(qty, 0)


def clean_code_038(raw: str) -> str:
    """Normalise a promo code of group 38: letters and digits only, max 16 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 16:
        cleaned = cleaned[:16]
    return cleaned


def delivery_estimate_038(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 38; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_038(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 38: members over 563, or anyone over 1126 with 3+ items."""
    if is_member and subtotal >= 563:
        return True
    return subtotal >= 1126 and items >= 3


def tier_discount_039(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 39: 7% above 136, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 136:
        rate += 7 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_039 = {"green": 29, "grey": 21, "navy": 1, "white": 48}


def colour_surcharge_039(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 39; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_039.get(key, 0) * max(qty, 0)


def clean_code_039(raw: str) -> str:
    """Normalise a promo code of group 39: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_039(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 39; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_039(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 39: members over 523, or anyone over 1046 with 3+ items."""
    if is_member and subtotal >= 523:
        return True
    return subtotal >= 1046 and items >= 3


def tier_discount_040(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 40: 8% above 113, plus 8 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 113:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 8 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_040 = {"blue": 27, "green": 32, "white": 26, "grey": 32}


def colour_surcharge_040(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 40; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_040.get(key, 0) * max(qty, 0)


def clean_code_040(raw: str) -> str:
    """Normalise a promo code of group 40: letters and digits only, max 20 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 20:
        cleaned = cleaned[:20]
    return cleaned


def delivery_estimate_040(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 40; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_040(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 40: members over 145, or anyone over 290 with 3+ items."""
    if is_member and subtotal >= 145:
        return True
    return subtotal >= 290 and items >= 3


def tier_discount_041(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 41: 9% above 48, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 48:
        rate += 9 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_041 = {"white": 35, "green": 24, "blue": 5, "grey": 21}


def colour_surcharge_041(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 41; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_041.get(key, 0) * max(qty, 0)


def clean_code_041(raw: str) -> str:
    """Normalise a promo code of group 41: letters and digits only, max 20 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 20:
        cleaned = cleaned[:20]
    return cleaned


def delivery_estimate_041(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 41; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_041(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 41: members over 707, or anyone over 1414 with 3+ items."""
    if is_member and subtotal >= 707:
        return True
    return subtotal >= 1414 and items >= 3


def tier_discount_042(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 42: 5% above 339, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 339:
        rate += 5 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_042 = {"navy": 17, "pink": 8, "white": 2, "grey": 16}


def colour_surcharge_042(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 42; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_042.get(key, 0) * max(qty, 0)


def clean_code_042(raw: str) -> str:
    """Normalise a promo code of group 42: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_042(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 42; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_042(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 42: members over 267, or anyone over 534 with 3+ items."""
    if is_member and subtotal >= 267:
        return True
    return subtotal >= 534 and items >= 3


def tier_discount_043(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 43: 4% above 17, plus 2 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 17:
        rate += 4 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 2 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_043 = {"red": 22, "white": 11, "pink": 15, "navy": 16}


def colour_surcharge_043(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 43; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_043.get(key, 0) * max(qty, 0)


def clean_code_043(raw: str) -> str:
    """Normalise a promo code of group 43: letters and digits only, max 22 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 22:
        cleaned = cleaned[:22]
    return cleaned


def delivery_estimate_043(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 43; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_043(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 43: members over 898, or anyone over 1796 with 3+ items."""
    if is_member and subtotal >= 898:
        return True
    return subtotal >= 1796 and items >= 3


def tier_discount_044(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 44: 3% above 31, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 31:
        rate += 3 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_044 = {"white": 24, "grey": 36, "red": 11, "pink": 7}


def colour_surcharge_044(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 44; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_044.get(key, 0) * max(qty, 0)


def clean_code_044(raw: str) -> str:
    """Normalise a promo code of group 44: letters and digits only, max 16 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 16:
        cleaned = cleaned[:16]
    return cleaned


def delivery_estimate_044(ordered: date, handling_days: int = 2) -> date:
    """Business-day estimate for lane 44; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_044(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 44: members over 113, or anyone over 226 with 3+ items."""
    if is_member and subtotal >= 113:
        return True
    return subtotal >= 226 and items >= 3


def tier_discount_045(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 45: 9% above 398, plus 7 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 398:
        rate += 9 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 7 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_045 = {"pink": 49, "navy": 16, "black": 3, "white": 44}


def colour_surcharge_045(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 45; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_045.get(key, 0) * max(qty, 0)


def clean_code_045(raw: str) -> str:
    """Normalise a promo code of group 45: letters and digits only, max 10 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 10:
        cleaned = cleaned[:10]
    return cleaned


def delivery_estimate_045(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 45; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_045(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 45: members over 372, or anyone over 744 with 3+ items."""
    if is_member and subtotal >= 372:
        return True
    return subtotal >= 744 and items >= 3


def tier_discount_046(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 46: 1% above 245, plus 7 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 245:
        rate += 1 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 7 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_046 = {"black": 10, "grey": 14, "pink": 22, "red": 5}


def colour_surcharge_046(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 46; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_046.get(key, 0) * max(qty, 0)


def clean_code_046(raw: str) -> str:
    """Normalise a promo code of group 46: letters and digits only, max 24 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 24:
        cleaned = cleaned[:24]
    return cleaned


def delivery_estimate_046(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 46; weekends do not count."""
    current = ordered
    remaining = handling_days + 4
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_046(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 46: members over 570, or anyone over 1140 with 3+ items."""
    if is_member and subtotal >= 570:
        return True
    return subtotal >= 1140 and items >= 3


def tier_discount_047(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 47: 2% above 127, plus 1 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 127:
        rate += 2 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 1 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_047 = {"red": 41, "green": 30, "navy": 14, "teal": 45}


def colour_surcharge_047(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 47; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_047.get(key, 0) * max(qty, 0)


def clean_code_047(raw: str) -> str:
    """Normalise a promo code of group 47: letters and digits only, max 12 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 12:
        cleaned = cleaned[:12]
    return cleaned


def delivery_estimate_047(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 47; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_047(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 47: members over 582, or anyone over 1164 with 3+ items."""
    if is_member and subtotal >= 582:
        return True
    return subtotal >= 1164 and items >= 3


def tier_discount_048(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 48: 2% above 344, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 344:
        rate += 2 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_048 = {"navy": 41, "pink": 46, "white": 46, "blue": 10}


def colour_surcharge_048(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 48; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_048.get(key, 0) * max(qty, 0)


def clean_code_048(raw: str) -> str:
    """Normalise a promo code of group 48: letters and digits only, max 12 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 12:
        cleaned = cleaned[:12]
    return cleaned


def delivery_estimate_048(ordered: date, handling_days: int = 1) -> date:
    """Business-day estimate for lane 48; weekends do not count."""
    current = ordered
    remaining = handling_days + 2
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_048(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 48: members over 859, or anyone over 1718 with 3+ items."""
    if is_member and subtotal >= 859:
        return True
    return subtotal >= 1718 and items >= 3


def tier_discount_049(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 49: 3% above 255, plus 3 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 255:
        rate += 3 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 3 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_049 = {"red": 44, "blue": 16, "teal": 44, "black": 28}


def colour_surcharge_049(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 49; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_049.get(key, 0) * max(qty, 0)


def clean_code_049(raw: str) -> str:
    """Normalise a promo code of group 49: letters and digits only, max 18 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 18:
        cleaned = cleaned[:18]
    return cleaned


def delivery_estimate_049(ordered: date, handling_days: int = 3) -> date:
    """Business-day estimate for lane 49; weekends do not count."""
    current = ordered
    remaining = handling_days + 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_049(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 49: members over 494, or anyone over 988 with 3+ items."""
    if is_member and subtotal >= 494:
        return True
    return subtotal >= 988 and items >= 3


def tier_discount_050(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 50: 8% above 302, plus 5 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 302:
        rate += 8 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 5 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_050 = {"green": 40, "white": 10, "teal": 21, "blue": 8}


def colour_surcharge_050(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 50; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_050.get(key, 0) * max(qty, 0)


def clean_code_050(raw: str) -> str:
    """Normalise a promo code of group 50: letters and digits only, max 16 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 16:
        cleaned = cleaned[:16]
    return cleaned


def delivery_estimate_050(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 50; weekends do not count."""
    current = ordered
    remaining = handling_days + 3
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_050(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 50: members over 455, or anyone over 910 with 3+ items."""
    if is_member and subtotal >= 455:
        return True
    return subtotal >= 910 and items >= 3


def tier_discount_051(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 51: 2% above 307, plus 6 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 307:
        rate += 2 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 6 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_051 = {"black": 30, "blue": 7, "white": 39, "green": 39}


def colour_surcharge_051(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 51; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_051.get(key, 0) * max(qty, 0)


def clean_code_051(raw: str) -> str:
    """Normalise a promo code of group 51: letters and digits only, max 18 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(cleaned) > 18:
        cleaned = cleaned[:18]
    return cleaned


def delivery_estimate_051(ordered: date, handling_days: int = 4) -> date:
    """Business-day estimate for lane 51; weekends do not count."""
    current = ordered
    remaining = handling_days + 0
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def eligible_for_gift_051(subtotal: float, items: int, is_member: bool) -> bool:
    """Gift campaign 51: members over 837, or anyone over 1674 with 3+ items."""
    if is_member and subtotal >= 837:
        return True
    return subtotal >= 1674 and items >= 3


def tier_discount_052(subtotal: float, loyalty_years: int = 0) -> float:
    """Discount for tier 52: 7% above 386, plus 7 extra points per loyalty year."""
    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    rate = 0.0
    if subtotal >= 386:
        rate += 7 / 100
    if loyalty_years > 0:
        rate += min(loyalty_years, 5) * 7 / 1000
    return round(subtotal * min(rate, 0.30), 2)


_SURCHARGE_052 = {"teal": 21, "pink": 40, "blue": 34, "white": 16}


def colour_surcharge_052(colour: str, qty: int = 1) -> int:
    """Per-unit surcharge for colour group 52; unknown colours cost nothing."""
    key = colour.strip().lower()
    return _SURCHARGE_052.get(key, 0) * max(qty, 0)


def quote_total(subtotal: float, weight_kg: float, zone: str) -> float:
    """Subtotal plus shipping, as shown on the checkout page."""
    return round(subtotal + shipping_cost(weight_kg, zone), 2)


RULES = {
    "tier_discount_000": tier_discount_000,
    "colour_surcharge_000": colour_surcharge_000,
    "clean_code_000": clean_code_000,
    "delivery_estimate_000": delivery_estimate_000,
    "eligible_for_gift_000": eligible_for_gift_000,
    "tier_discount_001": tier_discount_001,
    "colour_surcharge_001": colour_surcharge_001,
    "clean_code_001": clean_code_001,
    "delivery_estimate_001": delivery_estimate_001,
    "eligible_for_gift_001": eligible_for_gift_001,
    "tier_discount_002": tier_discount_002,
    "colour_surcharge_002": colour_surcharge_002,
    "clean_code_002": clean_code_002,
    "delivery_estimate_002": delivery_estimate_002,
    "eligible_for_gift_002": eligible_for_gift_002,
    "tier_discount_003": tier_discount_003,
    "colour_surcharge_003": colour_surcharge_003,
    "clean_code_003": clean_code_003,
    "delivery_estimate_003": delivery_estimate_003,
    "eligible_for_gift_003": eligible_for_gift_003,
    "tier_discount_004": tier_discount_004,
    "colour_surcharge_004": colour_surcharge_004,
    "clean_code_004": clean_code_004,
    "delivery_estimate_004": delivery_estimate_004,
    "eligible_for_gift_004": eligible_for_gift_004,
    "tier_discount_005": tier_discount_005,
    "colour_surcharge_005": colour_surcharge_005,
    "clean_code_005": clean_code_005,
    "delivery_estimate_005": delivery_estimate_005,
    "eligible_for_gift_005": eligible_for_gift_005,
    "tier_discount_006": tier_discount_006,
    "colour_surcharge_006": colour_surcharge_006,
    "clean_code_006": clean_code_006,
    "delivery_estimate_006": delivery_estimate_006,
    "eligible_for_gift_006": eligible_for_gift_006,
    "tier_discount_007": tier_discount_007,
    "colour_surcharge_007": colour_surcharge_007,
    "clean_code_007": clean_code_007,
    "delivery_estimate_007": delivery_estimate_007,
    "eligible_for_gift_007": eligible_for_gift_007,
    "tier_discount_008": tier_discount_008,
    "colour_surcharge_008": colour_surcharge_008,
    "clean_code_008": clean_code_008,
    "delivery_estimate_008": delivery_estimate_008,
    "eligible_for_gift_008": eligible_for_gift_008,
    "tier_discount_009": tier_discount_009,
    "colour_surcharge_009": colour_surcharge_009,
    "clean_code_009": clean_code_009,
    "delivery_estimate_009": delivery_estimate_009,
    "eligible_for_gift_009": eligible_for_gift_009,
    "tier_discount_010": tier_discount_010,
    "colour_surcharge_010": colour_surcharge_010,
    "clean_code_010": clean_code_010,
    "delivery_estimate_010": delivery_estimate_010,
    "eligible_for_gift_010": eligible_for_gift_010,
    "tier_discount_011": tier_discount_011,
    "colour_surcharge_011": colour_surcharge_011,
    "clean_code_011": clean_code_011,
    "delivery_estimate_011": delivery_estimate_011,
    "eligible_for_gift_011": eligible_for_gift_011,
    "tier_discount_012": tier_discount_012,
    "colour_surcharge_012": colour_surcharge_012,
    "clean_code_012": clean_code_012,
    "delivery_estimate_012": delivery_estimate_012,
    "eligible_for_gift_012": eligible_for_gift_012,
    "tier_discount_013": tier_discount_013,
    "colour_surcharge_013": colour_surcharge_013,
    "clean_code_013": clean_code_013,
    "delivery_estimate_013": delivery_estimate_013,
    "eligible_for_gift_013": eligible_for_gift_013,
    "tier_discount_014": tier_discount_014,
    "colour_surcharge_014": colour_surcharge_014,
    "clean_code_014": clean_code_014,
    "delivery_estimate_014": delivery_estimate_014,
    "eligible_for_gift_014": eligible_for_gift_014,
    "tier_discount_015": tier_discount_015,
    "colour_surcharge_015": colour_surcharge_015,
    "clean_code_015": clean_code_015,
    "delivery_estimate_015": delivery_estimate_015,
    "eligible_for_gift_015": eligible_for_gift_015,
    "tier_discount_016": tier_discount_016,
    "colour_surcharge_016": colour_surcharge_016,
    "clean_code_016": clean_code_016,
    "delivery_estimate_016": delivery_estimate_016,
    "eligible_for_gift_016": eligible_for_gift_016,
    "tier_discount_017": tier_discount_017,
    "colour_surcharge_017": colour_surcharge_017,
    "clean_code_017": clean_code_017,
    "delivery_estimate_017": delivery_estimate_017,
    "eligible_for_gift_017": eligible_for_gift_017,
    "tier_discount_018": tier_discount_018,
    "colour_surcharge_018": colour_surcharge_018,
    "clean_code_018": clean_code_018,
    "delivery_estimate_018": delivery_estimate_018,
    "eligible_for_gift_018": eligible_for_gift_018,
    "tier_discount_019": tier_discount_019,
    "colour_surcharge_019": colour_surcharge_019,
    "clean_code_019": clean_code_019,
    "delivery_estimate_019": delivery_estimate_019,
    "eligible_for_gift_019": eligible_for_gift_019,
    "tier_discount_020": tier_discount_020,
    "colour_surcharge_020": colour_surcharge_020,
    "clean_code_020": clean_code_020,
    "delivery_estimate_020": delivery_estimate_020,
    "eligible_for_gift_020": eligible_for_gift_020,
    "tier_discount_021": tier_discount_021,
    "colour_surcharge_021": colour_surcharge_021,
    "clean_code_021": clean_code_021,
    "delivery_estimate_021": delivery_estimate_021,
    "eligible_for_gift_021": eligible_for_gift_021,
    "tier_discount_022": tier_discount_022,
    "colour_surcharge_022": colour_surcharge_022,
    "clean_code_022": clean_code_022,
    "delivery_estimate_022": delivery_estimate_022,
    "eligible_for_gift_022": eligible_for_gift_022,
    "tier_discount_023": tier_discount_023,
    "colour_surcharge_023": colour_surcharge_023,
    "clean_code_023": clean_code_023,
    "delivery_estimate_023": delivery_estimate_023,
    "eligible_for_gift_023": eligible_for_gift_023,
    "tier_discount_024": tier_discount_024,
    "colour_surcharge_024": colour_surcharge_024,
    "clean_code_024": clean_code_024,
    "delivery_estimate_024": delivery_estimate_024,
    "eligible_for_gift_024": eligible_for_gift_024,
    "tier_discount_025": tier_discount_025,
    "colour_surcharge_025": colour_surcharge_025,
    "clean_code_025": clean_code_025,
    "delivery_estimate_025": delivery_estimate_025,
    "eligible_for_gift_025": eligible_for_gift_025,
    "shipping_cost": shipping_cost,
    "tier_discount_026": tier_discount_026,
    "colour_surcharge_026": colour_surcharge_026,
    "clean_code_026": clean_code_026,
    "delivery_estimate_026": delivery_estimate_026,
    "eligible_for_gift_026": eligible_for_gift_026,
    "tier_discount_027": tier_discount_027,
    "colour_surcharge_027": colour_surcharge_027,
    "clean_code_027": clean_code_027,
    "delivery_estimate_027": delivery_estimate_027,
    "eligible_for_gift_027": eligible_for_gift_027,
    "tier_discount_028": tier_discount_028,
    "colour_surcharge_028": colour_surcharge_028,
    "clean_code_028": clean_code_028,
    "delivery_estimate_028": delivery_estimate_028,
    "eligible_for_gift_028": eligible_for_gift_028,
    "tier_discount_029": tier_discount_029,
    "colour_surcharge_029": colour_surcharge_029,
    "clean_code_029": clean_code_029,
    "delivery_estimate_029": delivery_estimate_029,
    "eligible_for_gift_029": eligible_for_gift_029,
    "tier_discount_030": tier_discount_030,
    "colour_surcharge_030": colour_surcharge_030,
    "clean_code_030": clean_code_030,
    "delivery_estimate_030": delivery_estimate_030,
    "eligible_for_gift_030": eligible_for_gift_030,
    "tier_discount_031": tier_discount_031,
    "colour_surcharge_031": colour_surcharge_031,
    "clean_code_031": clean_code_031,
    "delivery_estimate_031": delivery_estimate_031,
    "eligible_for_gift_031": eligible_for_gift_031,
    "tier_discount_032": tier_discount_032,
    "colour_surcharge_032": colour_surcharge_032,
    "clean_code_032": clean_code_032,
    "delivery_estimate_032": delivery_estimate_032,
    "eligible_for_gift_032": eligible_for_gift_032,
    "tier_discount_033": tier_discount_033,
    "colour_surcharge_033": colour_surcharge_033,
    "clean_code_033": clean_code_033,
    "delivery_estimate_033": delivery_estimate_033,
    "eligible_for_gift_033": eligible_for_gift_033,
    "tier_discount_034": tier_discount_034,
    "colour_surcharge_034": colour_surcharge_034,
    "clean_code_034": clean_code_034,
    "delivery_estimate_034": delivery_estimate_034,
    "eligible_for_gift_034": eligible_for_gift_034,
    "tier_discount_035": tier_discount_035,
    "colour_surcharge_035": colour_surcharge_035,
    "clean_code_035": clean_code_035,
    "delivery_estimate_035": delivery_estimate_035,
    "eligible_for_gift_035": eligible_for_gift_035,
    "tier_discount_036": tier_discount_036,
    "colour_surcharge_036": colour_surcharge_036,
    "clean_code_036": clean_code_036,
    "delivery_estimate_036": delivery_estimate_036,
    "eligible_for_gift_036": eligible_for_gift_036,
    "tier_discount_037": tier_discount_037,
    "colour_surcharge_037": colour_surcharge_037,
    "clean_code_037": clean_code_037,
    "delivery_estimate_037": delivery_estimate_037,
    "eligible_for_gift_037": eligible_for_gift_037,
    "tier_discount_038": tier_discount_038,
    "colour_surcharge_038": colour_surcharge_038,
    "clean_code_038": clean_code_038,
    "delivery_estimate_038": delivery_estimate_038,
    "eligible_for_gift_038": eligible_for_gift_038,
    "tier_discount_039": tier_discount_039,
    "colour_surcharge_039": colour_surcharge_039,
    "clean_code_039": clean_code_039,
    "delivery_estimate_039": delivery_estimate_039,
    "eligible_for_gift_039": eligible_for_gift_039,
    "tier_discount_040": tier_discount_040,
    "colour_surcharge_040": colour_surcharge_040,
    "clean_code_040": clean_code_040,
    "delivery_estimate_040": delivery_estimate_040,
    "eligible_for_gift_040": eligible_for_gift_040,
    "tier_discount_041": tier_discount_041,
    "colour_surcharge_041": colour_surcharge_041,
    "clean_code_041": clean_code_041,
    "delivery_estimate_041": delivery_estimate_041,
    "eligible_for_gift_041": eligible_for_gift_041,
    "tier_discount_042": tier_discount_042,
    "colour_surcharge_042": colour_surcharge_042,
    "clean_code_042": clean_code_042,
    "delivery_estimate_042": delivery_estimate_042,
    "eligible_for_gift_042": eligible_for_gift_042,
    "tier_discount_043": tier_discount_043,
    "colour_surcharge_043": colour_surcharge_043,
    "clean_code_043": clean_code_043,
    "delivery_estimate_043": delivery_estimate_043,
    "eligible_for_gift_043": eligible_for_gift_043,
    "tier_discount_044": tier_discount_044,
    "colour_surcharge_044": colour_surcharge_044,
    "clean_code_044": clean_code_044,
    "delivery_estimate_044": delivery_estimate_044,
    "eligible_for_gift_044": eligible_for_gift_044,
    "tier_discount_045": tier_discount_045,
    "colour_surcharge_045": colour_surcharge_045,
    "clean_code_045": clean_code_045,
    "delivery_estimate_045": delivery_estimate_045,
    "eligible_for_gift_045": eligible_for_gift_045,
    "tier_discount_046": tier_discount_046,
    "colour_surcharge_046": colour_surcharge_046,
    "clean_code_046": clean_code_046,
    "delivery_estimate_046": delivery_estimate_046,
    "eligible_for_gift_046": eligible_for_gift_046,
    "tier_discount_047": tier_discount_047,
    "colour_surcharge_047": colour_surcharge_047,
    "clean_code_047": clean_code_047,
    "delivery_estimate_047": delivery_estimate_047,
    "eligible_for_gift_047": eligible_for_gift_047,
    "tier_discount_048": tier_discount_048,
    "colour_surcharge_048": colour_surcharge_048,
    "clean_code_048": clean_code_048,
    "delivery_estimate_048": delivery_estimate_048,
    "eligible_for_gift_048": eligible_for_gift_048,
    "tier_discount_049": tier_discount_049,
    "colour_surcharge_049": colour_surcharge_049,
    "clean_code_049": clean_code_049,
    "delivery_estimate_049": delivery_estimate_049,
    "eligible_for_gift_049": eligible_for_gift_049,
    "tier_discount_050": tier_discount_050,
    "colour_surcharge_050": colour_surcharge_050,
    "clean_code_050": clean_code_050,
    "delivery_estimate_050": delivery_estimate_050,
    "eligible_for_gift_050": eligible_for_gift_050,
    "tier_discount_051": tier_discount_051,
    "colour_surcharge_051": colour_surcharge_051,
    "clean_code_051": clean_code_051,
    "delivery_estimate_051": delivery_estimate_051,
    "eligible_for_gift_051": eligible_for_gift_051,
    "tier_discount_052": tier_discount_052,
    "colour_surcharge_052": colour_surcharge_052,
}
