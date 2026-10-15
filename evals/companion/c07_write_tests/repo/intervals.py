"""Closed-open integer/float intervals [start, end) as (start, end) tuples."""

from __future__ import annotations

Interval = tuple[float, float]


def normalize(interval: Interval) -> Interval:
    """Validate an interval; zero-length and reversed intervals are errors."""
    start, end = interval
    if end <= start:
        raise ValueError(f"empty or reversed interval: {interval!r}")
    return (start, end)


def overlaps(a: Interval, b: Interval) -> bool:
    """True if the two intervals share at least one point. Touching ends do not overlap."""
    a, b = normalize(a), normalize(b)
    return a[0] < b[1] and b[0] < a[1]


def merge(intervals: list[Interval]) -> list[Interval]:
    """Union of intervals as a sorted list of disjoint intervals.

    Intervals that merely touch ([1, 2) and [2, 3)) are merged into one.
    """
    ordered = sorted(normalize(i) for i in intervals)
    merged: list[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def total_length(intervals: list[Interval]) -> float:
    """Length of the union; overlapping parts are counted once."""
    return sum(end - start for start, end in merge(intervals))


def subtract(base: Interval, holes: list[Interval]) -> list[Interval]:
    """The parts of `base` not covered by any hole, as sorted disjoint intervals."""
    start, end = normalize(base)
    result: list[Interval] = []
    cursor = start
    for h_start, h_end in merge(holes):
        if h_end <= cursor:
            continue
        if h_start >= end:
            break
        if h_start > cursor:
            result.append((cursor, h_start))
        cursor = max(cursor, h_end)
        if cursor >= end:
            break
    if cursor < end:
        result.append((cursor, end))
    return result


def clamp(interval: Interval, window: Interval) -> Interval | None:
    """The part of `interval` inside `window`, or None when they do not overlap."""
    lo = max(normalize(interval)[0], normalize(window)[0])
    hi = min(interval[1], window[1])
    return (lo, hi) if lo < hi else None


def contains(interval: Interval, point: float) -> bool:
    """Closed on the left, open on the right."""
    start, end = normalize(interval)
    return start <= point < end
