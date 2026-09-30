"""Reject answers that claim work or local inspection without tool evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\r?\n+")
_WORK_CLAIM = re.compile(
    r"\b(?:I(?:['\N{RIGHT SINGLE QUOTATION MARK}]ve| have)?|"
    r"we(?:['\N{RIGHT SINGLE QUOTATION MARK}]ve| have)?)\s+(?:just\s+|already\s+)?"
    r"(?:analyzed|applied|built|calculated|checked|compiled|committed|configured|created|deleted|"
    r"deployed|downloaded|drafted|edited|"
    r"examined|executed|exported|fetched|fixed|generated|implemented|imported|inspected|installed|"
    r"migrated|modified|opened|organized|packaged|patched|read|removed|rendered|reviewed|ran|run|"
    r"saved|searched|tested|updated|verified|wrote|written)\b",
    re.IGNORECASE,
)
_TERSE_WORK_CLAIM = re.compile(
    r"^(?:[-*]\s*)?(?:(?:done|completed|finished)(?:\s*[:\N{EM DASH}-]\s*)?|successfully\s+)?"
    r"(?:applied|built|calculated|compiled|committed|configured|created|deleted|deployed|downloaded|"
    r"drafted|edited|exported|fetched|fixed|generated|implemented|imported|installed|migrated|"
    r"modified|organized|packaged|patched|removed|rendered|ran|saved|tested|updated|verified|wrote)\b",
    re.IGNORECASE,
)
_COMPLETION_CLAIM = re.compile(r"^(?:[-*]\s*)?(?:done|completed|finished)[.!]?$", re.IGNORECASE)
_LOCAL_READ_CLAIM = re.compile(
    r"\bI\s+(?:analyzed|checked|examined|inspected|opened|reviewed|looked\s+at|searched)\b|"
    r"\bI\s+read\s+(?:your|the|this)\s+(?:file|readme|log|code|folder|project|directory)\b|"
    r"\baccording\s+to\s+(?:your|the)\s+(?:file|log|repo|project|folder|readme)\b|"
    r"\bin\s+your\s+(?:file|folder|repo|project|directory)\b",
    re.IGNORECASE,
)
_REFUSAL = (
    "I have not done that. Ask me to do it and I will use tools, then report only what they return."
)


@dataclass(frozen=True)
class ClaimScreen:
    claims_unverified_work: bool
    without_work_claims: str


def screen_unverified(answer: str) -> ClaimScreen:
    """Drop sentences that assert actions or local reads the runtime did not perform."""
    text = answer.strip()
    chunks = [chunk.strip() for chunk in _SENTENCE.split(text) if chunk.strip()]
    claims_work = any(
        _WORK_CLAIM.search(chunk)
        or _TERSE_WORK_CLAIM.search(chunk)
        or _COMPLETION_CLAIM.search(chunk)
        or _LOCAL_READ_CLAIM.search(chunk)
        for chunk in chunks
    )
    if not text or not claims_work:
        return ClaimScreen(False, text)
    kept = [
        chunk
        for chunk in chunks
        if not _WORK_CLAIM.search(chunk)
        and not _TERSE_WORK_CLAIM.search(chunk)
        and not _COMPLETION_CLAIM.search(chunk)
        and not _LOCAL_READ_CLAIM.search(chunk)
    ]
    softened = " ".join(kept).strip() or _REFUSAL
    return ClaimScreen(True, softened)
