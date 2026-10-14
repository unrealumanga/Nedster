"""What the user asked for: a workspace change, an answer, or unknown.

``classify`` is deterministic and conservative. It splits the objective into sentences and clauses and looks
only at the head word of each clause (after polite request prefixes such as "please" / "can you"):

- an imperative edit verb ("fix", "add", "rename", ...) heading a clause that is not negated -> CHANGE
  (one such clause anywhere wins: "why does x fail? fix it" is a change request);
- otherwise an interrogative sentence ("how do I fix X?", "what ...", "is it ...?") or an information verb
  ("explain", "describe", "list", "show", ...) or an explicit "do not change anything" -> QUESTION;
- otherwise UNKNOWN (acceptance then behaves as it did before intents existed).

A caller-provided intent (``--expect change|answer``) always wins over the heuristic.
"""

from __future__ import annotations

import re
from enum import StrEnum

__all__ = ["CHANGE_VERBS", "Intent", "QUESTION_VERBS", "classify", "parse_expect"]


class Intent(StrEnum):
    CHANGE = "change"
    QUESTION = "question"
    UNKNOWN = "unknown"


CHANGE_VERBS = frozenset(
    {
        "add", "adjust", "append", "apply", "bump", "change", "clean", "convert", "correct", "create", "delete",
        "deprecate", "disable", "drop", "edit", "enable", "extract", "fix", "format", "generate", "implement",
        "inline", "insert", "introduce", "make", "merge", "migrate", "modify", "move", "optimize", "optimise",
        "patch", "port", "reformat", "refactor", "remove", "rename", "reorganize", "repair", "replace",
        "resolve", "restore", "restructure", "revert", "rewrite", "set", "simplify", "split", "support",
        "translate", "tweak", "update", "upgrade", "downgrade", "write",
    }
)
QUESTION_VERBS = frozenset(
    {
        "analyse", "analyze", "check", "clarify", "compare", "count", "describe", "explain", "find", "identify",
        "inspect", "list", "locate", "look", "outline", "read", "review", "search", "show", "summarise",
        "summarize", "tell", "understand", "walk",
    }
)
_WH_WORDS = frozenset({"how", "what", "what's", "whats", "when", "where", "which", "who", "whom", "whose", "why"})
_AUX_WORDS = frozenset(
    {
        "am", "are", "can", "could", "did", "do", "does", "has", "have", "is", "may", "might", "must", "shall",
        "should", "was", "were", "will", "would", "isn't", "aren't", "doesn't", "don't", "didn't", "can't",
        "won't", "shouldn't",
    }
)
# Request forms stripped from the start of a clause; "can you add" is a request, "can I add?" is a question.
_PREFIXES = tuple(
    p.split()
    for p in (
        "would you mind", "would you please", "could you please", "can you please", "i would like you to",
        "i'd like you to", "i want you to", "i need you to", "go ahead and", "help me", "can you", "could you",
        "would you", "will you", "can u", "please", "pls", "kindly", "let's", "lets", "just", "now", "also",
        "then", "and", "so", "if so", "if needed", "if necessary", "ok", "okay", "but", "first", "finally", "next", "instead",
    )
)
_NEGATIONS = (("do", "not"), ("don't",), ("dont",), ("never",), ("without",), ("no",))
_DETERMINERS = frozenset(
    {"a", "all", "an", "any", "it", "its", "my", "our", "that", "the", "them", "these", "this", "those", "your"}
)
_NO_EDIT_OBJECTS = CHANGE_VERBS | {"changes", "commit", "edits", "files", "modifications", "touch"}
_CODE = re.compile(r"`[^`]*`")
_SENTENCE = re.compile(r"[?!;]+|\.+(?=\s|$)|\n+")
_CLAUSE = re.compile(r",|\bthen\b")
_WORD = re.compile(r"[a-z][a-z'’]*")


def parse_expect(value: str | None) -> Intent | None:
    """``--expect`` flag value -> intent (``None`` when unset). Raises ``ValueError`` on anything else."""
    if value is None or not value.strip():
        return None
    key = value.strip().lower()
    mapping = {"change": Intent.CHANGE, "answer": Intent.QUESTION, "question": Intent.QUESTION}
    if key not in mapping:
        raise ValueError(f"--expect must be 'change' or 'answer', not {value!r}")
    return mapping[key]


def classify(objective: str, override: Intent | None = None) -> Intent:
    if override is not None:
        return Intent(override)
    text = _CODE.sub(" code ", objective.lower())
    kinds: set[str] = set()
    for sentence in _SENTENCE.split(text):
        kinds.update(_sentence_kinds(sentence))
    if "change" in kinds:
        return Intent.CHANGE
    if kinds & {"question", "no_edit"}:
        return Intent.QUESTION
    return Intent.UNKNOWN


def _words(text: str) -> list[str]:
    return [w.replace("’", "'") for w in _WORD.findall(text)]


def _strip_prefixes(words: list[str]) -> tuple[list[str], bool]:
    """Drop leading request forms; report whether one was a request ("can you ...")."""
    request = False
    changed = True
    while words and changed:
        changed = False
        for prefix in _PREFIXES:
            if words[: len(prefix)] == prefix:
                request = request or len(prefix) > 1 and prefix[-1] in {"you", "u", "to", "me", "and"}
                words = words[len(prefix):]
                changed = True
                break
    return words, request


def _negated(words: list[str]) -> bool:
    return any(words[: len(neg)] == list(neg) for neg in _NEGATIONS)


def _sentence_kinds(sentence: str) -> set[str]:
    words, request = _strip_prefixes(_words(sentence))
    if not words:
        return set()
    if not request and (words[0] in _WH_WORDS or (words[0] in _AUX_WORDS and not _negated(words))):
        return {"question"}  # interrogative: an edit verb inside it ("how do I fix X") is not a request
    kinds: set[str] = set()
    for clause in _clauses(sentence):
        kind = _clause_kind(clause)
        if kind:
            kinds.add(kind)
    return kinds


def _clauses(sentence: str) -> list[list[str]]:
    """Split on commas / "then", and on "and <edit verb> <determiner>" ("explain it and fix the bug")."""
    clauses: list[list[str]] = []
    for part in _CLAUSE.split(sentence):
        words = _words(part)
        start = 0
        for i in range(1, len(words) - 1):
            if words[i] == "and" and words[i + 1] in CHANGE_VERBS and i + 2 < len(words) and words[i + 2] in _DETERMINERS:
                clauses.append(words[start:i])
                start = i + 1
        clauses.append(words[start:])
    return clauses


def _clause_kind(words: list[str]) -> str | None:
    words, _ = _strip_prefixes(words)
    if not words:
        return None
    if _negated(words):
        rest = words[2:] if words[0] == "do" else words[1:]
        rest, _ = _strip_prefixes(rest)
        return "no_edit" if rest and rest[0] in _NO_EDIT_OBJECTS else None
    head = words[0]
    if head in CHANGE_VERBS:
        return "change"
    if head in QUESTION_VERBS or head in _WH_WORDS:
        return "question"
    return None
