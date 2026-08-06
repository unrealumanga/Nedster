from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9_./:-]+")
_CITATION = re.compile(r"\[S(\d+)]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_EXACT_VALUE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\d+(?:\.\d+)*(?:%|[A-Za-z]+)?|[A-Za-z_][\w.-]*/[\w./-]+)(?![A-Za-z0-9_])"
)
_OPERATIONAL = re.compile(
    r"\b(?:completed|created|read|retrieved|ran|searched|updated|written|successfully)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True, slots=True)
class GroundingSource:
    source_id: str
    label: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class GroundingCheck:
    valid: bool
    errors: tuple[str, ...]


class GroundingHarness:
    """Turn scoped tool evidence into a small, verifiable source pack."""

    def __init__(self, *, max_sources: int = 5, max_source_chars: int = 6_000):
        self.max_sources = max_sources
        self.max_source_chars = max_source_chars

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {
            token.lower()
            for token in _WORD.findall(value)
            if len(token) > 1 and token.lower() not in _STOP_WORDS
        }

    def _score(self, objective: str, text: str, prior: float) -> float:
        query_terms = self._terms(objective)
        if not query_terms:
            return prior
        overlap = len(query_terms & self._terms(text)) / len(query_terms)
        return prior + overlap

    def _candidates(self, evidence: list[dict[str, Any]], objective: str) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        for item in evidence:
            action = item.get("action") or {}
            result = item.get("result") or {}
            if not result.get("ok") or not isinstance(result.get("output"), dict):
                continue
            tool = str(action.get("tool") or "tool")
            output = result["output"]
            if tool == "read_file" and output.get("content"):
                label = str(output.get("path") or action.get("arguments", {}).get("path") or "file")
                candidates.append((label, str(output["content"]), 1.0))
            elif tool in {"search_text", "project_search"}:
                grouped: dict[str, list[str]] = {}
                for match in output.get("matches") or []:
                    path = str(match.get("path") or "search result")
                    grouped.setdefault(path, []).append(
                        f"{path}:{match.get('line', '?')}: {match.get('text', '')}"
                    )
                candidates.extend((path, "\n".join(lines), 0.8) for path, lines in grouped.items())
                if not grouped:
                    pattern = str(
                        action.get("arguments", {}).get("pattern")
                        or action.get("arguments", {}).get("query")
                        or ""
                    )
                    candidates.append(
                        (
                            "project text search",
                            f"pattern={pattern}\ncount={output.get('count', 0)}\n"
                            f"truncated={bool(output.get('truncated'))}",
                            0.6,
                        )
                    )
            elif tool == "list_files" and "files" in output:
                inventory = (
                    f"returned_count={output.get('count', len(output['files']))}\n"
                    f"total_matches={output.get('total_matches', len(output['files']))}\n"
                    f"truncated={bool(output.get('truncated'))}\n"
                    + "\n".join(map(str, output["files"]))
                )
                candidates.append(("project file inventory", inventory, 0.35))
            elif tool == "git_diff":
                rendered = (
                    f"tracked_count={output.get('tracked_count', 0)}\n"
                    f"tracked_files={json.dumps(output.get('tracked_files') or [])}\n"
                    f"untracked_count={output.get('untracked_count', 0)}\n"
                    f"untracked_files={json.dumps(output.get('untracked_files') or [])}\n"
                    f"status:\n{output.get('status', '')}\n"
                    f"stat:\n{output.get('stat', '')}\n"
                    f"truncated={bool(output.get('truncated'))}\n"
                    f"diff:\n{output.get('diff', '')}"
                )
                candidates.append(("working tree diff", rendered, 0.9))
            elif tool in {"run_command", "run_tests"}:
                streams = "\n".join(
                    part
                    for part in (str(output.get("stdout") or ""), str(output.get("stderr") or ""))
                    if part
                )
                rendered = (
                    f"returncode={output.get('returncode')}\n"
                    f"timed_out={bool(output.get('timed_out'))}\n{streams}"
                ).strip()
                if tool == "run_tests":
                    rendered = f"tests_run={output.get('tests_run')}\n{rendered}"
                elif output.get("assertions"):
                    rendered = (
                        f"assertions_passed={output.get('assertions_passed')}\n"
                        f"assertions={json.dumps(output.get('assertions'), sort_keys=True)}\n"
                        f"{rendered}"
                    )
                if rendered:
                    command_parts = (
                        action.get("arguments", {}).get("command")
                        or output.get("command")
                        or []
                    )
                    command = " ".join(map(str, command_parts))
                    verifier_prior = (
                        1.2
                        if output.get("returncode") == 0 and not output.get("timed_out")
                        else 0.75
                    )
                    candidates.append(
                        (f"command: {command}".strip(), rendered, verifier_prior)
                    )
            elif tool in {"write_file", "replace_text"}:
                label = str(output.get("path") or tool)
                arguments = action.get("arguments") or {}
                change = {"result": output}
                if tool == "write_file":
                    change["written_content"] = str(arguments.get("content") or "")
                else:
                    change["replaced_text"] = str(arguments.get("old") or "")
                    change["replacement_text"] = str(arguments.get("new") or "")
                candidates.append((f"change: {label}", json.dumps(change, sort_keys=True), 0.75))
            elif tool == "web_search":
                retrieved = str(output.get("retrieved_at") or "")
                for result_item in output.get("results") or []:
                    url = str(result_item.get("url") or "web result")
                    title = str(result_item.get("title") or url)
                    snippet = str(result_item.get("snippet") or "")
                    candidates.append(
                        (url, f"{title}\n{snippet}\nRetrieved: {retrieved}", 0.65)
                    )
            elif tool == "web_fetch" and output.get("content"):
                url = str(output.get("url") or "web page")
                retrieved = str(output.get("retrieved_at") or "")
                candidates.append(
                    (url, f"{output['content']}\nRetrieved: {retrieved}", 0.9)
                )
            elif tool == "market_price" and output.get("quotes"):
                observed = str(output.get("observed_at") or "")
                for quote in output["quotes"]:
                    url = str(quote.get("source_url") or quote.get("provider") or "market quote")
                    candidates.append(
                        (
                            url,
                            json.dumps({**quote, "observed_at": observed}, sort_keys=True),
                            1.0,
                        )
                    )

        deduplicated: dict[str, tuple[str, str, float]] = {}
        for label, text, prior in candidates:
            clean = text.strip()[: self.max_source_chars]
            if not clean:
                continue
            score = self._score(objective, f"{label}\n{clean}", prior)
            previous = deduplicated.get(label)
            if previous is None or (score, len(clean)) > (previous[2], len(previous[1])):
                deduplicated[label] = (label, clean, score)
        return sorted(deduplicated.values(), key=lambda value: value[2], reverse=True)

    def collect(self, evidence: list[dict[str, Any]], objective: str) -> list[GroundingSource]:
        selected = self._candidates(evidence, objective)[: self.max_sources]
        return [
            GroundingSource(f"S{index}", label, text, score)
            for index, (label, text, score) in enumerate(selected, 1)
        ]

    @staticmethod
    def render(sources: list[GroundingSource]) -> str:
        # Strongest evidence is nearest the question, at the end of the prompt.
        return "\n\n".join(
            f"[{source.source_id}] (source: {source.label})\n{source.text}" for source in reversed(sources)
        )

    @staticmethod
    def source_labels(sources: list[GroundingSource]) -> list[str]:
        return [f"[{source.source_id}] {source.label}" for source in sources]

    @staticmethod
    def verify(answer: str, sources: list[GroundingSource]) -> GroundingCheck:
        errors: list[str] = []
        stripped = answer.strip()
        if not stripped:
            return GroundingCheck(False, ("answer is empty",))
        if not sources:
            return GroundingCheck(True, ())

        source_map = {source.source_id: source.text for source in sources}
        cited = {f"S{value}" for value in _CITATION.findall(stripped)}
        invalid = sorted(cited - set(source_map))
        if invalid:
            errors.append("unknown citations: " + ", ".join(invalid))
        if not cited and "INSUFFICIENT_CONTEXT" not in stripped:
            errors.append("grounded answer has no source citations")

        sentence_text = re.sub(
            r"([.!?])\s*((?:\[S\d+])+)",
            r" \2\1",
            stripped,
        )
        for sentence in _SENTENCE.split(sentence_text):
            sentence = sentence.strip()
            if not sentence or sentence.startswith(("Sources:", "INSUFFICIENT_CONTEXT")):
                continue
            sentence_citations = {f"S{value}" for value in _CITATION.findall(sentence)}
            if len(sentence.split()) >= 5 and not sentence_citations:
                if _OPERATIONAL.search(sentence) and any(
                    source.label.casefold() in sentence.casefold() for source in sources
                ):
                    continue
                errors.append(f"uncited sentence: {sentence[:100]}")
                continue
            cited_text = "\n".join(
                source_map[source_id] for source_id in sentence_citations if source_id in source_map
            )
            for exact in _EXACT_VALUE.findall(_CITATION.sub("", sentence)):
                if exact not in cited_text:
                    errors.append(f"exact value is unsupported by its citation: {exact}")

        return GroundingCheck(not errors, tuple(dict.fromkeys(errors)))
