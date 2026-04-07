from __future__ import annotations
import os
import re
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

JOURNAL_ROOT = Path.home() / ".aria" / "journal"
INGEST_LOG = Path.home() / ".aria" / "ingest_log.txt"

# ── Topic detection ───────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "crypto-trading": [
        "bybit",
        "binance",
        "okx",
        "pnl",
        "bot",
        "scalper",
        "hybrid",
        "trading",
        "crypto",
        "usdt",
        "signal",
    ],
    "aria-system": [
        "modelfile",
        "fix",
        "directive",
        "rag",
        "embedding",
        "chromadb",
        "aria",
        "memory",
        "milestone",
    ],
    "research": [
        "research",
        "find",
        "compare",
        "recipe",
        "how to build",
        "best",
        "opentrader",
        "digitalfortune",
        "stack",
    ],
    "coding": [
        "python",
        "rust",
        "go",
        "typescript",
        "docker",
        "error",
        "bug",
        "fix",
        "implement",
        "write",
    ],
    "system": [
        "install",
        "setup",
        "config",
        "pop-os",
        "nvidia",
        "vram",
        "ollama",
        "venv",
        "service",
    ],
}


def detect_topic(messages: list) -> str:
    """Detect primary topic from first few messages."""
    text = " ".join(m.get("content", "")[:200] for m in messages[:6]).lower()
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Quality scoring ───────────────────────────────────────────────────────────
def score_session(messages: list, tasks_completed: int = 0) -> int:
    """Score session quality. Returns 0-5."""
    if not messages:
        return 0
    score = 0
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assist_msgs = [m for m in messages if m.get("role") == "assistant"]

    # Substantive exchanges
    if len(user_msgs) >= 5:
        score += 2
    elif len(user_msgs) >= 3:
        score += 1

    # Task completion
    score += min(tasks_completed, 2)

    # Poke loop detection (bad signal)
    pokes = sum(
        1
        for m in user_msgs
        if m.get("content", "").strip().rstrip("!?.")
        in {"so", "?", "so?", "well?", "and?", "then?"}
    )
    if pokes >= 3:
        score -= 2
    elif pokes >= 2:
        score -= 1

    # Response length (very short = probably stuck)
    avg_len = sum(len(m.get("content", "")) for m in assist_msgs) / max(
        len(assist_msgs), 1
    )
    if avg_len < 50:
        score -= 1

    return max(0, score)


# ── Frontmatter builder ───────────────────────────────────────────────────────
def build_frontmatter(
    session_id: str,
    topic: str,
    project: Optional[str],
    summary: str,
    decisions: list,
    research_items: list,
    code_files: list,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d")
    decisions_str = "\n".join(f"  - {d}" for d in decisions) or "  - none"
    research_str = "\n".join(f"  - {r}" for r in research_items) or "  - none"
    code_str = "\n".join(f"  - {c}" for c in code_files) or "  - none"
    return (
        f"---\n"
        f"date: {now}\n"
        f"session_id: {session_id}\n"
        f"topic: {topic}\n"
        f"project: {project or 'none'}\n"
        f"summary: {summary}\n"
        f"decisions:\n{decisions_str}\n"
        f"research:\n{research_str}\n"
        f"code_written:\n{code_str}\n"
        f"---\n\n"
    )


# ── Journal entry writer ──────────────────────────────────────────────────────
def write_journal_entry(
    session_id: str,
    topic: str,
    project: Optional[str],
    summary: str,
    decisions: list,
    research_items: list,
    code_files: list,
    narrative: str,
    findings: list,
    open_questions: list,
    quality: int,
) -> Optional[Path]:
    """
    Write a journal entry. Returns path or None if quality too low.
    quality 0 = decisions only
    quality 1 = decisions + code
    quality 2+ = full entry
    """
    if quality <= 0 and not decisions:
        return None  # Nothing worth saving

    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:30]
    filename = f"{date_str}-{slug}-{session_id[:6]}.md"

    # Route to subdirectory based on topic
    subdir = {
        "research": "research",
        "crypto-trading": "sessions",
        "aria-system": "sessions",
        "coding": "sessions",
        "system": "sessions",
    }.get(topic, "sessions")

    entry_path = JOURNAL_ROOT / subdir / filename

    # Build content based on quality level
    fm = build_frontmatter(
        session_id, topic, project, summary, decisions, research_items, code_files
    )

    sections = [fm]

    if quality >= 2 and narrative:
        sections.append(f"## Session Summary\n{narrative}\n")

    if findings:
        sections.append(
            "## Key Findings\n" + "\n".join(f"- {f}" for f in findings) + "\n"
        )

    if decisions:
        sections.append(
            "## Decisions Made\n" + "\n".join(f"- {d}" for d in decisions) + "\n"
        )

    if code_files:
        sections.append(
            "## Code & Configs\n" + "\n".join(f"- `{c}`" for c in code_files) + "\n"
        )

    if open_questions:
        sections.append(
            "## Open Questions\n" + "\n".join(f"- {q}" for q in open_questions) + "\n"
        )

    content = "\n".join(sections)
    entry_path.write_text(content, encoding="utf-8")
    return entry_path


# ── Index updater ────────────────────────────────────────────────────────────
def update_index(
    session_id: str,
    topic: str,
    project: Optional[str],
    summary: str,
    entry_path: Path,
) -> None:
    """Append one row to the master index table."""
    index_path = JOURNAL_ROOT / "index.md"
    date_str = datetime.now().strftime("%Y-%m-%d")
    rel_path = entry_path.relative_to(JOURNAL_ROOT)
    row = (
        f"| {date_str} | {session_id[:8]} | {topic} "
        f"| {project or 'none'} | [{summary[:60]}]({rel_path}) |\n"
    )
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(row)


# ── Project document updater ──────────────────────────────────────────────────
def update_project_doc(
    project: str,
    status: str,
    what_works: list,
    known_issues: list,
    next_steps: list,
    key_files: dict,
) -> Path:
    """Update or create a living project document."""
    proj_path = JOURNAL_ROOT / "projects" / f"{project.upper()}.md"
    date_str = datetime.now().strftime("%Y-%m-%d")

    if proj_path.exists():
        # Append to history section
        existing = proj_path.read_text(encoding="utf-8")
        history_line = f"- {date_str}: {status}\n"
        if "## History" in existing:
            existing = existing.replace("## History\n", f"## History\n{history_line}")
        else:
            existing += f"\n## History\n{history_line}"
        # Update status line
        existing = re.sub(r"## Status:.*", f"## Status: {status}", existing)
        proj_path.write_text(existing, encoding="utf-8")
    else:
        works_str = "\n".join(f"- {w}" for w in what_works) or "- (none yet)"
        issues_str = "\n".join(f"- {i}" for i in known_issues) or "- (none)"
        steps_str = "\n".join(f"- {s}" for s in next_steps) or "- (none)"
        files_str = (
            "\n".join(f"- `{k}` → {v}" for k, v in key_files.items()) or "- (none)"
        )
        content = (
            f"# {project.upper()}\n\n"
            f"## Status: {status}\n\n"
            f"## Key Files\n{files_str}\n\n"
            f"## What Works\n{works_str}\n\n"
            f"## Known Issues\n{issues_str}\n\n"
            f"## Next Steps\n{steps_str}\n\n"
            f"## History\n- {date_str}: {status}\n"
        )
        proj_path.write_text(content, encoding="utf-8")
    return proj_path


# ── Decision logger ───────────────────────────────────────────────────────────
def log_decision(
    decision: str,
    context: str = "",
    project: Optional[str] = None,
) -> Path:
    """Log an explicit H2 decision to the decisions directory."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r"[^a-z0-9]+", "-", decision.lower().replace("'", ""))[:40]
    path = JOURNAL_ROOT / "decisions" / f"{date_str}-{slug}.md"
    content = (
        f"# Decision: {decision}\n\n"
        f"**Date:** {date_str}\n"
        f"**Project:** {project or 'general'}\n\n"
        f"## Context\n{context or 'No additional context recorded.'}\n"
    )
    path.write_text(content, encoding="utf-8")

    # Also append one-liner to milestones
    milestones_path = Path.home() / ".aria" / "milestones.md"
    with open(milestones_path, "a", encoding="utf-8") as f:
        f.write(f"\n[DECISION] {date_str}: {decision}")
    return path


# ── Research capture ──────────────────────────────────────────────────────────
def capture_research(
    query: str,
    findings: str,
    sources: list,
    topic: str = "research",
) -> Path:
    """Immediately persist web search findings to research journal."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:30]
    path = JOURNAL_ROOT / "research" / f"{slug}.md"

    entry = (
        f"\n## {date_str} — {query}\n\n"
        f"### Findings\n{findings[:2000]}\n\n"
        f"### Sources\n" + "\n".join(f"- {s}" for s in sources[:10]) + "\n"
    )

    # Append to topic file (accumulates over time)
    with open(path, "a", encoding="utf-8") as f:
        if path.stat().st_size == 0 if path.exists() else True:
            f.write(f"# Research: {topic}\n")
        f.write(entry)
    return path


# ── Code snippet saver ────────────────────────────────────────────────────────
def save_code_snippet(
    filename: str,
    content: str,
    source: str,
    purpose: str,
    verified: bool = False,
) -> Path:
    """Save a working code pattern to the code library."""
    path = JOURNAL_ROOT / "code" / filename
    header = (
        f"# Source: {source}\n"
        f"# Purpose: {purpose}\n"
        f"# Verified: {'YES - ' + datetime.now().strftime('%Y-%m-%d') if verified else 'UNVERIFIED'}\n"
        f"# Hardware: RTX 3060 Ti 8GB, i7-11700k, Pop!OS\n\n"
    )
    path.write_text(header + content, encoding="utf-8")
    return path


# ── Auto-ingest trigger ───────────────────────────────────────────────────────
def trigger_ingest_if_needed(project_root: str) -> bool:
    """
    Ingest journal into RAG if it changed in last 24 hours.
    Returns True if ingest was run.
    """
    import time

    journal_str = str(JOURNAL_ROOT)

    # Check if any journal file was modified in last 24h
    cutoff = time.time() - 86400
    changed = False
    for p in JOURNAL_ROOT.rglob("*.md"):
        if p.stat().st_mtime > cutoff:
            changed = True
            break

    if not changed:
        return False

    # Check if already ingested today
    today = datetime.now().strftime("%Y-%m-%d")
    if INGEST_LOG.exists():
        last = INGEST_LOG.read_text().strip().split("\n")[-1]
        if today in last and "journal" in last:
            return False

    # Run ingest
    try:
        result = subprocess.run(
            ["python3", "main.py", "ingest", "--folder", journal_str],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        with open(INGEST_LOG, "a") as f:
            f.write(f"{today} journal ingested\n")
        print(f"[Journal ingested into RAG]")
        return True
    except Exception as e:
        print(f"[Journal ingest failed: {e}]")
        return False


# ── Journal query ─────────────────────────────────────────────────────────────
def search_journal(query: str) -> str:
    """
    Fast grep-based search through journal files.
    Returns formatted results with file paths and snippets.
    """
    try:
        result = subprocess.run(
            ["grep", "-r", "-i", "-l", query, str(JOURNAL_ROOT)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
        if not files:
            return f"No journal entries found for: '{query}'"

        lines = [f"Found in {len(files)} journal file(s):"]
        for f in files[:5]:
            path = Path(f)
            # Get context lines
            ctx_result = subprocess.run(
                ["grep", "-i", "-A", "2", "-B", "1", query, f],
                capture_output=True,
                text=True,
                timeout=5,
            )
            snippet = ctx_result.stdout[:300].strip()
            lines.append(f"\n### {path.name}")
            lines.append(f"Path: {f}")
            lines.append(f"```\n{snippet}\n```")

        return "\n".join(lines)
    except Exception as e:
        return f"Journal search error: {e}"
