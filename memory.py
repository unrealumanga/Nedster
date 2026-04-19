
class SessionLog:
    """
    Durable append-only event log. Lives outside agent.py.
    Survives harness crashes. Enables wake/resume.
    Pattern: Anthropic Managed Agents session architecture.
    """
    def __init__(self, session_id: str):
        import os, json
        from pathlib import Path
        from datetime import datetime
        self.session_id = session_id
        log_dir = Path.home() / ".aria" / "session_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{session_id}.jsonl"
        self._events = []
        self._load()

    def _load(self):
        import json
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    try:
                        self._events.append(json.loads(line))
                    except Exception:
                        pass

    def emit(self, event_type: str, data: dict):
        """Write event to durable log immediately."""
        import json
        from datetime import datetime
        event = {
            "type": event_type,
            "ts": datetime.now().isoformat(),
            "data": data
        }
        self._events.append(event)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_events(self, last_n: int = None, event_type: str = None) -> list:
        events = self._events
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        if last_n:
            events = events[-last_n:]
        return events

    @classmethod
    def wake(cls, session_id: str) -> "SessionLog":
        """Resume a crashed session from its event log."""
        return cls(session_id)

# File: memory.py
import ollama
import uuid
import os

POISON_PHRASES = [
    "cannot create files",
    "don't have filesystem",
    "can't access",
    "without shell execution",
    "AI assistant without",
    "no filesystem access",
    "I'm an AI",
    "as an AI",
    "I cannot directly",
    "I don't have the ability",
]


def _strip_poison(summary: str) -> str:
    """Remove lines containing tool-limitation hallucinations."""
    lines = summary.split("\n")
    clean = []
    for line in lines:
        if not any(p.lower() in line.lower() for p in POISON_PHRASES):
            clean.append(line)
    return "\n".join(clean)


class MemoryManager:

    def _init_session_db(self):
        import sqlite3, os
        db_path = os.path.expanduser("~/.aria/sessions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS sessions
            USING fts5(session_id, date, topic, content)
        """)
        self._db.commit()
    
    def save_to_search_db(self, session_id: str, topic: str, content: str):
        from datetime import datetime
        self._db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?)",
            (session_id, datetime.now().isoformat(), topic, content[:5000])
        )
        self._db.commit()
    
    def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        try:
            cursor = self._db.execute(
                "SELECT session_id, date, topic, content "
                "FROM sessions WHERE sessions MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit)
            )
            return [{"session_id": r[0], "date": r[1],
                     "topic": r[2], "snippet": r[3][:200]}
                    for r in cursor.fetchall()]
        except Exception:
            return []
    def __init__(self, llm_model_name: str):
        self.model = llm_model_name
        self.short_term = []  # raw message dicts [{role, content}]
        self.session_summary = ""  # compressed summary of older turns
        self.session_id = uuid.uuid4().hex[:8]
        self.turn_count = 0
        self._in_tool_loop = False
        self._init_session_db()
        self._boot_milestones()

    def add_turn(self, user_msg: str, assistant_msg: str):
        """Add a new turn. Auto-compress when session gets long."""
        self.short_term.append({"role": "user", "content": user_msg})
        self.short_term.append({"role": "assistant", "content": assistant_msg})
        self.turn_count += 1
        # Compress when more than 20 messages accumulate
        if len(self.short_term) > 20:
            if getattr(self, "_in_tool_loop", False):
                return
            self._compress_session()

    def _compress_session(self):
        """
        Summarize the oldest messages, keeping the latest 14 (7 turns) as raw context.
        The summary is appended to any existing summary so context accumulates.
        """
        to_compress = self.short_term[:-14]
        keep = self.short_term[-14:]

        # Build a plain-text history string from to_compress
        history_text = ""
        for msg in to_compress:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content'][:400]}\n"

        # If there's an existing summary, include it
        prior = (
            f"Prior summary: {self.session_summary}\n\n" if self.session_summary else ""
        )

        compress_prompt = (
            f"{prior}"
            "Summarize the following conversation into 3-5 bullet points.\n"
            "Capture: key facts, decisions, what was accomplished.\n"
            "EXCLUDE any statements about 'I cannot create files' or\n"
            "'I don't have filesystem access' — these are wrong.\n"
            "EXCLUDE any statements about AI limitations.\n"
            "Include: file paths created, tools used, tasks completed.\n"
            "Be concise. No preamble.\n\n"
            f"{history_text}\nSummary:"
        )

        try:
            response = ollama.generate(
                model=self.model,
                prompt=compress_prompt,
                options={
                    "num_ctx": 1024,
                    "num_predict": 100,
                    "temperature": 0.0,
                    "think": False,
                },
            )
            new_summary = response["response"].strip()
            self.session_summary = _strip_poison(new_summary)
            self.short_term = keep
            print(
                f"  • [\x1b[38;5;245mSession compressed: {len(to_compress)} msgs → summary\x1b[0m]"
            )
        except Exception as e:
            # If compression fails, just drop oldest turns
            self.short_term = keep
            print(f"[Compression failed, truncated: {e}]")

    def get_context_messages(self) -> list:
        """
        Returns messages to inject into the LLM call:
        [summary_block (if exists)] + [raw recent turns]
        Total budget: ~600 tokens for memory (summary ~150 + 6 turns ~450)
        """
        messages = []
        if self.session_summary:
            messages.append(
                {
                    "role": "user",
                    "content": f"[Session History Summary]\n{self.session_summary}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Understood. I have context from our earlier conversation.",
                }
            )
        # Add last 6 turns (12 messages) raw
        messages.extend(self.short_term[-12:])
        return messages

    def get_last_n_turns_text(self, n: int = 4) -> str:
        """Returns last N turns as plain text for query rewriting."""
        msgs = self.short_term[-(n * 2) :]
        text = ""
        for msg in msgs:
            role = "User" if msg["role"] == "user" else "Assistant"
            text += f"{role}: {msg['content'][:300]}\n"
        return text

    def clear(self):
        self.short_term = []
        self.session_summary = ""
        self.turn_count = 0

    def _boot_milestones(self):
        """Silently load previous session context on startup."""
        import json
        path = os.path.expanduser("~/.aria/milestones.jsonl")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) < 3:
                return
            
            recent_entries = []
            for line in lines[-60:]:
                try:
                    entry = json.loads(line)
                    recent_entries.append(f"[{entry.get('timestamp')}] {entry.get('event')}")
                except json.JSONDecodeError:
                    continue
                    
            recent = "\n".join(recent_entries).strip()
            if recent:
                self.session_summary = (
                    "[Previous sessions context — use naturally, don't announce]\n"
                    + recent
                )
        except Exception:
            pass  # silently skip

    def get_pending_plan(self) -> str:
        """Return the last assistant message (likely the pending plan)."""
        for msg in reversed(self.short_term):
            if msg["role"] == "assistant":
                return msg["content"][:500]
        return ""
