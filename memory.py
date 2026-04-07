# File: memory.py
import ollama
import uuid
import os


class MemoryManager:
    def __init__(self, llm_model_name: str):
        self.model = llm_model_name
        self.short_term = []  # raw message dicts [{role, content}]
        self.session_summary = ""  # compressed summary of older turns
        self.session_id = uuid.uuid4().hex[:8]
        self.turn_count = 0
        self._in_tool_loop = False
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

        prompt = (
            f"{prior}"
            f"Summarize the following conversation turns into 3-5 bullet points.\n"
            f"Capture: key facts discussed, decisions made, topics covered.\n"
            f"Be concise. No preamble. Just bullet points.\n\n"
            f"{history_text}\nSummary:"
        )

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_ctx": 1024,
                    "num_predict": 100,
                    "temperature": 0.0,
                    "think": False,
                },
            )
            new_summary = response["response"].strip()
            self.session_summary = new_summary
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
        path = os.path.expanduser("~/.aria/milestones.md")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                lines = f.readlines()
            if len(lines) < 3:
                return
            recent = "".join(lines[-60:]).strip()
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
