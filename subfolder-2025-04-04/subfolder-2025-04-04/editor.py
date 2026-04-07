"""Nedster Editor - Multi-file code editor with diff support"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import unified_diff, SequenceMatcher


class FileEditor:
    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.pending_edits: Dict[
            str, Tuple[str, str]
        ] = {}  # path -> (original, new_content)
        self.undo_stack: List[Tuple[str, str]] = []  # (path, original_content)

    def parse_edit_blocks(self, llm_response: str) -> List[dict]:
        """
        Parse LLM output for edit blocks in ALL these formats:

        Format A (preferred - NEDSTER native):
        <edit file="path/to/file.py">
        <old>exact old code</old>
        <new>replacement code</new>
        </edit>

        Format B (create/overwrite):
        <create file="path/to/new.py">
        full file content
        </create>

        Format C (bash output - treat as run_bash):
        ```bash
        command here
        ```

        Format D (markdown code with filename comment):
        ```python
        # path/to/file.py
        full content
        ```

        Return list of dicts:
        {"type": "edit|create|bash", "path": str,
         "old": str, "new": str, "content": str, "cmd": str}
        """
        results = []

        # Format A: <edit file="..."><old>...</old><new>...</new></edit>
        edit_pattern = re.compile(
            r'<edit\s+file=["\']([^"\']+)["\']>\s*'
            r"<old>(.*?)</old>\s*"
            r"<new>(.*?)</new>\s*"
            r"</edit>",
            re.DOTALL | re.IGNORECASE,
        )
        for m in edit_pattern.finditer(llm_response):
            results.append(
                {
                    "type": "edit",
                    "path": m.group(1).strip(),
                    "old": m.group(2).strip(),
                    "new": m.group(3).strip(),
                }
            )

        # Format B: <create file="...">...</create>
        create_pattern = re.compile(
            r'<create\s+file=["\']([^"\']+)["\']>\s*(.*?)\s*</create>',
            re.DOTALL | re.IGNORECASE,
        )
        for m in create_pattern.finditer(llm_response):
            results.append(
                {
                    "type": "create",
                    "path": m.group(1).strip(),
                    "content": m.group(2).strip(),
                }
            )

        # Format C: ```bash ... ```
        bash_pattern = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)
        for m in bash_pattern.finditer(llm_response):
            cmd = m.group(1).strip()
            if cmd:
                results.append(
                    {
                        "type": "bash",
                        "cmd": cmd,
                    }
                )

        # Format D: ```language\n# path/to/file.py\ncontent\n```
        code_file_pattern = re.compile(
            r"```(?:python|javascript|typescript|java|go|rust|c|cpp|ruby|php|swift|kt|scala|shell|json|yaml|yml|toml|xml|html|css|sql|md|txt)\n"
            r"#\s*([\w\.\-_/]+\.(?:py|js|ts|go|rs|java|c|cpp|h|hpp|rb|php|sh|md|txt|json|yaml|yml|toml|xml|html|css|sql|env))\n"
            r"(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )
        for m in code_file_pattern.finditer(llm_response):
            results.append(
                {
                    "type": "create",
                    "path": m.group(1).strip(),
                    "content": m.group(2).strip(),
                }
            )

        return results

    def apply_edit(self, edit: dict, auto: bool = False) -> str:
        """
        Apply an edit block.

        For type=edit:
          - Read file, find exact `old` block, replace with `new`
          - If old not found: fuzzy match (difflib SequenceMatcher >0.8)
          - Store original in undo_stack
          - If auto=False: show colored diff, ask Y/n
          - Write file
          - Return: "Edited path/to/file.py (+5/-3 lines)"

        For type=create:
          - If file exists and auto=False: show diff, confirm
          - Write file
          - Return: "Created path/to/file.py (42 lines)"
        """
        edit_type = edit.get("type")
        path = edit.get("path")

        if not path or not edit_type:
            return "Error: Invalid edit block (missing path or type)"

        if edit_type == "edit":
            return self._apply_replace_edit(edit, path, auto)
        elif edit_type == "create":
            return self._apply_create_edit(edit, path, auto)
        else:
            return f"Error: Unknown edit type: {edit_type}"

    def _apply_replace_edit(self, edit: dict, path: str, auto: bool) -> str:
        """Apply a replace-style edit."""
        filepath = Path(path) if Path(path).is_absolute() else None
        if not filepath:
            # Try to find in common locations
            for base in [self.project_dir, Path.cwd(), Path.home()]:
                candidate = base / path
                if candidate.exists():
                    filepath = candidate
                    break

        if not filepath or not filepath.exists():
            return f"Error: File not found: {path}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        old_block = edit.get("old", "")
        new_block = edit.get("new", "")

        if not old_block:
            return "Error: Empty <old> block in edit"

        # Try exact match first
        from tui import NedsterTUI

        tui = NedsterTUI()
        tui.print_status(f"[Edit detected: {path} | Searching for match...]", "dim")

        if old_block in original:
            new_content = original.replace(old_block, new_block, 1)
        else:
            # Fuzzy match with SequenceMatcher
            new_content, ratio = self._fuzzy_replace(original, old_block, new_block)
            if new_content is None:
                tui.print_error(
                    f"Could not find matching code block in {path}:\n{old_block[:100]}..."
                )
                return f"Error: Could not find matching code block in {path}"
            else:
                tui.print_status(
                    f"[Fuzzy match: {int(ratio * 100)}% — showing diff for review]",
                    "warning",
                )

        # Store for undo
        self.undo_stack.append((str(filepath), original))

        # Show diff if not auto mode
        if not auto:
            from tui import NedsterTUI

            tui = NedsterTUI()
            tui.print_diff(path, original, new_content)

            try:
                response = input("Apply? [Y/n] ").strip().lower()
                if response not in ("", "y", "yes"):
                    return "Edit cancelled by user"
            except (EOFError, KeyboardInterrupt):
                return "Edit cancelled"

        # Write the file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            # Restore on write error
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(original)
            return f"Error writing file: {e}"

        # Calculate diff stats
        old_lines = old_block.count("\n") + 1
        new_lines = new_block.count("\n") + 1
        added = max(0, new_lines - old_lines)
        removed = max(0, old_lines - new_lines)

        return f"Edited {path} (+{added}/-{removed} lines)"

    def _apply_create_edit(self, edit: dict, path: str, auto: bool) -> str:
        """Apply a create/overwrite-style edit."""
        content = edit.get("content", "")
        filepath = Path(path) if Path(path).is_absolute() else self.project_dir / path

        # Create parent dirs if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        original = ""
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    original = f.read()
            except Exception:
                original = ""

        # If file exists and not auto, show diff
        if original and not auto:
            from tui import NedsterTUI

            tui = NedsterTUI()
            tui.print_diff(path, original, content)

            try:
                response = input("Overwrite? [Y/n] ").strip().lower()
                if response not in ("", "y", "yes"):
                    return "Edit cancelled by user"
            except (EOFError, KeyboardInterrupt):
                return "Edit cancelled"

        # Store for undo if file existed
        if original:
            self.undo_stack.append((str(filepath), original))

        # Write the file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"Error writing file: {e}"

        lines = content.count("\n") + 1
        if original:
            return f"Overwritten {path} ({lines} lines)"
        else:
            return f"Created {path} ({lines} lines)"

    def _fuzzy_replace(
        self, content: str, old_block: str, new_block: str
    ) -> Tuple[Optional[str], float]:
        """
        Fuzzy match old_block in content using SequenceMatcher.
        Returns new content with replacement, or None if no match found.
        """
        # Normalize whitespace for comparison but preserve original
        old_normalized = " ".join(old_block.split())
        content_normalized = " ".join(content.split())

        # Try to find a fuzzy match
        matcher = SequenceMatcher(None, content_normalized, old_normalized)
        match = matcher.find_longest_match(
            0, len(content_normalized), 0, len(old_normalized)
        )

        if match.size < len(old_normalized) * 0.75:
            return None, 0.0  # Not a good enough match

        # Find the actual position in original content
        # Map normalized positions back to original
        old_start_norm = match.a
        old_end_norm = match.a + match.size

        # Convert normalized positions to character positions in original
        # This is approximate - count words
        content_words = content.split()
        old_words = old_block.split()

        word_start = sum(len(w) + 1 for w in content_words[: match.a])
        word_end = sum(len(w) + 1 for w in content_words[: match.a + match.size])

        # Find best matching span in original content
        best_match = None
        best_ratio = 0.8

        for i in range(len(content_words) - len(old_words) + 1):
            candidate = " ".join(content_words[i : i + len(old_words)])
            ratio = SequenceMatcher(None, candidate, old_normalized).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                # Calculate char positions
                char_start = sum(len(w) + 1 for w in content_words[:i])
                char_end = sum(len(w) + 1 for w in content_words[: i + len(old_words)])
                best_match = (char_start, char_end)

        if best_match:
            char_start, char_end = best_match
            return content[:char_start] + new_block + content[char_end:], best_ratio

        return None, 0.0

    def show_diff(self, path: str, original: str, new: str) -> None:
        """Print colored unified diff using rich or ANSI codes."""
        from tui import NedsterTUI

        tui = NedsterTUI()
        tui.print_diff(path, original, new)

    def undo_last(self) -> str:
        """Restore last edited file from undo_stack."""
        if not self.undo_stack:
            return "Nothing to undo"

        path, original = self.undo_stack.pop()

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(original)
            return f"Undone: {path}"
        except Exception as e:
            return f"Error undoing: {e}"

    def get_pending_diffs(self) -> Dict[str, Tuple[str, str]]:
        """Return all pending edits."""
        return self.pending_edits.copy()

    def clear_pending(self) -> None:
        """Clear pending edits."""
        self.pending_edits = {}
