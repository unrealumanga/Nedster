import re
from tools import write_file, list_dir, read_file, run_bash
import os

DIRECT_PATTERNS = [
    (
        re.compile(
            r"create\s+(?:a\s+)?(?:file\s+)?"
            r'["\']?(/[\w/\-_.]+|~/[\w/\-_.]+)["\']?'
            r'(?:\s+(?:with|containing|and write|write)\s+["\']?(.+?)["\']?)?$',
            re.IGNORECASE,
        ),
        "create_file",
    ),
    (
        re.compile(
            r'write\s+["\'](.+?)["\']'
            r'\s+(?:to|into)\s+["\']?(/[\w/\-_.]+)["\']?',
            re.IGNORECASE,
        ),
        "write_to_file",
    ),
    (
        re.compile(
            r'(?:show|list|what.?s in)\s+["\']?(/[\w/\-_.]+|~/[\w/\-_.]+)["\']?',
            re.IGNORECASE,
        ),
        "list_dir",
    ),
    (
        re.compile(
            r'(?:read|show|cat|open)\s+["\']?(/[\w/\-_.]+|~/[\w/\-_.]+)["\']?',
            re.IGNORECASE,
        ),
        "read_file",
    ),
]


def _try_direct_execute(user_input: str) -> str | None:
    """
    Bypass model for simple, unambiguous file operations.
    Returns result string if handled, None if model should handle.
    """
    for pattern, action in DIRECT_PATTERNS:
        m = pattern.search(user_input.strip())
        if not m:
            continue

        if action == "create_file":
            path = os.path.expanduser(m.group(1))
            content = m.group(2) or ""
            # Strip quotes
            content = content.strip("'\"")
            result = write_file(path, content)
            return f"Done. {result}"

        elif action == "write_to_file":
            content = m.group(1).strip("'\"")
            path = os.path.expanduser(m.group(2))
            result = write_file(path, content)
            return f"Done. {result}"

        elif action == "list_dir":
            path = os.path.expanduser(m.group(1))
            return list_dir(path)

        elif action == "read_file":
            path = os.path.expanduser(m.group(1))
            return read_file(path)

    return None


# In REPL loop, BEFORE agent.generate():
# direct_result = _try_direct_execute(user_input)
# if direct_result:
#     print(direct_result)
#     # Also verify with actual disk check
#     import re, os
#     path_m = re.search(r'(/home/\S+|~/\S+)', user_input)
#     if path_m:
#         path = os.path.expanduser(path_m.group(1))
#         if os.path.exists(path):
#             print(f"[Verified ✓] {path} exists on disk")
#     continue  # Skip agent.generate() entirely
