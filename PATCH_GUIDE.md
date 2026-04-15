# Nedster Bug Fixes — Patch Guide

## Summary of bugs fixed

### Bug 1 [CRITICAL] — Tools never execute, no output produced
**File:** `agent.py` → `NedsterAgent._execute_response()`
**Lines:** ~1508–1538

**Root cause:** The `futures` dict was declared but never populated.
- Tools in `SAFE_PARALLEL` created a `ThreadPoolExecutor` but called
  `executor.execute()` synchronously and discarded the result.
- Tools NOT in `SAFE_PARALLEL` hit the bare `else: print_warning(...)` branch
  and never ran at all.
- The `as_completed(futures)` loop then iterated over an empty dict → zero
  results accumulated into `tool_msg_accumulator`.

**Effect:** Every tool call was silently dropped. The agent printed tool call
headers (activity feed) but produced no tool results, so it gave empty or
hallucinated responses — exactly the "gibberish / not completing tasks" symptom.

**Fix:** Replace the entire `_execute_response` method with the version in
`agent_execute_response_fix.py`. All tools now execute via
`self.executor.execute()` and every result is appended to
`tool_msg_accumulator`.

---

### Bug 2 [HIGH] — parse_tool_calls: Format 4 (bash blocks) unreachable
**File:** `tools.py` → `parse_tool_calls()`
**Lines:** ~3454–3463

**Root cause:** Format 4 (bare ` ```bash``` ` block detection) and a second
`return results` were placed AFTER an earlier `return results` on line 3454.
Python never reaches code after a `return`.

**Effect:** The agent couldn't interpret responses from weak models that fall
back to markdown code blocks instead of `<tool>` XML.

**Fix:** Replace the entire `parse_tool_calls()` function with the version in
`tools_parse_tool_calls_fix.py`. All formats are now collected in order with
early-returns only between format groups (not inside them), and Format 4 is
properly positioned before Format 5/6/7.

---

### Bug 3 [MEDIUM] — _strip_weak_model_artifacts: orphaned dead code + NameError
**File:** `agent.py` → `NedsterAgent._strip_weak_model_artifacts()`
**Lines:** ~1163–1167

**Root cause:** Lines checking `accumulated[-_REPEAT_THRESHOLD:]` and
`accumulated.split()[-20:]` appear after `return text.strip()`. They reference
`accumulated` (undefined in this method) and `_REPEAT_THRESHOLD` (also
undefined). If these lines ever executed they would raise `NameError`.

**Effect:** Minor (dead code), but also fails to strip DIRECTIVE leakage from
chat-only models switching mid-session.

**Fix:** Replace `_strip_weak_model_artifacts()` with the version in
`agent_strip_weak_model_fix.py`. Dead code removed; added regex patterns to
strip leaked system-prompt DIRECTIVE blocks.

---

### Bug 4 [LOW] — cmd_stats() / print_stats() NameError on Windows
**File:** `nedster.py` → `print_stats()` and `cmd_stats()`
**Lines:** ~1912–2004

**Root cause:** The original `print_stats()` had the todo-reading block (which
references `project_dir`) grafted before the function's docstring, but the
function signature was `print_stats()` with no parameters. `cmd_stats()` then
called `print_stats()` with no arguments — causing a `NameError` for
`project_dir` inside the todo block.

Also: `nvidia-smi` format args were missing `--format=csv,noheader,nounits`,
so parsing would fail silently on some systems.

**Fix:** Merge both into a single clean `print_stats(project_dir=".")` with
`cmd_stats(project_dir=".")` calling it. Use the version in
`nedster_cmd_stats_fix.py`.

---

## How to apply

### agent.py

1. Find and replace the `_execute_response` method (starts at the line
   `def _execute_response(`) — replace the entire method body with
   `agent_execute_response_fix.py`.

2. Find and replace `_strip_weak_model_artifacts` — replace entire method
   with `agent_strip_weak_model_fix.py`.

### tools.py

3. Find `def parse_tool_calls(text: str) -> list:` and replace the entire
   function (ends just before `class ContinuityWatchdog`) with
   `tools_parse_tool_calls_fix.py`.

### nedster.py

4. Find both `def print_stats(` and `def cmd_stats():` — replace with the
   merged version in `nedster_cmd_stats_fix.py`.

---

## Additional improvement — chat-only model tool leak

When you `/switch lfm2.5-1.2b-instruct:latest`, the session log shows:
```
<tool call name="todowrite">
```
appearing in the output. This is because `agent.py` sets
`agent.tool_use_enabled = False` correctly, but the `generate()` method still
injects TOOL_CAPABILITY_ANCHOR and kickstart messages for file-op patterns,
giving the weak model system-prompt context it then regurgitates.

**Recommended fix** in `generate()`, wrap the anchor injections:
```python
# Only inject tool anchors if tool_use_enabled
if getattr(self, 'tool_use_enabled', True):
    messages.append({"role": "user", "content": TOOL_CAPABILITY_ANCHOR})
    messages.append({"role": "assistant", "content": "Ready. Executing."})
```
And skip the kickstart block too when `not tool_use_enabled`.
