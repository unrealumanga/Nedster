def parse_tool_calls(text: str) -> list:
    """
    Extract tool calls handling ALL format variants Aria might generate:
    - Correct:  <tool name="bash">{"cmd": "ls"}</tool>
    - Broken 1: <tool name="bash"><parameter=cmd>ls</parameter>
    - Broken 2: <tool_call>{"name": "bash", "cmd": "ls"}</tool_call>
    - Broken 3: ```bash\nls\n```  (markdown code blocks)

    FIX: The original had dead code after early `return results` on Format 1.
    Formats 2, 3, 4 were only reachable if Format 1 found nothing, but
    Format 4 (bare ```bash``` fallback) was placed AFTER a `return results`
    on line 3454 and was NEVER reached. Consolidated into a single-pass approach
    that collects all formats before returning.
    """
    import re, json

    results = []

    # Format 1: Correct JSON format  <tool name="X">{json}</tool>
    pattern1 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>\s*(.*?)\s*</tool>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern1.finditer(text):
        name = m.group(1).strip()
        args_raw = m.group(2).strip()
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            try:
                args = json.loads(_repair_json(args_raw))
            except Exception:
                args = _parse_kv(args_raw)
        except Exception:
            args = _parse_kv(args_raw)
        results.append({"name": name, "args": args})

    # If we found Format 1 calls, return early (avoid double-counting)
    if results:
        return results

    # Format 2: Broken <parameter=X> format
    pattern2 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>.*?<parameter[= ]+(\w+)>\s*(.*?)\s*(?:</parameter>|$)',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern2.finditer(text):
        name = m.group(1).strip()
        key = m.group(2).strip()
        val = m.group(3).strip()
        results.append({"name": name, "args": {key: val}})

    if results:
        return results

    # Format 3: <tool_call>{json}</tool_call>
    pattern3 = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
    )
    for m in pattern3.finditer(text):
        try:
            blob = json.loads(m.group(1))
            name = blob.pop("name", blob.pop("tool", "run_bash"))
            results.append({"name": name, "args": blob})
        except Exception:
            pass

    if results:
        return results

    # Format 4: Bare ```bash``` or ```sh``` code blocks — treat as run_bash
    # FIX: This was placed AFTER a `return results` in the original and was
    # therefore dead code. Moved here so it actually executes.
    bash_blocks = re.findall(r'```(?:bash|sh|shell)\n(.*?)\n```', text, re.DOTALL)
    for cmd in bash_blocks:
        cmd = cmd.strip()
        if cmd:
            results.append({"name": "run_bash", "args": {"cmd": cmd}})

    if results:
        return results

    # Format 5: Malformed closing tag - missing slash  <tool name="X">...</tool> vs <tool>
    pattern5 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>(.*?)<tool>', re.DOTALL | re.IGNORECASE
    )
    for m in pattern5.finditer(text):
        name = m.group(1).strip()
        args_raw = m.group(2).strip()
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            try:
                args = json.loads(_repair_json(args_raw))
            except Exception:
                args = _parse_kv(args_raw)
        except Exception:
            args = _parse_kv(args_raw)
        results.append({"name": name, "args": args})

    # Format 6: <create file="path">content</create>
    pattern_create = re.compile(
        r'<(?:tool\s+)?create\s+file=["\']?([^"\'>\s]+)["\']?>(.*?)</(?:tool|create)>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern_create.finditer(text):
        path = m.group(1).strip()
        content = m.group(2).strip()
        results.append(
            {"name": "write_file", "args": {"path": path, "content": content}}
        )

    # Format 7: self-closing tag  <tool name="X" />
    pattern_self_close = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']\s*/>', re.DOTALL | re.IGNORECASE
    )
    for m in pattern_self_close.finditer(text):
        name = m.group(1).strip()
        results.append({"name": name, "args": {}})

    return results
