"""Prove KV prefix reuse on a real Ollama: cold vs warm vs mutated-early-message requests.

    python -m benchmarks.prefix_cache --model nedster-worker [--url http://127.0.0.1:11434] [--kv-info]
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}]
SYSTEM = "You are a careful coding agent. Use tools to inspect files before answering. " * 20


def conversation(first_user: str) -> list[list[dict]]:
    """The same 3-turn conversation, as the growing message list sent on each turn."""
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first_user}]
    call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}
    result = {"role": "tool", "tool_name": "read_file", "content": "def add(a, b):\n    return a + b\n" * 30}
    turn2 = [*base, call, result, {"role": "user", "content": "Now summarise it in one sentence."}]
    turn3 = [*turn2, {"role": "assistant", "content": "It defines add."}, {"role": "user", "content": "Any bugs?"}]
    return [base, turn2, turn3]


def ms(ns: int | None) -> float:
    return round((ns or 0) / 1e6, 1)


def send(client: httpx.Client, url: str, model: str, messages: list[dict]) -> dict:
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": False,
            "options": {"num_predict": 16, "temperature": 0}}
    response = client.post(f"{url}/api/chat", json=body)
    response.raise_for_status()
    return response.json()


def run_series(client: httpx.Client, url: str, model: str, label: str, first_user: str) -> list[tuple]:
    rows = []
    for turn, messages in enumerate(conversation(first_user), start=1):
        data = send(client, url, model, messages)
        rows.append((label, turn, data.get("prompt_eval_count", 0), ms(data.get("prompt_eval_duration")),
                     ms(data.get("load_duration")), ms(data.get("total_duration"))))
    return rows


def kv_info() -> dict[str, str]:
    """Best effort: read the env of a local ollama server process; otherwise 'unknown'."""
    found = {"OLLAMA_KV_CACHE_TYPE": "unknown", "OLLAMA_FLASH_ATTENTION": "unknown"}
    try:
        import psutil

        for proc in psutil.process_iter(["name"]):
            if "ollama" not in (proc.info["name"] or "").lower():
                continue
            try:
                env = proc.environ()
            except (psutil.Error, OSError):
                continue
            for key in found:
                if key in env:
                    found[key] = env[key]
            return found
    except ImportError:
        pass
    for key in found:  # fall back to this process environment (same shell that launched the server)
        found[key] = os.getenv(key, "unknown")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.prefix_cache")
    parser.add_argument("--model", default="nedster-worker")
    parser.add_argument("--url", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--kv-info", action="store_true")
    args = parser.parse_args(argv)
    url = args.url if "://" in args.url else f"http://{args.url}"
    if args.kv_info:
        for key, value in kv_info().items():
            print(f"{key}={value}")
    try:
        with httpx.Client(timeout=300) as client:
            client.get(f"{url}/api/version").raise_for_status()
            rows = run_series(client, url, args.model, "cold", "Look at a.py and tell me what it does.")
            rows += run_series(client, url, args.model, "warm", "Look at a.py and tell me what it does.")
            rows += run_series(client, url, args.model, "mutated", "Look at b.py and tell me what it does.")
    except httpx.HTTPStatusError as exc:
        print(f"ERROR: Ollama at {url} rejected the request ({exc.response.status_code}); is model {args.model!r} installed?")
        return 1
    except (httpx.HTTPError, OSError) as exc:
        print(f"SKIP: Ollama unreachable at {url} ({exc})")
        return 0
    print(f"{'series':8} {'turn':>4} {'prompt_eval_count':>18} {'prompt_eval_ms':>15} {'load_ms':>9} {'total_ms':>9}")
    for label, turn, count, prefill, load, total in rows:
        print(f"{label:8} {turn:>4} {count:>18} {prefill:>15} {load:>9} {total:>9}")
    print("Reuse is working when 'warm' prompt_eval_count/ms is far below 'cold' (only new tokens are evaluated); "
          "'mutated' should re-evaluate nearly everything because the early message changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
