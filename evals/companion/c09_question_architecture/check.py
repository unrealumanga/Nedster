from __future__ import annotations

import re
from pathlib import Path
from evals.companion import helpers

REQUIRED = ["receive_webhook", "parse_event", "route", "apply_rules", "enqueue", "run_once", "send", "retry_with_backoff"]


def check(root: Path, answer: str) -> tuple[bool, str]:
    if helpers.tree(root) != helpers.tree(helpers.original(__file__, '')):
        return False, "answer-only task changed repository files"
    low = answer
    missing = [name for name in REQUIRED if not re.search(rf"\b{name}\b", low)]
    if missing:
        return False, f"answer does not mention: {', '.join(missing)}"
    if not re.search(r"dedup|duplicate", low, re.I):
        return False, "answer never mentions the duplicate suppression step"
    if not re.search(r"slack", low, re.I):
        return False, "answer never mentions Slack"
    # Check local facts, not the order in which an author chooses to explain them.
    facts = {
        'signature verification': r'(verif\w*|check\w*|validat\w*).{0,80}(signature|hmac)|(signature|hmac).{0,80}(verif\w*|check\w*|validat\w*)',
        'duplicate suppression': r'(dedup\w*|duplicate|repeat).{0,100}(drop\w*|suppress\w*|skip\w*|filter\w*|window|prevent\w*)|(drop\w*|suppress\w*|skip\w*|filter\w*|prevent\w*).{0,100}(duplicate|repeat)',
        'routing channels': r'apply_rules.{0,100}(channel|team|severity)|(channel|team|severity).{0,100}apply_rules',
        'queued jobs': r'enqueue.{0,100}(job|channel|queue)|(job|channel|queue).{0,100}enqueue',
        'worker dispatch': r'run_once.{0,140}(pop|drain|dequeue|job|sender)|(pop|drain|dequeue|job|sender).{0,140}run_once',
        'Slack HTTP delivery': r'(slack|send\b).{0,180}(http_post|post\w*|webhook)|(http_post|post\w*).{0,180}slack',
        'retry cause': r'retry_with_backoff.{0,140}(OSError|I/O|network|error)|(OSError|I/O|network|error).{0,140}retry_with_backoff',
        'backoff behavior': r'doubl\w*|exponential|1\s*,\s*2\s*,\s*4',
    }
    missing = [name for name, pattern in facts.items() if not re.search(pattern, low, re.I | re.S)]
    if missing:
        return False, 'answer lacks supported flow facts: ' + ', '.join(missing)
    return True, "answer traces the full webhook -> Slack path"
