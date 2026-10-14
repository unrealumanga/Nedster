"""HTTP entry point. The web framework calls receive_webhook() for POST /alerts."""

from __future__ import annotations

import hashlib
import hmac

from notifier.config import SETTINGS
from notifier.events import parse_event
from notifier.router import ROUTER


def verify_signature(body: bytes, signature: str) -> bool:
    expected = hmac.new(SETTINGS["webhook_secret"].encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def receive_webhook(payload: dict, body: bytes, signature: str) -> tuple[int, str]:
    if not verify_signature(body, signature):
        return 401, "bad signature"
    try:
        alert = parse_event(payload)
    except (KeyError, ValueError) as exc:
        return 400, str(exc)
    queued = ROUTER.route(alert)
    return 202, f"queued {queued} notification(s)"
