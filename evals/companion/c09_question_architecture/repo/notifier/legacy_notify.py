"""Old direct-to-Slack path from before the queue existed. Nothing imports this any more."""

from __future__ import annotations

from notifier.senders.slack import format_slack_message, http_post


def legacy_send(alert, url: str) -> int:
    return http_post(url, format_slack_message(alert))
