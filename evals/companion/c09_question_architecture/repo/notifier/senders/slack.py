"""Slack delivery through an incoming webhook."""

from __future__ import annotations

import json
import urllib.request

from notifier.config import SETTINGS
from notifier.retry import retry_with_backoff

EMOJI = {"warning": ":warning:", "critical": ":rotating_light:"}


def format_slack_message(alert) -> dict:
    icon = EMOJI.get(alert.severity, ":information_source:")
    return {"text": f"{icon} *{alert.title}* from {alert.source} (team {alert.team})"}


def http_post(url: str, body: dict) -> int:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status


class SlackSender:
    def send(self, job) -> None:
        body = format_slack_message(job.alert)
        retry_with_backoff(lambda: http_post(SETTINGS["slack_webhook_url"], body))
