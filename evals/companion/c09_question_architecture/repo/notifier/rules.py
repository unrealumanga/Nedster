"""Routing rules: which channels should hear about an alert."""

from __future__ import annotations

from notifier.config import SETTINGS


def apply_rules(alert) -> list[str]:
    """Channels for this alert. Info-level alerts are dropped; critical ones also page."""
    if alert.severity == "info":
        return []
    channels = list(SETTINGS["channels_by_team"].get(alert.team, ["slack"]))
    if alert.severity == "critical" and "pager" not in channels:
        channels.append("pager")
    return channels
