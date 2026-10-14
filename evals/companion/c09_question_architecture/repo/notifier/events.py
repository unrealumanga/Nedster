"""Alert model and webhook payload parsing."""

from __future__ import annotations

from dataclasses import dataclass, field

SEVERITIES = ("info", "warning", "critical")


@dataclass
class Alert:
    source: str
    title: str
    severity: str = "info"
    team: str = "ops"
    labels: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return f"{self.source}:{self.title}:{self.severity}"


def parse_event(payload: dict) -> Alert:
    """Turn a raw webhook body into an Alert, validating the severity."""
    severity = str(payload.get("severity", "info")).lower()
    if severity not in SEVERITIES:
        raise ValueError(f"bad severity {severity!r}")
    return Alert(
        source=str(payload["source"]),
        title=str(payload["title"]),
        severity=severity,
        team=str(payload.get("team", "ops")),
        labels=dict(payload.get("labels", {})),
    )
