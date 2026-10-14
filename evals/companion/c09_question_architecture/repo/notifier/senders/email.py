"""Email delivery (only used for the db team)."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage


class EmailSender:
    def send(self, job) -> None:
        message = EmailMessage()
        message["Subject"] = f"[{job.alert.severity}] {job.alert.title}"
        message["To"] = f"{job.alert.team}@example.invalid"
        message.set_content(f"{job.alert.title} from {job.alert.source}")
        with smtplib.SMTP("localhost") as smtp:
            smtp.send_message(message)
