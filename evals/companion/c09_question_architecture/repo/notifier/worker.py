"""Background worker: drains the queue and hands each job to its channel sender."""

from __future__ import annotations

from notifier.queue import QUEUE
from notifier.senders.email import EmailSender
from notifier.senders.slack import SlackSender

SENDERS = {"slack": SlackSender(), "email": EmailSender()}


def run_once(queue=QUEUE, senders=SENDERS) -> bool:
    """Process one job. Returns False when the queue was empty."""
    job = queue.pop()
    if job is None:
        return False
    sender = senders.get(job.channel)
    if sender is None:
        return True  # no sender for this channel (e.g. pager): drop it
    sender.send(job)
    return True


def run_forever(poll=None) -> None:
    while True:
        if not run_once() and poll:
            poll()
