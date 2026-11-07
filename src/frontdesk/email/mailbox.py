"""The mailbox adapter: the little the drafts need from a mailbox, whatever serves it.

List the conversations a search finds, read one, create a draft reply in it, put labels on it.
There is no way to send: no adapter has one, and Gmail's is reached with a scope that cannot send
(frontdesk.email.gmail). `FileMailbox` serves tests and evaluations from a file, in memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

# The labels a provider puts on a message that matter here (Gmail's system label names).
SPAM = "SPAM"
SENT = "SENT"
DRAFT = "DRAFT"


@dataclass(frozen=True)
class MailMessage:
    """One message of a conversation. `headers` holds the few headers the screening reads, by
    lower-case name; `body` is the plain text, quotes included."""

    id: str
    sender: str  # the From header
    subject: str
    body: str
    message_id: str = ""  # the RFC 5322 Message-ID, for In-Reply-To
    references: str = ""
    reply_to: str = ""
    sent_at: datetime | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    labels: frozenset[str] = frozenset()
    attachments: int = 0


@dataclass(frozen=True)
class MailThread:
    id: str
    messages: tuple[MailMessage, ...]

    @property
    def last(self) -> MailMessage:
        return self.messages[-1]


@dataclass(frozen=True)
class Draft:
    """A reply that waits in the mailbox for a person to send it."""

    thread_id: str
    sender: str  # the mailbox's own address
    to: str
    subject: str
    body: str
    in_reply_to: str
    references: str


class Mailbox(Protocol):
    def thread_ids(self, query: str, limit: int) -> list[str]:
        """The conversations the search finds, newest first, at most `limit`."""
        ...

    def thread(self, thread_id: str) -> MailThread: ...

    def create_draft(self, draft: Draft) -> str:
        """Create the draft in its conversation; its ID."""
        ...

    def add_labels(self, thread_id: str, names: Sequence[str]) -> None:
        """Put the labels on the conversation, creating those the mailbox lacks."""
        ...


def reply_draft(thread: MailThread, *, mailbox: str, body: str) -> Draft:
    """The draft that answers the last message of `thread`, in the same conversation."""
    last = thread.last
    subject = last.subject.strip()
    if not subject.lower().startswith(("re:", "r:", "ri:")):
        subject = f"Re: {subject}" if subject else "Re:"
    references = " ".join(part for part in (last.references, last.message_id) if part)
    return Draft(
        thread_id=thread.id,
        sender=mailbox,
        to=last.reply_to or last.sender,
        subject=subject,
        body=body,
        in_reply_to=last.message_id,
        references=references,
    )


class FileMailbox:
    """Conversations from a YAML file, held in memory; drafts and labels are kept on the object
    (and written to `out` when given), never sent anywhere. For tests, evaluations and a try on
    the Mac."""

    def __init__(self, threads: Sequence[MailThread], *, out: Path | None = None) -> None:
        self._threads = {thread.id: thread for thread in threads}
        self._order = [thread.id for thread in threads]
        self._out = out
        self.drafts: list[Draft] = []
        self.labels: dict[str, list[str]] = {}

    @classmethod
    def from_file(cls, path: Path, *, out: Path | None = None) -> FileMailbox:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        threads = data.get("threads", []) if isinstance(data, dict) else []
        return cls([parse_thread(item) for item in threads], out=out)

    def thread_ids(self, query: str, limit: int) -> list[str]:
        return self._order[:limit]

    def thread(self, thread_id: str) -> MailThread:
        return self._threads[thread_id]

    def create_draft(self, draft: Draft) -> str:
        self.drafts.append(draft)
        self._save()
        return f"draft-{len(self.drafts)}"

    def add_labels(self, thread_id: str, names: Sequence[str]) -> None:
        found = self.labels.setdefault(thread_id, [])
        found.extend(name for name in names if name not in found)
        self._save()

    def _save(self) -> None:
        if self._out is None:
            return
        state = {
            "drafts": [draft.__dict__ for draft in self.drafts],
            "labels": self.labels,
        }
        self._out.write_text(yaml.safe_dump(state, allow_unicode=True), encoding="utf-8")


def parse_thread(item: Mapping[str, Any]) -> MailThread:
    """A conversation as the file writes it: `id` and `messages`, each message with `id`,
    `from`, `subject`, `body` and, if any, `headers`, `labels`, `message_id`, `date`."""
    messages = tuple(
        _parse_message(index, message) for index, message in enumerate(item["messages"])
    )
    return MailThread(id=str(item["id"]), messages=messages)


def _parse_message(index: int, item: Mapping[str, Any]) -> MailMessage:
    headers = {str(name).lower(): str(value) for name, value in (item.get("headers") or {}).items()}
    sent = item.get("date")
    return MailMessage(
        id=str(item.get("id", f"m{index + 1}")),
        sender=str(item["from"]),
        subject=str(item.get("subject", "")),
        body=str(item.get("body", "")),
        message_id=str(item.get("message_id", f"<m{index + 1}@example.com>")),
        references=str(item.get("references", "")),
        reply_to=str(item.get("reply_to", "")),
        sent_at=sent if isinstance(sent, datetime) else None,
        headers=headers,
        labels=frozenset(str(label) for label in item.get("labels", [])),
        attachments=int(item.get("attachments", 0)),
    )
