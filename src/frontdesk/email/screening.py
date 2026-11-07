"""Screening in code, before any model (design §3.2): what gets no draft, and why.

- the last message is the mailbox's own (the staff has answered) or a draft;
- the message is in spam;
- a list or automatic message: List-Unsubscribe, Auto-Submitted other than "no", Precedence
  bulk, list or junk;
- a sender that takes no answer: noreply, no-reply, donotreply, mailer-daemon, postmaster.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from frontdesk.email.mailbox import DRAFT, SENT, SPAM, MailThread
from frontdesk.tenancy.pack import address_of


class Screen(StrEnum):
    STAFF_ANSWERED = "staff_answered"
    SPAM = "spam"
    LIST = "list"
    AUTOMATIC = "automatic"
    NO_REPLY = "no_reply"


_NO_REPLY: Final = re.compile(
    r"^(?:no-?reply|do-?not-?reply|mailer-daemon|postmaster|bounces?)\b", re.IGNORECASE
)
_BULK: Final = frozenset({"bulk", "list", "junk"})


def screen(thread: MailThread, *, mailbox: str) -> Screen | None:
    """Why the conversation gets no draft, or None when its last message may get one."""
    last = thread.last
    if address_of(last.sender) == mailbox.lower() or last.labels & {SENT, DRAFT}:
        return Screen.STAFF_ANSWERED
    if SPAM in last.labels:
        return Screen.SPAM
    headers = last.headers
    if "list-unsubscribe" in headers or headers.get("precedence", "").strip().lower() in _BULK:
        return Screen.LIST
    if headers.get("auto-submitted", "no").strip().lower() != "no":
        return Screen.AUTOMATIC
    if _NO_REPLY.match(address_of(last.sender)):
        return Screen.NO_REPLY
    return None
