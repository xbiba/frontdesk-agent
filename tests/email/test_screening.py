"""Screening in code before any model, the sender's text without quotes, and the reply's headers."""

from pathlib import Path

import pytest

from frontdesk.email.mailbox import FileMailbox, reply_draft
from frontdesk.email.screening import Screen, screen
from frontdesk.email.text import MAX_CHARS, shorten, split_quotes
from tests.email.helpers import MAILBOX, message, thread


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, None),
        ({"sender": "Clinic <INFO@example.com>"}, Screen.STAFF_ANSWERED),
        ({"labels": frozenset({"SENT"})}, Screen.STAFF_ANSWERED),
        ({"labels": frozenset({"SPAM"})}, Screen.SPAM),
        ({"headers": {"list-unsubscribe": "<mailto:x@example.com>"}}, Screen.LIST),
        ({"headers": {"precedence": "Bulk"}}, Screen.LIST),
        ({"headers": {"auto-submitted": "auto-replied"}}, Screen.AUTOMATIC),
        ({"headers": {"auto-submitted": "no"}}, None),
        ({"sender": "noreply@portal.example.com"}, Screen.NO_REPLY),
        ({"sender": "Portal <no-reply@portal.example.com>"}, Screen.NO_REPLY),
        ({"sender": "MAILER-DAEMON@example.com"}, Screen.NO_REPLY),
        ({"sender": "Noreen <noreen@example.com>"}, None),
    ],
)
def test_screening(fields: dict[str, object], expected: Screen | None) -> None:
    assert screen(thread(message(**fields)), mailbox=MAILBOX) is expected


def test_only_the_last_message_counts() -> None:
    answered = thread(message(), message(id="m2", sender=MAILBOX))
    asked_again = thread(message(sender=MAILBOX), message(id="m2"))
    assert screen(answered, mailbox=MAILBOX) is Screen.STAFF_ANSWERED
    assert screen(asked_again, mailbox=MAILBOX) is None


def test_the_quoted_messages_are_context() -> None:
    body = (
        "Perfetto, allora giovedì alle 10.\r\n\r\n"
        "> Il 06/11/2026 Clinica ha scritto:\n> Costa 100 EUR a seduta."
    )
    text = split_quotes(body)
    assert text.new == "Perfetto, allora giovedì alle 10."
    assert "Costa 100 EUR" in text.quoted
    assert not text.truncated


@pytest.mark.parametrize(
    "header",
    [
        "On Fri, 6 Nov 2026 at 10:00, Clinic <info@example.com> wrote:",
        "Il giorno ven 6 nov 2026 alle 10:00 Clinica <info@example.com> ha scritto:",
        "Në pre, 6 nën 2026, Klinika <info@example.com> shkroi:",
        "-----Original Message-----",
    ],
)
def test_the_quote_starts_at_the_clients_line(header: str) -> None:
    text = split_quotes(f"Grazie!\n\n{header}\nOld text here")
    assert text.new == "Grazie!"
    assert "Old text here" in text.quoted


def test_a_long_email_is_cut_and_flagged() -> None:
    text = split_quotes("a" * (MAX_CHARS + 10))
    assert len(text.new) == MAX_CHARS
    assert text.truncated


def test_shorten_cuts_at_a_word() -> None:
    assert shorten("uno due tre", 100) == "uno due tre"
    assert shorten("uno due tre quattro", 9) == "uno due …"


def test_the_draft_answers_the_last_message_in_its_conversation() -> None:
    last = message(
        id="m2",
        subject="Re: Fisioterapia",
        message_id="<m2@example.com>",
        references="<m0@example.com> <m1@example.com>",
        reply_to="anna@example.com",
    )
    draft = reply_draft(thread(message(), last), mailbox=MAILBOX, body="Gentile Anna")
    assert draft.thread_id == "t1"
    assert draft.subject == "Re: Fisioterapia"
    assert draft.to == "anna@example.com"
    assert draft.in_reply_to == "<m2@example.com>"
    assert draft.references == "<m0@example.com> <m1@example.com> <m2@example.com>"
    first = reply_draft(thread(), mailbox=MAILBOX, body="x")
    assert (first.subject, first.to) == ("Re: Fisioterapia", "Anna Leka <anna.leka@example.com>")


def test_the_file_mailbox_keeps_drafts_and_labels(tmp_path: Path) -> None:
    source = tmp_path / "box.yaml"
    source.write_text(
        "threads:\n"
        "  - id: t1\n"
        "    messages:\n"
        "      - {from: 'a@example.com', subject: Hi, body: Ciao, headers: {Precedence: bulk},"
        " labels: [INBOX], attachments: 1}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.yaml"
    box = FileMailbox.from_file(source, out=out)
    assert box.thread_ids("in:inbox", 30) == ["t1"]
    got = box.thread("t1").last
    assert (got.headers, got.labels, got.attachments) == ({"precedence": "bulk"}, {"INBOX"}, 1)
    box.create_draft(reply_draft(box.thread("t1"), mailbox=MAILBOX, body="Ciao"))
    box.add_labels("t1", ["Bozza AI", "Bozza AI"])
    assert box.labels == {"t1": ["Bozza AI"]}
    assert "Bozza AI" in out.read_text(encoding="utf-8")
