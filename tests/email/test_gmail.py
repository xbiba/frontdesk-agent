"""The Gmail adapter on a mock transport: the search, a thread, a draft in its conversation, the
labels; never a request that sends."""

import base64
import json
from email import message_from_bytes

import httpx2
import pytest

from frontdesk.email.gmail import GmailMailbox, MailboxError, mime
from frontdesk.email.mailbox import reply_draft
from tests.email.helpers import MAILBOX


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


THREAD = {
    "id": "t1",
    "messages": [
        {
            "id": "m1",
            "threadId": "t1",
            "labelIds": ["INBOX", "UNREAD"],
            "internalDate": "1794038400000",
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "From", "value": "Anna <anna@example.com>"},
                    {"name": "Subject", "value": "Fisioterapia"},
                    {"name": "Message-ID", "value": "<m1@example.com>"},
                    {"name": "List-Unsubscribe", "value": "<mailto:u@example.com>"},
                    {"name": "X-Other", "value": "dropped"},
                ],
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64("Quanto costa?")}},
                    {"mimeType": "text/html", "body": {"data": _b64("<p>Quanto costa?</p>")}},
                    {"mimeType": "application/pdf", "filename": "referto.pdf", "body": {}},
                ],
            },
        },
        {
            "id": "m2",
            "payload": {
                "mimeType": "text/html",
                "headers": [{"name": "From", "value": "b@example.com"}],
                "body": {"data": _b64("<div>Ciao<br>a tutti &amp; grazie</div><style>x{}</style>")},
            },
        },
    ],
}


class Gmail:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/gmail/v1/users/me")
        if path == "/threads":
            return httpx2.Response(200, json={"threads": [{"id": "t1"}, {"id": "t2"}]})
        if path == "/threads/t1":
            return httpx2.Response(200, json=THREAD)
        if path == "/drafts":
            return httpx2.Response(200, json={"id": "d1"})
        if path == "/labels" and request.method == "GET":
            return httpx2.Response(200, json={"labels": [{"id": "L1", "name": "Bozza AI"}]})
        if path == "/labels":
            return httpx2.Response(200, json={"id": "L2"})
        if path == "/threads/t1/modify":
            return httpx2.Response(200, json={})
        return httpx2.Response(404, json={"error": "secret detail"})

    def box(self) -> GmailMailbox:
        return GmailMailbox(
            lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(self.handler))
        )


def test_the_search_and_a_thread() -> None:
    gmail = Gmail()
    box = gmail.box()
    assert box.thread_ids("in:inbox newer_than:3d", 1) == ["t1"]
    assert gmail.requests[0].url.params["q"] == "in:inbox newer_than:3d"
    assert gmail.requests[0].headers["Authorization"] == "Bearer tok"
    first, second = box.thread("t1").messages
    assert first.body == "Quanto costa?"
    assert first.attachments == 1
    assert first.headers == {"list-unsubscribe": "<mailto:u@example.com>"}
    assert "UNREAD" in first.labels
    assert first.sent_at is not None
    assert "Ciao" in second.body
    assert "a tutti & grazie" in second.body
    assert "x{}" not in second.body


def test_a_draft_goes_in_its_conversation_and_the_labels_are_made() -> None:
    gmail = Gmail()
    box = gmail.box()
    draft = reply_draft(box.thread("t1"), mailbox=MAILBOX, body="Gentile Anna, [FIRMA]")
    assert box.create_draft(draft) == "d1"
    box.add_labels("t1", ["Bozza AI", "AI ignorata"])
    sent = json.loads(gmail.requests[1].content)
    assert sent["message"]["threadId"] == "t1"
    parsed = message_from_bytes(base64.urlsafe_b64decode(sent["message"]["raw"]))
    assert parsed["In-Reply-To"] is None  # the last message has no Message-ID to answer
    assert parsed["To"] == "b@example.com"
    modify = json.loads(gmail.requests[-1].content)
    assert modify == {"addLabelIds": ["L1", "L2"]}
    assert not any("send" in str(request.url) for request in gmail.requests)


def test_the_reply_headers_thread_it() -> None:
    from tests.email.helpers import message, thread

    draft = reply_draft(thread(message()), mailbox=MAILBOX, body="Ciao")
    parsed = message_from_bytes(mime(draft))
    assert parsed["In-Reply-To"] == "<m1@example.com>"
    assert parsed["References"] == "<m1@example.com>"
    assert parsed["Subject"] == "Re: Fisioterapia"
    assert parsed["From"] == MAILBOX


@pytest.mark.parametrize("path", ["/messages/send", "/drafts/send", "/messages/x:send"])
def test_a_request_that_sends_is_refused_before_it_leaves(path: str) -> None:
    gmail = Gmail()
    with pytest.raises(MailboxError, match="never send"):
        gmail.box()._call("POST", path)
    assert gmail.requests == []


def test_an_error_carries_a_status_not_a_body() -> None:
    with pytest.raises(MailboxError, match="status 404") as raised:
        Gmail().box().thread("t9")
    assert "secret" not in str(raised.value)


def test_a_huge_or_crafted_html_part_is_capped_before_it_is_parsed() -> None:
    import time

    from frontdesk.email import gmail

    crafted = "<p>Ciao</p>" + "<script>" * 200_000 + "<style>" + "a" * 3_000_000
    item = {
        "id": "m1",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [{"name": "From", "value": "a@example.com"}],
            "parts": [{"mimeType": "text/html", "body": {"data": _b64(crafted)}}] * 20,
        },
    }
    started = time.perf_counter()
    found = gmail._message(item)
    assert time.perf_counter() - started < 2.0
    assert found.body.startswith("Ciao")
    assert len(found.body) <= gmail.MAX_BODY_CHARS


def test_script_and_style_are_dropped_from_html() -> None:
    from frontdesk.email.gmail import _strip_html

    text = _strip_html("<p>Uno</p><script>alert(1)</script><style>p{}</style><p>Due &amp; tre</p>")
    assert "alert" not in text
    assert "p{}" not in text
    assert "Uno" in text
    assert "Due & tre" in text
