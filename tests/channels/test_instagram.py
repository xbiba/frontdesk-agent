"""Instagram (M4): the signature, the events of a delivery and the Send API, on payloads shaped like
Meta's documentation, with synthetic IDs and texts."""

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest

from frontdesk.channels.instagram import (
    GRAPH,
    InstagramSender,
    InstagramWebhook,
    TokenRefreshError,
    account_user_id,
    parse_events,
    refresh_token,
    signature_ok,
    subscribe,
)
from frontdesk.channels.messages import Inbound, InboundKind, SendOutcome, SendStatus

SECRET = "app-secret"
ACCOUNT = "17841400000000001"
PERSON = "1234567890123456"
WHEN = 1760430000000
SENT_AT = datetime.fromtimestamp(WHEN / 1000, UTC)
URL = f"{GRAPH}/v25.0/{ACCOUNT}/messages"


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _delivery(*items: dict[str, Any], account: str = ACCOUNT) -> bytes:
    entry = {"id": account, "time": WHEN, "messaging": list(items)}
    return json.dumps({"object": "instagram", "entry": [entry]}).encode()


def _from_person(**message: Any) -> dict[str, Any]:
    return {
        "sender": {"id": PERSON},
        "recipient": {"id": ACCOUNT},
        "timestamp": WHEN,
        "message": {"mid": "m_1", **message},
    }


def test_a_delivery_signed_with_the_app_secret_is_accepted() -> None:
    body = _delivery(_from_person(text="ciao"))
    assert signature_ok(SECRET, body, _sign(body))
    assert InstagramWebhook(SECRET).signature_ok(body, _sign(body))


@pytest.mark.parametrize("header", [None, "", "sha1=abc", "sha256=" + "0" * 64, "sha256=àè"])
def test_a_delivery_without_the_right_signature_is_refused(header: str | None) -> None:
    assert not signature_ok(SECRET, _delivery(_from_person(text="ciao")), header)


def test_a_signature_of_another_body_or_secret_is_refused() -> None:
    body = _delivery(_from_person(text="ciao"))
    assert not signature_ok(SECRET, body + b" ", _sign(body))
    assert not signature_ok(SECRET, body, _sign(body, "other"))


def test_a_text_message_is_read() -> None:
    (event,) = parse_events(_delivery(_from_person(text="  Siete aperti sabato? ")))
    assert event == Inbound(
        ACCOUNT, PERSON, InboundKind.TEXT, "m_1", "Siete aperti sabato?", SENT_AT
    )


def test_a_story_reply_counts_as_text() -> None:
    reply = {"story": {"url": "https://cdn.example/s", "id": "s1"}}
    (event,) = parse_events(_delivery(_from_person(text="Quanto costa?", reply_to=reply)))
    assert event.kind is InboundKind.TEXT


def test_an_echo_names_the_person_it_was_sent_to() -> None:
    echo = {
        "sender": {"id": ACCOUNT},
        "recipient": {"id": PERSON},
        "timestamp": WHEN,
        "message": {"mid": "m_2", "text": "Buongiorno", "is_echo": True},
    }
    (event,) = parse_events(_delivery(echo))
    assert event == Inbound(ACCOUNT, PERSON, InboundKind.ECHO, "m_2", "Buongiorno", SENT_AT)


def test_an_unsent_message_is_read_as_deleted() -> None:
    (event,) = parse_events(_delivery(_from_person(is_deleted=True)))
    assert (event.kind, event.message_id) == (InboundKind.DELETED, "m_1")


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("image", InboundKind.ATTACHMENT),
        ("audio", InboundKind.ATTACHMENT),
        ("file", InboundKind.ATTACHMENT),
        ("share", InboundKind.IGNORED),
        ("story_mention", InboundKind.IGNORED),
        ("ig_reel", InboundKind.IGNORED),
    ],
)
def test_attachments_are_media_or_ignored(kind: str, expected: InboundKind) -> None:
    attachments = [{"type": kind, "payload": {"url": "https://cdn.example/a"}}]
    (event,) = parse_events(_delivery(_from_person(attachments=attachments)))
    assert event.kind is expected
    assert event.text is None


def test_reactions_and_reads_are_ignored() -> None:
    base = {"sender": {"id": PERSON}, "recipient": {"id": ACCOUNT}, "timestamp": WHEN}
    reaction = {**base, "reaction": {"mid": "m_1", "action": "react", "reaction": "love"}}
    read = {**base, "read": {"mid": "m_1"}}
    kinds = [event.kind for event in parse_events(_delivery(reaction, read))]
    assert kinds == [InboundKind.IGNORED, InboundKind.IGNORED]


def test_events_keep_their_order_across_entries() -> None:
    first = json.loads(_delivery(_from_person(text="uno")))["entry"]
    second = json.loads(_delivery({**_from_person(), "message": {"mid": "m_2", "text": "due"}}))
    body = json.dumps({"object": "instagram", "entry": first + second["entry"]}).encode()
    assert [event.text for event in parse_events(body)] == ["uno", "due"]


def test_an_event_of_an_unknown_shape_is_ignored_not_an_error() -> None:
    (event,) = parse_events(_delivery({"something": "new"}))
    assert event.kind is InboundKind.IGNORED
    assert event.user_id == ""


@pytest.mark.parametrize(
    "body", [b"not json", b"\xff\xfe", json.dumps({"object": "page", "entry": []}).encode()]
)
def test_a_delivery_that_is_not_instagrams_raises(body: bytes) -> None:
    with pytest.raises(ValueError, match="delivery"):
        parse_events(body)


def _sender(handler: Any) -> InstagramSender:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return InstagramSender(lambda: "t0ken", client, account_id=ACCOUNT, api_version="v25.0")


async def test_a_text_is_sent_to_the_person_with_the_token() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"recipient_id": PERSON, "message_id": "out_1"})

    outcome = await _sender(handler).send(PERSON, "Siamo aperti sabato.")
    assert outcome == SendOutcome(SendStatus.SENT, message_id="out_1")
    (request,) = seen
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Bearer t0ken"
    assert json.loads(request.content) == {
        "recipient": {"id": PERSON},
        "message": {"text": "Siamo aperti sabato."},
    }


async def test_a_request_that_never_left_is_tried_again() -> None:
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx2.ConnectError("refused", request=request)
        return httpx2.Response(200, json={"message_id": "out_1"})

    assert (await _sender(handler).send(PERSON, "ciao")).status is SendStatus.SENT
    assert len(calls) == 3


async def test_three_connection_failures_fail_the_send() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    outcome = await _sender(handler).send(PERSON, "ciao")
    assert outcome == SendOutcome(SendStatus.FAILED, reason="connect")


async def test_a_timeout_after_sending_is_uncertain_and_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        raise httpx2.ReadTimeout("slow", request=request)

    outcome = await _sender(handler).send(PERSON, "ciao")
    assert outcome == SendOutcome(SendStatus.UNCERTAIN, reason="transport")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (500, {"error": {"code": 2}}, SendOutcome(SendStatus.UNCERTAIN, reason="status_500")),
        (400, {"error": {"code": 190}}, SendOutcome(SendStatus.FAILED, reason="token")),
        (
            400,
            {"error": {"code": 10, "error_subcode": 2018278}},
            SendOutcome(SendStatus.FAILED, reason="status_400_code_10"),
        ),
        (403, {}, SendOutcome(SendStatus.FAILED, reason="status_403")),
        (200, {"recipient_id": PERSON}, SendOutcome(SendStatus.UNCERTAIN, reason="no_message_id")),
    ],
)
async def test_meta_s_answers_become_outcomes(
    status: int, body: dict[str, Any], expected: SendOutcome
) -> None:
    sender = _sender(lambda request: httpx2.Response(status, json=body))
    assert await sender.send(PERSON, "ciao") == expected


async def test_the_token_is_refreshed() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"access_token": "new", "expires_in": 5183944})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        assert await refresh_token(client, "old") == "new"
    (request,) = seen
    assert request.url.path == "/refresh_access_token"
    assert request.url.params["grant_type"] == "ig_refresh_token"


async def test_a_refused_refresh_names_only_the_status() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": {"message": "the token old is expired"}})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(TokenRefreshError, match=r"^status 400$"):
            await refresh_token(client, "old")


async def test_the_account_id_comes_from_me() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v25.0/me"
        return httpx2.Response(200, json={"user_id": ACCOUNT, "username": "mcu_test", "id": "9"})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        assert await account_user_id(client, "t", "v25.0") == (ACCOUNT, "mcu_test")


async def test_the_account_is_subscribed_to_messages() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"success": True})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        await subscribe(client, "t", "v25.0")
    assert seen[0].url.path == "/v25.0/me/subscribed_apps"
    assert seen[0].url.params["subscribed_fields"] == "messages"
