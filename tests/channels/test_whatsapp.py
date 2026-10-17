"""WhatsApp (W1): the events of a delivery and the sends, texts and templates, on payloads shaped
like Meta's documentation (read 2026-10-17), with synthetic numbers and texts."""

import json
from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest

from frontdesk.channels.messages import Inbound, InboundKind, SendOutcome, SendStatus
from frontdesk.channels.whatsapp import (
    GRAPH,
    TEMPLATE,
    Template,
    WhatsAppSender,
    WhatsAppWebhook,
)

SECRET = "app-secret"
PNID = "106540352242922"
STAFF = "355691234567"
WHEN = "1760430000"
SENT_AT = datetime.fromtimestamp(int(WHEN), UTC)
URL = f"{GRAPH}/v25.0/{PNID}/messages"
PAYLOAD = "request:" + "ab" * 16 + ":confirm"


def _delivery(*messages: dict[str, Any], statuses: list[dict[str, Any]] | None = None) -> bytes:
    value: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "15550783881", "phone_number_id": PNID},
    }
    if messages:
        value["contacts"] = [{"profile": {"name": "Staff"}, "wa_id": STAFF}]
        value["messages"] = list(messages)
    if statuses is not None:
        value["statuses"] = statuses
    entry = {"id": "102290129340398", "changes": [{"value": value, "field": "messages"}]}
    return json.dumps({"object": "whatsapp_business_account", "entry": [entry]}).encode()


def _message(kind: str, **content: Any) -> dict[str, Any]:
    return {"from": STAFF, "id": "wamid.1", "timestamp": WHEN, "type": kind, **content}


def _parse(body: bytes) -> list[Inbound]:
    return WhatsAppWebhook(SECRET).parse(body)


def test_a_text_message_is_read() -> None:
    assert _parse(_delivery(_message("text", text={"body": "Ciao"}))) == [
        Inbound(PNID, STAFF, InboundKind.TEXT, "wamid.1", "Ciao", SENT_AT)
    ]


def test_a_tap_on_a_template_button_is_a_button_event_with_its_payload() -> None:
    tap = _message("button", button={"payload": PAYLOAD, "text": "Conferma"})
    (event,) = _parse(_delivery(tap))
    assert event.kind is InboundKind.BUTTON
    assert event.text == PAYLOAD
    assert (event.account_id, event.user_id, event.message_id) == (PNID, STAFF, "wamid.1")


def test_a_tap_on_an_interactive_reply_button_is_a_button_event() -> None:
    reply = {"type": "button_reply", "button_reply": {"id": PAYLOAD, "title": "Conferma"}}
    (event,) = _parse(_delivery(_message("interactive", interactive=reply)))
    assert (event.kind, event.text) == (InboundKind.BUTTON, PAYLOAD)


@pytest.mark.parametrize("kind", ["audio", "image", "video", "document", "sticker", "location"])
def test_voice_notes_images_and_documents_are_attachments(kind: str) -> None:
    (event,) = _parse(_delivery(_message(kind, **{kind: {"id": "media-1"}})))
    assert event == Inbound(PNID, STAFF, InboundKind.ATTACHMENT, "wamid.1", sent_at=SENT_AT)


def test_delivery_statuses_are_no_events() -> None:
    status = {"id": "wamid.9", "status": "delivered", "timestamp": WHEN, "recipient_id": STAFF}
    assert _parse(_delivery(statuses=[status])) == []


def test_a_message_without_a_phone_number_is_ignored() -> None:
    # Business-scoped user IDs (Meta, April 2026): for a person with a username, "from" is left
    # out unless the business wrote to or heard from that number in the last 30 days. The staff
    # list is by number, so such a message has no one to match.
    hidden = {"id": "wamid.1", "timestamp": WHEN, "type": "text", "text": {"body": "Ciao"}}
    (event,) = _parse(_delivery({**hidden, "from_user_id": "AL.1234567890"}))
    assert (event.kind, event.user_id, event.text) == (InboundKind.IGNORED, "", None)


def test_a_sender_written_with_a_plus_is_read_without_it() -> None:
    (event,) = _parse(_delivery({**_message("text", text={"body": "Ciao"}), "from": f"+{STAFF}"}))
    assert event.user_id == STAFF


@pytest.mark.parametrize(
    "body", [b"not json", b"\xff\xfe", json.dumps({"object": "instagram", "entry": []}).encode()]
)
def test_a_body_that_is_no_json_cannot_be_read(body: bytes) -> None:
    with pytest.raises(ValueError, match="delivery"):
        _parse(body)


def _sender(handler: Any) -> WhatsAppSender:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return WhatsAppSender(lambda: "t0ken", client, phone_number_id=PNID, api_version="v25.0")


def _sent(seen: list[httpx2.Request]) -> Any:
    (request,) = seen
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Bearer t0ken"
    return json.loads(request.content)


async def test_a_text_goes_out_with_the_token() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"messages": [{"id": "wamid.9"}]})

    outcome = await _sender(handler).send(STAFF, "Ciao")
    assert outcome == SendOutcome(SendStatus.SENT, message_id="wamid.9")
    assert _sent(seen) == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": STAFF,
        "type": "text",
        "text": {"preview_url": False, "body": "Ciao"},
    }


async def test_a_template_goes_out_with_its_parameters_and_button_payloads() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"messages": [{"id": "wamid.9"}]})

    params = (
        "venerdì 23 ottobre, 10:00",
        "Trattamento osteopatico",
        "Dr. Salvatore Oliverio",
        "Mario Rossi",
    )
    decline = "request:" + "ab" * 16 + ":decline"
    template = Template("frontdesk_booking_request", "it", params, (PAYLOAD, decline))
    assert Template.from_text(template.to_text()) == template
    outcome = await _sender(handler).send(STAFF, template.to_text(), kind=TEMPLATE)
    assert outcome == SendOutcome(SendStatus.SENT, message_id="wamid.9")
    body = _sent(seen)
    assert (body["to"], body["type"]) == (STAFF, "template")
    sent = body["template"]
    assert (sent["name"], sent["language"]) == ("frontdesk_booking_request", {"code": "it"})
    text, confirm, refuse = sent["components"]
    assert text == {"type": "body", "parameters": [{"type": "text", "text": p} for p in params]}
    assert confirm == {
        "type": "button",
        "sub_type": "quick_reply",
        "index": "0",
        "parameters": [{"type": "payload", "payload": PAYLOAD}],
    }
    assert refuse["index"] == "1"
    assert refuse["parameters"] == [{"type": "payload", "payload": decline}]


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, {"messages": []}, SendOutcome(SendStatus.UNCERTAIN, reason="no_message_id")),
        (500, {"error": {"code": 2}}, SendOutcome(SendStatus.UNCERTAIN, reason="status_500")),
        (401, {"error": {"code": 190}}, SendOutcome(SendStatus.FAILED, reason="token")),
        (
            400,
            {"error": {"code": 131047}},
            SendOutcome(SendStatus.FAILED, reason="status_400_code_131047"),
        ),
    ],
)
async def test_send_outcomes(status: int, body: dict[str, Any], expected: SendOutcome) -> None:
    sender = _sender(lambda request: httpx2.Response(status, json=body))
    assert await sender.send(STAFF, "Ciao") == expected


async def test_a_request_that_never_left_is_tried_again() -> None:
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx2.ConnectError("refused", request=request)
        return httpx2.Response(200, json={"messages": [{"id": "wamid.9"}]})

    assert (await _sender(handler).send(STAFF, "Ciao")).status is SendStatus.SENT
    assert len(calls) == 3


async def test_a_read_error_after_sending_is_uncertain() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadError("reset", request=request)

    outcome = await _sender(handler).send(STAFF, "Ciao")
    assert outcome == SendOutcome(SendStatus.UNCERTAIN, reason="transport")
