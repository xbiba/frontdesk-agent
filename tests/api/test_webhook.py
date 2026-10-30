"""The webhook (M4 design §4 steps 1-3): Meta's check, the signature, the size, the switch, and a
task for each conversation with work."""

import hashlib
import hmac
import json
import logging

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from frontdesk.api.app import create_webhook_app
from frontdesk.api.webhook import MAX_BODY_BYTES, read_limited
from frontdesk.channels.inbox import BURST, Inbox, conversation_key
from frontdesk.channels.instagram import InstagramWebhook
from frontdesk.channels.whatsapp import WhatsAppWebhook
from frontdesk.channels.wiring import Ingress
from frontdesk.db.engine import make_engine, make_session_factory
from frontdesk.settings import Settings
from tests.channels.fakes import NOW, RecordingDispatcher

pytestmark = pytest.mark.db
SECRET = "app-secret"
VERIFY = "verify-me"
ACCOUNT = "17841400000000001"
ALICE = "1111111111111111"
CODE = "demo-prova-1234"
KEY = conversation_key("demo-clinic", "instagram", ALICE)
PATH = "/webhooks/instagram"


def _app(
    app_url: str, *, off: bool = False, dispatcher: RecordingDispatcher | None = None
) -> tuple[TestClient, RecordingDispatcher, Inbox]:
    inbox = Inbox(
        make_session_factory(make_engine(app_url)),
        business_id="demo-clinic",
        channel="instagram",
        account_id=ACCOUNT,
        testers_only=True,
        pairing_code=CODE,
        now=lambda: NOW,
    )
    sink = dispatcher or RecordingDispatcher()
    ingress = Ingress(
        webhook=InstagramWebhook(SECRET), inbox=inbox, dispatcher=sink, verify_token=VERIFY, off=off
    )
    app = create_webhook_app(Settings(), ingresses=(ingress,), engine=create_engine(app_url))
    return TestClient(app), sink, inbox


def _delivery(text: str) -> bytes:
    item = {
        "sender": {"id": ALICE},
        "recipient": {"id": ACCOUNT},
        "timestamp": 1760430000000,
        "message": {"mid": "m_1", "text": text},
    }
    entry = {"id": ACCOUNT, "time": 1760430000000, "messaging": [item]}
    return json.dumps({"object": "instagram", "entry": [entry]}).encode()


def _signature(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _post(client: TestClient, body: bytes, signature: str | None) -> httpx.Response:
    headers = {"content-type": "application/json"}
    if signature is not None:
        headers["x-hub-signature-256"] = signature
    return client.post(PATH, content=body, headers=headers)


def test_meta_s_check_gets_the_challenge_back(app_url: str) -> None:
    client, _, _ = _app(app_url)
    params = {"hub.mode": "subscribe", "hub.verify_token": VERIFY, "hub.challenge": "1158201444"}
    response = client.get(PATH, params=params)
    assert response.status_code == 200
    assert response.text == "1158201444"


@pytest.mark.parametrize(
    "params",
    [
        {"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "1"},
        {"hub.mode": "unsubscribe", "hub.verify_token": VERIFY, "hub.challenge": "1"},
        {},
    ],
)
def test_a_check_without_the_verify_token_is_refused(app_url: str, params: dict[str, str]) -> None:
    client, _, _ = _app(app_url)
    assert client.get(PATH, params=params).status_code == 403


def test_a_signed_delivery_is_stored_and_its_conversation_dispatched(app_url: str) -> None:
    client, sink, inbox = _app(app_url)
    body = _delivery(CODE)
    assert _post(client, body, _signature(body)).status_code == 200
    assert inbox.count_testers() == 1
    assert sink.calls == [(KEY, BURST)]


def test_an_unsigned_delivery_is_refused_and_stores_nothing(
    app_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    client, sink, inbox = _app(app_url)
    body = _delivery(CODE)
    with caplog.at_level(logging.INFO, logger="frontdesk.api.webhook"):
        assert _post(client, body, None).status_code == 403
        assert _post(client, body, "sha256=" + "0" * 64).status_code == 403
    assert inbox.count_testers() == 0
    assert sink.calls == []
    # Final review of M4, M4: a wrong app secret drops every delivery. A warning, not an alert:
    # anyone on the internet may post an unsigned body.
    levels = {r.levelno for r in caplog.records if r.msg == "channel.bad_signature"}
    assert levels == {logging.WARNING}


def test_a_delivery_too_large_is_refused(app_url: str) -> None:
    client, _, _ = _app(app_url)
    body = b"x" * (MAX_BODY_BYTES + 1)
    assert _post(client, body, _signature(body)).status_code == 413


def test_the_switch_drops_everything(app_url: str) -> None:
    client, sink, inbox = _app(app_url, off=True)
    body = _delivery(CODE)
    assert _post(client, body, _signature(body)).status_code == 200
    assert inbox.count_testers() == 0
    assert sink.calls == []


def test_a_signed_delivery_that_cannot_be_read_is_answered_and_alerted(
    app_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    client, sink, _ = _app(app_url)
    with caplog.at_level(logging.INFO, logger="frontdesk.api.webhook"):
        assert _post(client, b"not json", _signature(b"not json")).status_code == 200
    assert sink.calls == []
    assert any(
        r.levelno == logging.ERROR and r.msg == "channel.unreadable_delivery"
        for r in caplog.records
    )


def test_a_task_that_cannot_be_queued_makes_meta_deliver_again(app_url: str) -> None:
    client, _, inbox = _app(app_url, dispatcher=RecordingDispatcher(fail=True))
    body = _delivery(CODE)
    assert _post(client, body, _signature(body)).status_code == 500
    assert inbox.count_testers() == 1  # stored: Meta's next delivery or the sweep finds it


async def test_a_body_past_the_limit_is_not_read_to_its_end() -> None:
    chunk = b"x" * 400_000
    pulled: list[int] = []

    async def receive() -> dict[str, object]:
        pulled.append(1)
        return {"type": "http.request", "body": chunk, "more_body": len(pulled) < 10}

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    assert await read_limited(request) is None
    assert len(pulled) == 3  # 1.2 MB is past the limit: the other 7 chunks are never read


async def test_a_body_within_the_limit_is_read_whole() -> None:
    parts = [b"uno", b"due", b""]

    async def receive() -> dict[str, object]:
        body = parts.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(parts)}

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    assert await read_limited(request) == b"unodue"


# --- WhatsApp W1: its own route, its own verify token, only the staff's numbers -------------------
WA_SECRET = "wa-app-secret"
WA_VERIFY = "wa-verify"
PNID = "106540352242922"
STAFF = "355691234567"


def _apps(app_url: str) -> tuple[TestClient, RecordingDispatcher, Inbox]:
    """The webhook service with both channels, as a pack that lists WhatsApp has it."""
    factory = make_session_factory(make_engine(app_url))
    sink = RecordingDispatcher()
    instagram = Inbox(
        factory,
        business_id="demo-clinic",
        channel="instagram",
        account_id=ACCOUNT,
        testers_only=True,
        pairing_code=CODE,
        now=lambda: NOW,
    )
    whatsapp = Inbox(
        factory,
        business_id="demo-clinic",
        channel="whatsapp",
        account_id=PNID,
        testers_only=False,
        pairing_code=None,
        now=lambda: NOW,
        allowed_users=frozenset({STAFF}),
    )
    ingresses = (
        Ingress(InstagramWebhook(SECRET), instagram, sink, VERIFY, off=False),
        Ingress(WhatsAppWebhook(WA_SECRET), whatsapp, sink, WA_VERIFY, off=False),
    )
    app = create_webhook_app(Settings(), ingresses=ingresses, engine=create_engine(app_url))
    return TestClient(app), sink, whatsapp


def _tap(user: str) -> bytes:
    message = {
        "from": user,
        "id": "wamid.1",
        "timestamp": "1760430000",
        "type": "button",
        "button": {"payload": "request:" + "ab" * 16 + ":confirm", "text": "Conferma"},
    }
    value = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "15550783881", "phone_number_id": PNID},
        "messages": [message],
    }
    entry = {"id": "102290129340398", "changes": [{"value": value, "field": "messages"}]}
    return json.dumps({"object": "whatsapp_business_account", "entry": [entry]}).encode()


def test_a_signed_whatsapp_tap_is_stored_and_dispatched(app_url: str) -> None:
    client, sink, _ = _apps(app_url)
    body = _tap(STAFF)
    signature = "sha256=" + hmac.new(WA_SECRET.encode(), body, hashlib.sha256).hexdigest()
    headers = {"content-type": "application/json", "x-hub-signature-256": signature}
    assert client.post("/webhooks/whatsapp", content=body, headers=headers).status_code == 200
    assert sink.calls == [(conversation_key("demo-clinic", "whatsapp", STAFF), BURST)]
    stranger = _tap("393339999999")
    signed = "sha256=" + hmac.new(WA_SECRET.encode(), stranger, hashlib.sha256).hexdigest()
    headers["x-hub-signature-256"] = signed
    assert client.post("/webhooks/whatsapp", content=stranger, headers=headers).status_code == 200
    assert len(sink.calls) == 1  # a number off the staff list: nothing stored, nothing to do


def test_the_whatsapp_subscription_check_answers_with_its_own_verify_token(app_url: str) -> None:
    client, _, _ = _apps(app_url)
    params = {"hub.mode": "subscribe", "hub.verify_token": WA_VERIFY, "hub.challenge": "42"}
    assert client.get("/webhooks/whatsapp", params=params).text == "42"
    params["hub.verify_token"] = VERIFY  # Instagram's
    assert client.get("/webhooks/whatsapp", params=params).status_code == 403
