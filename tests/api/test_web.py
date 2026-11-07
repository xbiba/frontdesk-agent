"""The site's routes (design of 2026-11-07 §2-§3): only the listed sites, a message stored once and
dispatched, the replies read back after a cursor, the size and address limits, the switch, tester
mode, the config and the widget."""

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from frontdesk.api.app import create_app, create_webhook_app
from frontdesk.api.ratelimit import SlidingWindow
from frontdesk.api.web import MAX_WEB_BODY_BYTES, web_router
from frontdesk.channels.inbox import BURST, TESTER_WELCOME, Inbox, conversation_key
from frontdesk.channels.runner import Outcome, Runner
from frontdesk.channels.web import WEB, WebReplies, WebSender
from frontdesk.channels.wiring import WebIngress
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import ChannelMessage
from frontdesk.settings import Settings
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.channels.fakes import Clock, FakeTurn, RecordingDispatcher
from tests.tenancy.helpers import web_section

pytestmark = pytest.mark.db
SITE = "https://clinic.example"
SESSION = "5f0c6a1e-2b7d-4c59-9d2e-0123456789ab"
KEY = conversation_key("demo-clinic", WEB, SESSION)
CODE = "demo-prova-1234"


def web_pack(clinic_pack: BusinessPack, **changes: Any) -> BusinessPack:
    raw = clinic_pack.model_dump(mode="json")
    raw["channels"] = {"web": web_section(**changes)}
    return BusinessPack.model_validate(raw)


class Site:
    def __init__(
        self,
        app_url: str,
        pack: BusinessPack,
        *,
        testers: bool = False,
        off: bool = False,
        limiter: SlidingWindow | None = None,
        dispatcher: RecordingDispatcher | None = None,
    ) -> None:
        self.factory = make_session_factory(make_engine(app_url))
        self.clock = Clock()
        self.pack = pack
        self.inbox = Inbox(
            self.factory,
            business_id=pack.id,
            channel=WEB,
            account_id=pack.id,
            testers_only=testers,
            pairing_code=CODE if testers else None,
            now=self.clock,
        )
        self.dispatcher = dispatcher or RecordingDispatcher()
        assert pack.channels.web is not None
        web = WebIngress(
            channel=pack.channels.web,
            pack=pack,
            inbox=self.inbox,
            replies=WebReplies(self.factory, business_id=pack.id),
            dispatcher=self.dispatcher,
            off=off,
        )
        if limiter is None:
            app = create_webhook_app(Settings(), ingresses=(), web=web)
        else:
            app = create_app(Settings(), public=True)
            app.include_router(web_router(web, limiter=limiter))
        self.client = TestClient(app)
        self.turn = FakeTurn()
        self.runner = Runner(
            self.factory, pack=pack, channel=WEB, sender=WebSender(), turn=self.turn, now=self.clock
        )

    def post(self, text: str, mid: str = "m1", origin: str | None = SITE, **extra: Any) -> Any:
        headers = {"content-type": "application/json"}
        if origin is not None:
            headers["origin"] = origin
        body = {"session": SESSION, "id": mid, "text": text, **extra}
        return self.client.post("/web/messages", content=json.dumps(body), headers=headers)

    def replies(self, after: str = "0", origin: str | None = SITE) -> Any:
        headers = {"origin": origin} if origin else {}
        params = {"session": SESSION, "after": after}
        return self.client.get("/web/replies", params=params, headers=headers)

    async def work(self) -> Outcome:
        self.clock.advance(seconds=3)
        return await self.runner.process(KEY)

    def stored(self) -> int:
        with business_session(self.factory, self.pack.id) as session:
            count = session.scalar(
                select(func.count())
                .select_from(ChannelMessage)
                .where(ChannelMessage.direction == "in")
            )
        return count or 0


@pytest.fixture
def pack(clinic_pack: BusinessPack) -> BusinessPack:
    return web_pack(clinic_pack)


async def test_a_message_goes_through_the_worker_and_its_reply_is_read(
    app_url: str, pack: BusinessPack
) -> None:
    site = Site(app_url, pack)
    response = site.post("Siete aperti sabato?")
    assert response.status_code == 202
    assert response.headers["access-control-allow-origin"] == SITE
    assert site.dispatcher.calls == [(KEY, BURST)]
    assert site.replies().json() == {"replies": [], "waiting": True}
    assert await site.work() is Outcome.DONE
    found = site.replies().json()
    assert [r["text"] for r in found["replies"]] == ["Risposta: Siete aperti sabato?"]
    assert found["waiting"] is False
    # After the last cursor, nothing again: the widget never shows a reply twice.
    assert site.replies(after=found["replies"][-1]["cursor"]).json()["replies"] == []


def test_a_retry_of_the_same_message_is_stored_once(app_url: str, pack: BusinessPack) -> None:
    site = Site(app_url, pack)
    assert site.post("Ciao").status_code == 202
    assert site.post("Ciao").status_code == 202
    assert site.stored() == 1


@pytest.mark.parametrize("origin", [None, "https://other.example"])
def test_a_message_from_a_site_not_listed_is_refused(
    app_url: str, pack: BusinessPack, origin: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    site = Site(app_url, pack)
    with caplog.at_level(logging.INFO, logger="frontdesk.api.web"):
        response = site.post("Ciao", origin=origin)
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers
    assert site.stored() == 0
    assert any(r.msg == "web.bad_origin" for r in caplog.records)


def test_reading_from_a_site_not_listed_is_refused(app_url: str, pack: BusinessPack) -> None:
    site = Site(app_url, pack)
    assert site.replies(origin="https://other.example").status_code == 403
    assert (
        site.client.get("/web/config", headers={"origin": "https://x.example"}).status_code == 403
    )
    # The test page on the same origin sends no Origin with a GET.
    assert site.replies(origin=None).status_code == 200


def test_the_preflight_answers_only_the_listed_sites(app_url: str, pack: BusinessPack) -> None:
    site = Site(app_url, pack)
    asked = {"origin": SITE, "access-control-request-method": "POST"}
    response = site.client.options("/web/messages", headers=asked)
    assert response.status_code == 204
    assert response.headers["access-control-allow-origin"] == SITE
    assert "POST" in response.headers["access-control-allow-methods"]
    other = {**asked, "origin": "https://other.example"}
    assert site.client.options("/web/messages", headers=other).status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        json.dumps({"session": "short", "id": "m1", "text": "Ciao"}),
        json.dumps({"session": SESSION, "id": "bad id!", "text": "Ciao"}),
        json.dumps({"session": SESSION, "id": "m1", "text": "   "}),
        json.dumps({"session": SESSION, "id": "m1", "text": 3}),
    ],
)
def test_a_malformed_message_is_refused(app_url: str, pack: BusinessPack, body: str) -> None:
    site = Site(app_url, pack)
    headers = {"origin": SITE, "content-type": "application/json"}
    response = site.client.post("/web/messages", content=body, headers=headers)
    assert response.status_code == 422
    assert site.stored() == 0


def test_a_text_past_the_engine_s_length_is_refused(app_url: str, pack: BusinessPack) -> None:
    site = Site(app_url, pack)
    assert site.post("a" * 2001).json() == {"error": "too_long"}
    headers = {"origin": SITE, "content-type": "application/json"}
    big = b"x" * (MAX_WEB_BODY_BYTES + 1)
    assert site.client.post("/web/messages", content=big, headers=headers).status_code == 413
    assert site.stored() == 0


def test_an_address_past_its_messages_is_refused(app_url: str, pack: BusinessPack) -> None:
    site = Site(app_url, pack, limiter=SlidingWindow(2, 600))
    assert site.post("Uno", mid="m1").status_code == 202
    assert site.post("Due", mid="m2").status_code == 202
    response = site.post("Tre", mid="m3")
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert site.stored() == 2


def test_the_switch_stores_nothing(app_url: str, pack: BusinessPack) -> None:
    site = Site(app_url, pack, off=True)
    assert site.post("Ciao").json() == {"error": "off"}
    assert site.stored() == 0


def test_a_task_that_cannot_be_queued_asks_the_widget_to_try_again(
    app_url: str, pack: BusinessPack
) -> None:
    site = Site(app_url, pack, dispatcher=RecordingDispatcher(fail=True))
    assert site.post("Ciao").status_code == 503
    assert site.stored() == 1  # it waits for the retry, or the sweep


async def test_tester_mode_answers_only_who_wrote_the_code(
    app_url: str, pack: BusinessPack
) -> None:
    site = Site(app_url, pack, testers=True)
    assert site.post("Ciao", mid="m1").status_code == 202
    assert site.replies().json() == {"replies": [], "waiting": False}  # the widget gives the number
    assert site.post(CODE, mid="m2").status_code == 202
    await site.work()
    assert [r["text"] for r in site.replies().json()["replies"]] == [TESTER_WELCOME]
    assert site.post("Ciao di nuovo", mid="m3").status_code == 202
    await site.work()
    texts = [r["text"] for r in site.replies().json()["replies"]]
    assert texts == [TESTER_WELCOME, "Risposta: Ciao di nuovo"]


def test_the_config_gives_the_widget_its_words_and_numbers(
    app_url: str, clinic_pack: BusinessPack
) -> None:
    pack = web_pack(clinic_pack)
    raw = pack.model_dump(mode="json")
    raw["location"]["whatsapp"] = "+355 69 000 0000"
    site = Site(app_url, BusinessPack.model_validate(raw))
    response = site.client.get("/web/config", headers={"origin": SITE})
    assert response.status_code == 200
    config = response.json()
    assert config["name"] == "Klinika Demo"
    assert config["colour"] == "#0a7d5a"
    assert config["default_language"] == "sq"
    assert config["texts"]["it"]["send"] == "Send it"
    assert config["handoff"]["en"] == pack.fixed_texts["en"][FixedTextKey.HANDOFF]
    assert config["phone"] == "+355000000000"
    assert config["whatsapp"] == "355690000000"
    assert config["privacy_url"] == "https://clinic.example/privacy"


def test_the_widget_is_the_same_file_for_every_business(app_url: str, pack: BusinessPack) -> None:
    site = Site(app_url, pack)
    response = site.client.get("/web/widget.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    script = response.text
    assert "Klinika" not in script
    assert "innerHTML" not in script  # plain text only: a reply is never parsed as HTML
    assert "sessionStorage" in script
    assert "attachShadow" in script
