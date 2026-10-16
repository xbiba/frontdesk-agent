"""The worker's routes (ADR-0011): a conversation for each task, the sweep, the purge, the token."""

import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, update

from frontdesk.api.app import create_worker_app
from frontdesk.channels.inbox import BURST, TESTER_WELCOME, Inbox, conversation_key
from frontdesk.channels.instagram import TokenRefreshError
from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.channels.runner import Runner
from frontdesk.channels.wiring import Worker
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import ChannelConversation
from frontdesk.settings import Settings
from frontdesk.tenancy.pack import BusinessPack
from tests.channels.fakes import NOW, Clock, FakeSender, FakeTurn, RecordingDispatcher

pytestmark = pytest.mark.db
BUSINESS = "demo-clinic"
CHANNEL = "instagram"
ACCOUNT = "17841400000000001"
ALICE = "1111111111111111"
CODE = "mcu-prova-1234"
KEY = conversation_key(BUSINESS, CHANNEL, ALICE)


async def _renewed() -> None:
    return None


class Service:
    def __init__(
        self,
        app_url: str,
        pack: BusinessPack,
        refresh: Callable[[], Awaitable[None]] = _renewed,
    ) -> None:
        self.factory = make_session_factory(make_engine(app_url))
        self.clock = Clock()
        self.sender = FakeSender()
        self.dispatcher = RecordingDispatcher()
        inbox = Inbox(
            self.factory,
            business_id=BUSINESS,
            channel=CHANNEL,
            account_id=ACCOUNT,
            testers_only=True,
            pairing_code=CODE,
            now=self.clock,
        )
        inbox.accept([Inbound(ACCOUNT, ALICE, InboundKind.TEXT, "pair", CODE, NOW)])
        self.inbox = inbox
        runner = Runner(
            self.factory,
            pack=pack,
            channel=CHANNEL,
            sender=self.sender,
            turn=FakeTurn(),
            now=self.clock,
        )
        worker = Worker(
            runner=runner,
            inbox=inbox,
            dispatcher=self.dispatcher,
            refresh=refresh,
            purge_bookings=lambda: 0,
        )
        app = create_worker_app(Settings(), worker=worker, engine=create_engine(app_url))
        self.client = TestClient(app)


def test_a_task_runs_its_conversation(app_url: str, clinic_pack: BusinessPack) -> None:
    service = Service(app_url, clinic_pack)
    assert service.client.post(f"/tasks/conversations/{KEY}").status_code == 200
    assert service.sender.sent == [(ALICE, TESTER_WELCOME)]


def test_a_task_for_a_busy_conversation_is_tried_again(
    app_url: str, clinic_pack: BusinessPack
) -> None:
    service = Service(app_url, clinic_pack)
    with business_session(service.factory, BUSINESS) as session:
        session.execute(update(ChannelConversation).values(lease_until=NOW + timedelta(minutes=1)))
    assert service.client.post(f"/tasks/conversations/{KEY}").status_code == 503
    assert service.sender.sent == []


def test_a_task_for_anything_else_is_not_found(app_url: str, clinic_pack: BusinessPack) -> None:
    service = Service(app_url, clinic_pack)
    assert service.client.post("/tasks/conversations/instagram-xyz").status_code == 404


def test_the_sweep_gives_back_lost_work(app_url: str, clinic_pack: BusinessPack) -> None:
    service = Service(app_url, clinic_pack)
    service.clock.advance(minutes=3)  # the welcome waited past its task
    assert service.client.post("/tasks/sweep").json() == {"work": 1, "uncertain": 0}
    assert service.dispatcher.calls == [(KEY, timedelta(0))]


def test_the_purge_says_what_it_removed(app_url: str, clinic_pack: BusinessPack) -> None:
    service = Service(app_url, clinic_pack)
    service.clock.advance(days=8)
    assert service.client.post("/tasks/purge").json() == {
        "messages": 2,
        "conversations": 0,
        "holds": 0,
    }


def test_the_token_is_renewed(app_url: str, clinic_pack: BusinessPack) -> None:
    calls: list[int] = []

    async def renew() -> None:
        calls.append(1)

    service = Service(app_url, clinic_pack, refresh=renew)
    assert service.client.post("/tasks/refresh-token").status_code == 200
    assert calls == [1]


def test_a_failed_renewal_is_an_alert(
    app_url: str, clinic_pack: BusinessPack, caplog: pytest.LogCaptureFixture
) -> None:
    async def refused() -> None:
        raise TokenRefreshError("status 400")

    service = Service(app_url, clinic_pack, refresh=refused)
    with caplog.at_level(logging.INFO, logger="frontdesk.api.worker"):
        assert service.client.post("/tasks/refresh-token").status_code == 502
    assert any(
        r.levelno == logging.ERROR and r.msg == "channel.token_refresh_failed"
        for r in caplog.records
    )


def test_a_message_left_waiting_by_a_run_gets_a_task_again(
    app_url: str, clinic_pack: BusinessPack
) -> None:
    # Final review of M4, M2: a message younger than the burst window, or one that came during a
    # turn, waited for the sweep, 2 to 12 minutes.
    service = Service(app_url, clinic_pack)
    service.client.post(f"/tasks/conversations/{KEY}")  # the welcome goes out
    service.inbox.accept([Inbound(ACCOUNT, ALICE, InboundKind.TEXT, "m1", "Ciao", NOW)])
    assert service.client.post(f"/tasks/conversations/{KEY}").status_code == 200
    assert service.dispatcher.calls == [(KEY, BURST)]


def test_a_renewal_that_meets_a_network_error_is_an_alert(
    app_url: str, clinic_pack: BusinessPack, caplog: pytest.LogCaptureFixture
) -> None:
    # Final review of M4, M9: a network error escaped as a 500, with no alert.
    async def unreachable() -> None:
        raise httpx2.ConnectError("no route")

    service = Service(app_url, clinic_pack, refresh=unreachable)
    with caplog.at_level(logging.INFO, logger="frontdesk.api.worker"):
        assert service.client.post("/tasks/refresh-token").status_code == 502
    assert any(
        r.levelno == logging.ERROR and r.msg == "channel.token_refresh_failed"
        for r in caplog.records
    )
