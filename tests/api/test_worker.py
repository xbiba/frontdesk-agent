"""The worker's routes (ADR-0011): a conversation for each task, the sweep, the purge."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, update

from frontdesk.api.app import create_worker_app
from frontdesk.channels.inbox import BURST, TESTER_WELCOME, Inbox, conversation_key, queue_reply
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
CODE = "demo-prova-1234"
KEY = conversation_key(BUSINESS, CHANNEL, ALICE)
STAFF = "355691234567"


class Notified:
    """The requests' stand-in: counts the times the worker asks it to notify."""

    def __init__(self) -> None:
        self.calls = 0

    async def notify(self) -> int:
        self.calls += 1
        return 0


class Service:
    def __init__(self, app_url: str, pack: BusinessPack) -> None:
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
        self.staff_sender = FakeSender()
        self.whatsapp = Inbox(
            self.factory,
            business_id=BUSINESS,
            channel="whatsapp",
            account_id="pnid",
            testers_only=False,
            pairing_code=None,
            now=self.clock,
            allowed_users=frozenset({STAFF}),
        )
        staff_runner = Runner(
            self.factory,
            pack=pack,
            channel="whatsapp",
            sender=self.staff_sender,
            turn=FakeTurn(),
            now=self.clock,
        )
        self.requests = Notified()
        worker = Worker(
            runners={CHANNEL: runner, "whatsapp": staff_runner},
            inbox={CHANNEL: inbox, "whatsapp": self.whatsapp},
            requests=self.requests,  # type: ignore[arg-type]
            dispatcher=self.dispatcher,
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


def test_each_conversation_runs_on_its_channel_s_runner(
    app_url: str, clinic_pack: BusinessPack
) -> None:
    service = Service(app_url, clinic_pack)
    staff = service.whatsapp.open(STAFF)
    with business_session(service.factory, BUSINESS) as session:
        queue_reply(
            session,
            business_id=BUSINESS,
            channel="whatsapp",
            conversation_id=staff,
            parts=[("fixed", "Confermata.")],
            now=NOW,
        )
    service.whatsapp.accept([Inbound("pnid", STAFF, InboundKind.TEXT, "w1", "ok", NOW)])
    assert service.client.post(f"/tasks/conversations/{staff}").status_code == 200
    assert service.staff_sender.sent[0] == (STAFF, "Confermata.")
    assert service.sender.sent == []


def test_after_a_patient_s_turn_the_worker_notifies_new_requests(
    app_url: str, clinic_pack: BusinessPack
) -> None:
    service = Service(app_url, clinic_pack)
    assert service.client.post(f"/tasks/conversations/{KEY}").status_code == 200
    assert service.requests.calls == 1
    staff = service.whatsapp.open(STAFF)
    assert service.client.post(f"/tasks/conversations/{staff}").status_code == 200
    assert service.requests.calls == 1  # a staff run sends no new request


def test_a_task_for_an_unknown_channel_is_not_found(
    app_url: str, clinic_pack: BusinessPack
) -> None:
    service = Service(app_url, clinic_pack)
    unknown = conversation_key(BUSINESS, "telegram", ALICE)
    assert service.client.post(f"/tasks/conversations/{unknown}").status_code == 404


def test_the_token_needs_no_renewal(app_url: str, clinic_pack: BusinessPack) -> None:
    # A system user's token does not expire: the weekly renewal of Instagram Login is gone.
    service = Service(app_url, clinic_pack)
    assert service.client.post("/tasks/refresh-token").status_code == 404
