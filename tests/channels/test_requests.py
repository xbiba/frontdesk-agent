"""Booking requests on WhatsApp (W1, spec D4-D6): the reception's numbers get the template, the
first tap decides on the calendar, and the patient hears the answer on Instagram while the window
is open. The calendar is the fixture's; the dispatcher records its calls."""

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.calendar import CalendarUnavailableError, FixtureCalendar
from frontdesk.booking.holds import attach, hold, mark_booked
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.channels.inbox import Inbox, conversation_key
from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.channels.requests import Requests, only_buttons, payload
from frontdesk.channels.staff import Role, Staff
from frontdesk.channels.whatsapp import TEMPLATE, Template
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import BookingHold, ChannelConversation, ChannelMessage
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Integrations
from tests.channels.fakes import Clock, RecordingDispatcher

pytestmark = pytest.mark.db
TIRANE = ZoneInfo("Europe/Tirane")
NOW = datetime(2026, 10, 20, 9, tzinfo=TIRANE)
TEN = datetime(2026, 10, 23, 10, tzinfo=TIRANE)
ANA = Staff("+355691234567", "Ana", Role.RECEPTION)
BESA = Staff("+355697654321", "Besa", Role.RECEPTION)
DOCTOR = Staff("+393331234567", "Dr. Arben Demo", Role.DOCTOR)
PATIENT = "1234567890123456"
PHONE = "+355690000001"
CONFIRMED = (
    "Confermata: venerdì 23 ottobre, 10:00, Visita ortopedica con Dr. Arben Demo, Mario Rossi."
)
CALL = (
    "Il paziente non è più raggiungibile in chat: lo chiami, il telefono è sulla prenotazione in "
    "cal.com."
)


@pytest.fixture
def pack(clinic_pack: BusinessPack) -> BusinessPack:
    calendar = CalendarIntegration(
        credentials_ref="adc", calendars={"dr-arben-demo": "cal-arben"}, requires_confirmation=True
    )
    return clinic_pack.model_copy(update={"integrations": Integrations(calendar=calendar)})


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


@pytest.fixture
def scheduler(tmp_path: Path) -> CalendarScheduler:
    path = tmp_path / "calendar.yaml"
    path.write_text("cal-arben: {blocks: [], busy: []}\n", encoding="utf-8")
    return CalendarScheduler(FixtureCalendar(path))


class Setup:
    def __init__(
        self,
        factory: sessionmaker[Session],
        pack: BusinessPack,
        scheduler: Any,
        *,
        language: str = "it",
    ) -> None:
        self.factory = factory
        self.pack = pack
        self.scheduler = scheduler
        self.clock = Clock(NOW)
        self.dispatcher = RecordingDispatcher()
        self.instagram = self._inbox("instagram", None)
        self.whatsapp = self._inbox("whatsapp", frozenset({ANA.user_id, BESA.user_id}))
        self.requests = Requests(
            factory,
            pack=pack,
            scheduler=scheduler,
            staff=(ANA, BESA, DOCTOR),
            whatsapp=self.whatsapp,
            dispatcher=self.dispatcher,
            now=self.clock,
            language=language,
        )

    def _inbox(self, channel: str, allowed: frozenset[str] | None) -> Inbox:
        return Inbox(
            self.factory,
            business_id=self.pack.id,
            channel=channel,
            account_id="acc",
            testers_only=False,
            pairing_code=None,
            now=self.clock,
            allowed_users=allowed,
        )

    def patient_key(self) -> str:
        """The patient's Instagram conversation, with a message of theirs a minute ago."""
        message = Inbound("acc", PATIENT, InboundKind.TEXT, "m-1", "sì", NOW)
        (key,) = self.instagram.accept([message]).work
        return key

    def request(self, conversation: str, *, language: str = "it") -> uuid.UUID:
        """A booking the clinic must confirm, made as the chat makes it."""
        end = TEN + timedelta(minutes=30)
        with business_session(self.factory, self.pack.id) as session:
            hold_id = hold(
                session,
                business_id=self.pack.id,
                conversation_id=conversation,
                doctor_id="dr-arben-demo",
                service_id="orthopedics-visit",
                start=TEN,
                end=end,
                now=NOW,
            )
            assert hold_id is not None
            attach(session, hold_id, name="Mario Rossi", phone=PHONE)
        self.scheduler.book(
            self.pack,
            "dr-arben-demo",
            "orthopedics-visit",
            TEN,
            end,
            name="Mario Rossi",
            phone=PHONE,
            reference=hold_id.hex,
            language=language,
        )
        with business_session(self.factory, self.pack.id) as session:
            mark_booked(session, hold_id, hold_id.hex, pending=True, language=language)
        return hold_id

    def row(self, hold_id: uuid.UUID) -> BookingHold:
        with business_session(self.factory, self.pack.id) as session:
            found = session.get(BookingHold, hold_id)
            assert found is not None
            session.expunge(found)
            return found

    def out(self, channel: str) -> list[tuple[str, str, str | None]]:
        with business_session(self.factory, self.pack.id) as session:
            rows = session.scalars(
                select(ChannelMessage)
                .where(ChannelMessage.channel == channel, ChannelMessage.direction == "out")
                .order_by(ChannelMessage.created_at)
            )
            return [(row.conversation_id, row.kind, row.text) for row in rows]


@pytest.fixture
def setup(factory: sessionmaker[Session], pack: BusinessPack, scheduler: Any) -> Setup:
    return Setup(factory, pack, scheduler)


def _staff_key(member: Staff) -> str:
    return conversation_key("demo-clinic", "whatsapp", member.user_id)


async def test_a_new_request_queues_one_template_per_reception_number(setup: Setup) -> None:
    hold_id = setup.request(setup.patient_key())
    assert await setup.requests.notify() == 1
    sent = setup.out("whatsapp")
    assert sorted(key for key, _, _ in sent) == sorted([_staff_key(ANA), _staff_key(BESA)])
    for _, kind, text in sent:
        assert kind == TEMPLATE
        assert text is not None
        assert Template.from_text(text) == Template(
            "frontdesk_booking_request",
            "it",
            ("venerdì 23 ottobre, 10:00", "Visita ortopedica", "Dr. Arben Demo", "Mario Rossi"),
            (payload(hold_id, accept=True), payload(hold_id, accept=False)),
        )
        assert "690000001" not in text  # never the patient's phone (Albi, 2026-10-16)
    assert sorted(key for key, _ in setup.dispatcher.calls) == sorted(
        [_staff_key(ANA), _staff_key(BESA)]
    )
    assert setup.row(hold_id).notified_at == NOW


async def test_a_request_is_notified_once(setup: Setup) -> None:
    setup.request(setup.patient_key())
    assert await setup.requests.notify() == 1
    assert await setup.requests.notify() == 0
    assert len(setup.out("whatsapp")) == 2


async def test_a_tap_on_confirm_confirms_on_the_calendar_and_tells_both(setup: Setup) -> None:
    patient = setup.patient_key()
    hold_id = setup.request(patient)
    await setup.requests.notify()
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert setup.scheduler.decided == [(hold_id.hex, True)]
    assert decision.reply == CONFIRMED
    told = (
        "La reception ha confermato il suo appuntamento: venerdì 23 ottobre, 10:00, Visita "
        "ortopedica con Dr. Arben Demo."
    )
    assert setup.out("instagram") == [(patient, "fixed", told)]
    assert (patient, timedelta(0)) in setup.dispatcher.calls
    row = setup.row(hold_id)
    assert (row.request_status, row.decided_by, row.decided_at, row.name, row.phone) == (
        "confirmed",
        "Ana",
        NOW,
        None,
        None,
    )


async def test_a_tap_on_reject_declines_and_offers_another_time(setup: Setup) -> None:
    patient = setup.patient_key()
    hold_id = setup.request(patient)
    decision = await setup.requests.decide(BESA.user_id, payload(hold_id, accept=False))
    assert setup.scheduler.decided == [(hold_id.hex, False)]
    assert decision.reply == (
        "Rifiutata: venerdì 23 ottobre, 10:00, Visita ortopedica con Dr. Arben Demo, Mario Rossi."
    )
    told = (
        "La reception non può confermare l'appuntamento di venerdì 23 ottobre, 10:00: per un altro "
        "orario mi scriva qui, oppure chiami la reception al +355 00 000 0000."
    )
    assert setup.out("instagram") == [(patient, "fixed", told)]
    assert (setup.row(hold_id).request_status, setup.row(hold_id).decided_by) == (
        "declined",
        "Besa",
    )


async def test_the_second_tap_hears_who_decided_and_when(setup: Setup) -> None:
    hold_id = setup.request(setup.patient_key())
    await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    setup.clock.advance(minutes=5)
    again = await setup.requests.decide(BESA.user_id, payload(hold_id, accept=False))
    assert again.reply == "Già confermata da Ana alle 09:00."
    assert setup.scheduler.decided == [(hold_id.hex, True)]


class SlowScheduler:
    """The fixture, slow to decide: a second tap comes while the first is on the calendar."""

    def __init__(self, inner: CalendarScheduler) -> None:
        self.inner = inner
        self.decided = inner.decided

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def decide(self, *args: Any, **kwargs: Any) -> None:
        time.sleep(0.3)
        self.inner.decide(*args, **kwargs)


def test_two_staff_tapping_at_once_make_one_decision(
    factory: sessionmaker[Session], pack: BusinessPack, scheduler: CalendarScheduler
) -> None:
    setup = Setup(factory, pack, SlowScheduler(scheduler))
    hold_id = setup.request(setup.patient_key())
    start = threading.Barrier(2)

    def tap(member: Staff, accept: bool) -> str:
        start.wait()
        return asyncio.run(
            setup.requests.decide(member.user_id, payload(hold_id, accept=accept))
        ).reply

    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(tap, [ANA, BESA], [True, False]))
    assert len(scheduler.decided) == 1
    first, second = sorted(replies, key=lambda reply: reply.startswith(("Confermata", "Rifiutata")))
    assert first.startswith("Già") or first == "Un collega la sta decidendo proprio ora."
    assert second.startswith(("Confermata", "Rifiutata"))


class DownScheduler(SlowScheduler):
    def decide(self, *args: Any, **kwargs: Any) -> None:
        raise CalendarUnavailableError("status 503")


async def test_when_cal_com_does_not_answer_the_request_stays_pending(
    factory: sessionmaker[Session], pack: BusinessPack, scheduler: CalendarScheduler
) -> None:
    setup = Setup(factory, pack, DownScheduler(scheduler))
    hold_id = setup.request(setup.patient_key())
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert decision.reply == "cal.com non risponde, riprova."
    row = setup.row(hold_id)
    assert (row.request_status, row.name) == ("pending", "Mario Rossi")
    assert setup.out("instagram") == []


async def test_a_request_gone_from_cal_com_is_said_so(setup: Setup) -> None:
    # Review Focus 1: reception declined it on cal.com directly.
    hold_id = setup.request(setup.patient_key())
    setup.scheduler.decide(
        setup.pack, "dr-arben-demo", TEN, TEN + timedelta(minutes=30), hold_id.hex, accept=False
    )
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert decision.reply == "La richiesta non è più in attesa su cal.com."
    row = setup.row(hold_id)
    assert (row.request_status, row.name) == ("gone", None)
    assert setup.out("instagram") == []
    again = await setup.requests.decide(BESA.user_id, payload(hold_id, accept=True))
    assert again.reply == "La richiesta non è più in attesa su cal.com."


@pytest.mark.parametrize(
    ("user", "text"),
    [
        pytest.param("393339999999", None, id="off-the-list"),
        pytest.param(ANA.user_id, "request:zz:confirm", id="unknown-payload"),
        pytest.param(ANA.user_id, f"request:{uuid.uuid4().hex}:confirm", id="unknown-request"),
    ],
)
async def test_a_tap_from_someone_not_on_the_list_decides_nothing(
    setup: Setup, user: str, text: str | None
) -> None:
    # Review Focus 2.
    hold_id = setup.request(setup.patient_key())
    decision = await setup.requests.decide(user, text or payload(hold_id, accept=True))
    assert decision.reply == "Non riconosco questo pulsante."
    assert setup.scheduler.decided == []
    assert setup.row(hold_id).request_status == "pending"


@pytest.mark.parametrize("gone", ["window", "purged", "staff", "switch"])
async def test_a_closed_window_or_a_purged_conversation_tells_the_staff_to_call(
    setup: Setup, gone: str
) -> None:
    # Review Focus 4; and, from the final review of W1 (I3, spec §6), a thread the clinic's staff
    # has taken on Instagram, or Instagram switched off: the answer would be dropped unsent.
    if gone == "purged":
        hold_id = setup.request("instagram-" + "0" * 40)
    else:
        patient = setup.patient_key()
        hold_id = setup.request(patient)
    if gone == "window":
        setup.clock.advance(hours=25)
    if gone == "staff":
        with business_session(setup.factory, setup.pack.id) as session:
            session.execute(
                update(ChannelConversation)
                .where(ChannelConversation.conversation_id == patient)
                .values(paused_until=NOW + timedelta(hours=1))
            )
    if gone == "switch":
        setup.requests = Requests(
            setup.factory,
            pack=setup.pack,
            scheduler=setup.scheduler,
            staff=(ANA, BESA, DOCTOR),
            whatsapp=setup.whatsapp,
            dispatcher=setup.dispatcher,
            now=setup.clock,
            patient_off=True,
        )
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert decision.reply.startswith("Confermata: ")
    assert decision.reply.endswith(CALL)
    assert setup.out("instagram") == []


async def test_the_staff_reads_albanian_when_reception_prefers_it(
    factory: sessionmaker[Session], pack: BusinessPack, scheduler: CalendarScheduler
) -> None:
    setup = Setup(factory, pack, scheduler, language="sq")
    hold_id = setup.request(setup.patient_key())
    await setup.requests.notify()
    (_, _, text), _ = setup.out("whatsapp")
    assert text is not None
    template = Template.from_text(text)
    assert template.language == "sq"
    assert template.params[:2] == ("e premte 23 tetor, 10:00", "Vizitë ortopedike")
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert decision.reply == (
        "U konfirmua: e premte 23 tetor, 10:00, Vizitë ortopedike me Dr. Arben Demo, Mario Rossi."
    )


def test_the_staff_s_texts_get_one_fixed_reply_in_their_language() -> None:
    # W1 answers only the buttons; the runner sends this to any text or voice note of the staff.
    assert (
        only_buttons("it") == "Per ora rispondo solo ai pulsanti delle richieste di prenotazione."
    )
    assert (
        only_buttons("sq") == "Për momentin u përgjigjem vetëm butonave të kërkesave për rezervim."
    )


# --- Final review of W1, I1: a claim that never finishes ------------------------------------------


async def test_a_stale_claim_can_be_decided_again(setup: Setup) -> None:
    # The worker that claimed it died before cal.com answered: the request stayed DECIDING.
    hold_id = setup.request(setup.patient_key())
    with business_session(setup.factory, setup.pack.id) as session:
        session.execute(
            update(BookingHold).values(
                request_status="deciding", decided_at=NOW - timedelta(minutes=10)
            )
        )
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert decision.reply == CONFIRMED
    assert setup.scheduler.decided == [(hold_id.hex, True)]


async def test_a_fresh_claim_is_left_to_the_colleague(setup: Setup) -> None:
    hold_id = setup.request(setup.patient_key())
    with business_session(setup.factory, setup.pack.id) as session:
        session.execute(
            update(BookingHold).values(
                request_status="deciding", decided_at=NOW - timedelta(minutes=1)
            )
        )
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert decision.reply == "Un collega la sta decidendo proprio ora."
    assert setup.scheduler.decided == []


class BrokenScheduler(SlowScheduler):
    def decide(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("a pack without the doctor's calendar")


async def test_an_unexpected_error_puts_the_request_back(
    factory: sessionmaker[Session], pack: BusinessPack, scheduler: CalendarScheduler
) -> None:
    setup = Setup(factory, pack, BrokenScheduler(scheduler))
    hold_id = setup.request(setup.patient_key())
    with pytest.raises(RuntimeError):
        await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    row = setup.row(hold_id)
    assert (row.request_status, row.name, row.decided_at) == ("pending", "Mario Rossi", None)


async def test_a_dispatch_failure_after_the_decision_still_answers_the_staff(
    factory: sessionmaker[Session], pack: BusinessPack, scheduler: CalendarScheduler
) -> None:
    setup = Setup(factory, pack, scheduler)
    patient = setup.patient_key()
    hold_id = setup.request(patient)
    setup.requests = Requests(
        factory,
        pack=pack,
        scheduler=scheduler,
        staff=(ANA, BESA, DOCTOR),
        whatsapp=setup.whatsapp,
        dispatcher=RecordingDispatcher(fail=True),
        now=setup.clock,
    )
    decision = await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert decision.reply == CONFIRMED
    # The patient's answer waits queued: the sweep gives it its task.
    assert [kind for _, kind, _ in setup.out("instagram")] == ["fixed"]


# --- Final review of W1, I5: what cal.com answered is in the logs ---------------------------------


def _fields(caplog: pytest.LogCaptureFixture, event: str) -> list[tuple[int, object]]:
    return [
        (record.levelno, getattr(record, "fields", None))
        for record in caplog.records
        if record.msg == event
    ]


async def test_a_calendar_that_does_not_answer_is_an_alert_with_its_status(
    factory: sessionmaker[Session],
    pack: BusinessPack,
    scheduler: CalendarScheduler,
    caplog: pytest.LogCaptureFixture,
) -> None:
    setup = Setup(factory, pack, DownScheduler(scheduler))
    hold_id = setup.request(setup.patient_key())
    with caplog.at_level(logging.INFO, logger="frontdesk.channels.requests"):
        await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert _fields(caplog, "requests.calendar_unavailable") == [
        (logging.ERROR, {"reason": "status 503"})
    ]


async def test_a_request_gone_from_cal_com_is_logged(
    setup: Setup, caplog: pytest.LogCaptureFixture
) -> None:
    hold_id = setup.request(setup.patient_key())
    setup.scheduler.decide(
        setup.pack, "dr-arben-demo", TEN, TEN + timedelta(minutes=30), hold_id.hex, accept=False
    )
    with caplog.at_level(logging.INFO, logger="frontdesk.channels.requests"):
        await setup.requests.decide(ANA.user_id, payload(hold_id, accept=True))
    assert [level for level, _ in _fields(caplog, "requests.gone")] == [logging.INFO]
