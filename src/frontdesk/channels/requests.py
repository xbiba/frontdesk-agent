"""Booking requests the clinic decides on WhatsApp (W1, spec D4-D6).

A booking the clinic confirms waits on cal.com as pending. `notify` sends each new one to every
reception number as the approved template, with the day and time, the service, the doctor and the
patient's name, never the phone (Albi, 2026-10-16). The first tap decides: the request is claimed
in one UPDATE, cal.com is told, and the patient hears the answer on Instagram while Meta's window
is open; otherwise the staff is asked to call. A later tap hears who decided and when.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

from sqlalchemy import CursorResult, and_, func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.calendar import CalendarUnavailableError, RequestGoneError
from frontdesk.booking.flow import phrase
from frontdesk.booking.holds import RequestStatus
from frontdesk.booking.reply import slot_time
from frontdesk.booking.scheduler import Scheduler
from frontdesk.channels.dispatch import Dispatcher, DispatchError
from frontdesk.channels.inbox import IN, REPLY_WINDOW, Inbox, queue_reply
from frontdesk.channels.staff import Role, Staff
from frontdesk.channels.templates import BOOKING_REQUEST
from frontdesk.channels.whatsapp import TEMPLATE, WHATSAPP, Template
from frontdesk.db.engine import business_session
from frontdesk.db.models import BookingHold, ChannelMessage
from frontdesk.observability.logs import log_alert, log_event
from frontdesk.tenancy.pack import BusinessPack, Language

logger = logging.getLogger(__name__)

PAYLOAD: Final = re.compile(r"^request:([0-9a-f]{32}):(confirm|decline)$")
_NOW: Final = timedelta(0)
# A claim older than this lost its worker: cal.com answers in seconds (final review of W1, I1).
STALE_CLAIM: Final = timedelta(minutes=5)
# The staff's replies: Italian by default, Albanian with FRONTDESK_WHATSAPP_STAFF_LANGUAGE=sq.
_COPY: Final[dict[str, dict[str, str]]] = {
    "it": {
        "confirmed": "Confermata: {when}, {service} con {doctor}, {name}.",
        "declined": "Rifiutata: {when}, {service} con {doctor}, {name}.",
        "already_confirmed": "Già confermata da {staff} alle {time}.",
        "already_declined": "Già rifiutata da {staff} alle {time}.",
        "deciding": "Un collega la sta decidendo proprio ora.",
        "unavailable": "cal.com non risponde, riprova.",
        "gone": "La richiesta non è più in attesa su cal.com.",
        "unknown": "Non riconosco questo pulsante.",
        "call": (
            "Il paziente non è più raggiungibile in chat: lo chiami, il telefono è sulla "
            "prenotazione in cal.com."
        ),
    },
    "sq": {
        "confirmed": "U konfirmua: {when}, {service} me {doctor}, {name}.",
        "declined": "U refuzua: {when}, {service} me {doctor}, {name}.",
        "already_confirmed": "E ka konfirmuar tashmë {staff} në orën {time}.",
        "already_declined": "E ka refuzuar tashmë {staff} në orën {time}.",
        "deciding": "Një koleg po e vendos tani.",
        "unavailable": "cal.com nuk përgjigjet, provoni përsëri.",
        "gone": "Kërkesa nuk është më në pritje në cal.com.",
        "unknown": "Nuk e njoh këtë buton.",
        "call": (
            "Pacienti nuk mund të kontaktohet më në chat: telefonojeni, numri është te "
            "rezervimi në cal.com."
        ),
    },
}


_ONLY_BUTTONS: Final = {
    "it": "Per ora rispondo solo ai pulsanti delle richieste di prenotazione.",
    "sq": "Për momentin u përgjigjem vetëm butonave të kërkesave për rezervim.",
}


def only_buttons(language: str) -> str:
    """The one reply to a staff member's text or voice note in W1, which answers only the
    buttons: no model reads the staff's thread."""
    return _ONLY_BUTTONS.get(language, _ONLY_BUTTONS["it"])


def payload(hold_id: uuid.UUID, *, accept: bool) -> str:
    """What a button of the request carries back: the request and the answer, nothing personal."""
    return f"request:{hold_id.hex}:{'confirm' if accept else 'decline'}"


@dataclass(frozen=True)
class Decision:
    reply: str  # for the staff member who tapped, in the staff's language


@dataclass(frozen=True)
class _Request:
    hold_id: uuid.UUID
    doctor_id: str
    service_id: str
    start: datetime
    end: datetime
    reference: str
    name: str
    conversation_id: str
    language: str | None


class Requests:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        pack: BusinessPack,
        scheduler: Scheduler,
        staff: tuple[Staff, ...],
        whatsapp: Inbox,
        dispatcher: Dispatcher,
        now: Callable[[], datetime],
        language: str = "it",
    ) -> None:
        self._factory = session_factory
        self._pack = pack
        self._scheduler = scheduler
        self._staff = staff
        self._whatsapp = whatsapp
        self._dispatcher = dispatcher
        self._now = now
        self._language = cast(Language, language)  # the settings allow it or sq
        self._zone = ZoneInfo(pack.timezone)

    async def notify(self) -> int:
        """Queues the template to every reception number for each request not yet sent, and
        returns how many requests went."""
        reception = [member for member in self._staff if member.role is Role.RECEPTION]
        now = self._now()
        with business_session(self._factory, self._pack.id) as session:
            rows = session.scalars(
                select(BookingHold)
                .where(BookingHold.request_status == RequestStatus.PENDING.value)
                .where(BookingHold.notified_at.is_(None))
                .with_for_update(skip_locked=True)  # two workers never send one request twice
            ).all()
            if not rows or not reception:
                return 0
            keys = [self._whatsapp.open(member.user_id) for member in reception]
            for row in rows:
                text = self._template(row).to_text()
                for key in keys:
                    queue_reply(
                        session,
                        business_id=self._pack.id,
                        channel=WHATSAPP,
                        conversation_id=key,
                        parts=[(TEMPLATE, text)],
                        now=now,
                    )
            session.execute(
                update(BookingHold)
                .where(BookingHold.id.in_([row.id for row in rows]))
                .values(notified_at=now)
            )
        for key in keys:
            await self._dispatcher.dispatch(key, delay=_NOW)
        log_event(logger, "requests.notified", count=len(rows), staff=len(keys))
        return len(rows)

    async def decide(self, user_id: str, text: str) -> Decision:
        """The staff member's tap: the first one decides on the calendar, any later one hears
        how it went."""
        member = next((item for item in self._staff if item.user_id == user_id), None)
        found = PAYLOAD.match(text)
        if member is None or found is None:
            return Decision(self._say("unknown"))
        hold_id, accept = uuid.UUID(found.group(1)), found.group(2) == "confirm"
        request = self._claim(hold_id)
        if request is None:
            return Decision(self._already(hold_id))
        try:
            await asyncio.to_thread(
                self._scheduler.decide,
                self._pack,
                request.doctor_id,
                request.start,
                request.end,
                request.reference,
                accept=accept,
            )
        except CalendarUnavailableError as error:
            # The message is a status, never content (frontdesk.booking.calcom): the first tap that
            # meets a refusal names its cause (final review of W1, I5).
            log_alert(logger, "requests.calendar_unavailable", reason=str(error))
            self._set(hold_id, RequestStatus.PENDING)
            return Decision(self._say("unavailable"))
        except RequestGoneError:
            log_event(logger, "requests.gone")
            self._set(hold_id, RequestStatus.GONE)
            return Decision(self._say("gone"))
        except Exception:
            self._set(hold_id, RequestStatus.PENDING)  # the next tap can decide it
            raise
        told = self._decided(request, member, accept=accept)
        reply = self._say(
            "confirmed" if accept else "declined",
            when=slot_time(request.start.astimezone(self._zone), self._language),
            service=self._service(request.service_id, self._language),
            doctor=self._doctor(request.doctor_id),
            name=request.name,
        )
        log_event(logger, "requests.decided", accept=accept, patient_told=told)
        if not told:
            return Decision(f"{reply} {self._say('call')}")
        try:
            await self._dispatcher.dispatch(request.conversation_id, delay=_NOW)
        except DispatchError:
            # The decision stands and the patient's answer is queued: the sweep gives it its task.
            log_event(logger, "requests.dispatch_failed")
        return Decision(reply)

    def _template(self, row: BookingHold) -> Template:
        start = cast(datetime, row.during.lower)  # a booked hold has its time
        params = (
            slot_time(start.astimezone(self._zone), self._language),
            self._service(row.service_id, self._language),
            self._doctor(row.doctor_id),
            row.name or "-",
        )
        payloads = (payload(row.id, accept=True), payload(row.id, accept=False))
        return Template(BOOKING_REQUEST, self._language, params, payloads)

    def _claim(self, hold_id: uuid.UUID) -> _Request | None:
        """PENDING to DECIDING in one UPDATE: of two taps at once, one gets the request. A claim
        whose worker died is taken again after STALE_CLAIM. Until the decision, decided_at holds
        the claim's time."""
        now = self._now()
        claimable = or_(
            BookingHold.request_status == RequestStatus.PENDING.value,
            and_(
                BookingHold.request_status == RequestStatus.DECIDING.value,
                BookingHold.decided_at < now - STALE_CLAIM,
            ),
        )
        with business_session(self._factory, self._pack.id) as session:
            row = session.execute(
                update(BookingHold)
                .where(BookingHold.id == hold_id)
                .where(claimable)
                .values(request_status=RequestStatus.DECIDING.value, decided_at=now)
                .returning(BookingHold)
            ).scalar_one_or_none()
            if row is None or row.during.lower is None or row.during.upper is None:
                return None
            return _Request(
                hold_id=row.id,
                doctor_id=row.doctor_id,
                service_id=row.service_id,
                start=row.during.lower,
                end=row.during.upper,
                reference=row.event_id or row.id.hex,
                name=row.name or "-",
                conversation_id=row.conversation_id,
                language=row.language,
            )

    def _already(self, hold_id: uuid.UUID) -> str:
        with business_session(self._factory, self._pack.id) as session:
            row = session.get(BookingHold, hold_id)
            status = row.request_status if row is not None else None
            if row is None or status is None:
                return self._say("unknown")
            if status == RequestStatus.DECIDING:
                return self._say("deciding")
            if status in (RequestStatus.CONFIRMED, RequestStatus.DECLINED) and row.decided_at:
                key = (
                    "already_confirmed" if status == RequestStatus.CONFIRMED else "already_declined"
                )
                at = row.decided_at.astimezone(self._zone)
                return self._say(key, staff=row.decided_by or "-", time=f"{at:%H:%M}")
            return self._say("gone")

    def _set(self, hold_id: uuid.UUID, status: RequestStatus) -> None:
        """Back to PENDING, unclaimed, when cal.com did not answer or the decision failed; GONE,
        without the name, when the request is no longer pending there."""
        values: dict[str, object] = {"request_status": status.value, "decided_at": None}
        if status is not RequestStatus.PENDING:
            values.update(name=None, decided_at=self._now())
        with business_session(self._factory, self._pack.id) as session:
            session.execute(
                update(BookingHold)
                .where(BookingHold.id == hold_id)
                .where(BookingHold.request_status == RequestStatus.DECIDING.value)
                .values(**values)
            )

    def _decided(self, request: _Request, member: Staff, *, accept: bool) -> bool:
        """The decision written, the name dropped, and the patient's answer queued while their
        window is open: whether they were told."""
        now = self._now()
        status = RequestStatus.CONFIRMED if accept else RequestStatus.DECLINED
        with business_session(self._factory, self._pack.id) as session:
            written = session.execute(
                update(BookingHold)
                .where(BookingHold.id == request.hold_id)
                .where(BookingHold.request_status == RequestStatus.DECIDING.value)
                .values(
                    request_status=status.value,
                    decided_by=member.name,
                    decided_at=now,
                    name=None,
                )
            )
            if cast(CursorResult[Any], written).rowcount == 0:
                return False  # another tap took the request after this claim went stale
            last = session.scalar(
                select(func.max(ChannelMessage.created_at))
                .where(ChannelMessage.business_id == self._pack.id)
                .where(ChannelMessage.conversation_id == request.conversation_id)
                .where(ChannelMessage.direction == IN)
            )
            if last is None or now - last > REPLY_WINDOW:
                return False
            queue_reply(
                session,
                business_id=self._pack.id,
                channel=request.conversation_id.split("-", 1)[0],
                conversation_id=request.conversation_id,
                parts=[("fixed", self._told(request, accept=accept))],
                now=now,
            )
        return True

    def _told(self, request: _Request, *, accept: bool) -> str:
        known = next((item for item in self._pack.languages if item == request.language), None)
        language = known or self._pack.default_language
        when = slot_time(request.start.astimezone(self._zone), language)
        if not accept:
            return phrase(
                "request_declined", language, when=when, contact=self._pack.location.phone
            )
        return phrase(
            "request_confirmed",
            language,
            when=when,
            service=self._service(request.service_id, language),
            doctor=self._doctor(request.doctor_id),
        )

    def _service(self, service_id: str, language: Language) -> str:
        service = next((item for item in self._pack.services if item.id == service_id), None)
        if service is None:
            return service_id
        names = service.names
        return names.get(language) or names.get(self._pack.default_language) or service_id

    def _doctor(self, doctor_id: str) -> str:
        return next(
            (doctor.name for doctor in self._pack.doctors if doctor.id == doctor_id), doctor_id
        )

    def _say(self, key: str, **values: str) -> str:
        return _COPY.get(self._language, _COPY["it"])[key].format(**values)
