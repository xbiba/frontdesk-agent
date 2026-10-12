"""cal.com as the scheduler: the free slots of an event type, and a booking that is said only once
cal.com has it."""

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx2
import pytest

from frontdesk.booking.calcom import BOOKINGS_VERSION, SLOTS_VERSION, CalcomScheduler
from frontdesk.booking.calendar import CalendarUnavailableError, FixtureCalendar
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.booking.slots import Interval
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Integrations

TIRANE = ZoneInfo("Europe/Tirane")
NOW = datetime(2026, 10, 20, 8, tzinfo=TIRANE)
Handler = Callable[[httpx2.Request], httpx2.Response]


def at(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=TIRANE)


@pytest.fixture
def pack(clinic_pack: BusinessPack) -> BusinessPack:
    calendar = CalendarIntegration(credentials_ref="calcom", calendars={"dr-arben-demo": "7441781"})
    return clinic_pack.model_copy(update={"integrations": Integrations(calendar=calendar)})


def _scheduler(handler: Handler) -> CalcomScheduler:
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return CalcomScheduler("cal_test_key", client)


def _slot(day: int, hour: int) -> dict[str, str]:
    return {
        "start": at(day, hour).isoformat(),
        "end": (at(day, hour) + timedelta(hours=1)).isoformat(),
    }


def test_the_free_slots_of_the_doctor_s_event_type(pack: BusinessPack) -> None:
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        days = {"2026-10-26": [_slot(26, 12)], "2026-10-23": [_slot(23, 9), _slot(23, 10)]}
        return httpx2.Response(200, json={"data": days})

    slots = _scheduler(answer).free_slots(pack, "dr-arben-demo", "orthopedics-visit", NOW)
    assert [slot.start for slot in slots] == [at(23, 9), at(23, 10), at(26, 12)]
    (request,) = seen
    assert request.url.path == "/v2/slots"
    assert dict(request.url.params) == {
        "eventTypeId": "7441781",
        "start": "2026-10-20T06:00:00Z",
        "end": "2026-11-03T07:00:00Z",  # 14 days later, after the change to winter time
        "timeZone": "Europe/Tirane",
        "format": "range",
    }
    assert request.headers["cal-api-version"] == SLOTS_VERSION
    assert request.headers["authorization"] == "Bearer cal_test_key"


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        pytest.param(lambda _: httpx2.Response(401, json={}), "status 401", id="refused"),
        pytest.param(
            lambda _: httpx2.Response(200, json={"status": "success"}),
            "malformed slots",
            id="no-data",
        ),
        pytest.param(
            lambda _: httpx2.Response(200, json={"data": {"d": [{"start": "x"}]}}),
            "malformed slot",
            id="bad-slot",
        ),
        pytest.param(
            lambda _: httpx2.Response(200, text="<html>"), "malformed response", id="not-json"
        ),
    ],
)
def test_slots_that_cannot_be_read_raise_with_a_status_only(
    pack: BusinessPack, handler: Handler, message: str
) -> None:
    with pytest.raises(CalendarUnavailableError, match=message) as raised:
        _scheduler(handler).free_slots(pack, "dr-arben-demo", "orthopedics-visit", NOW)
    assert "cal_test_key" not in str(raised.value)


def _book(scheduler: CalcomScheduler, pack: BusinessPack) -> None:
    scheduler.book(
        pack,
        "dr-arben-demo",
        "orthopedics-visit",
        at(23, 10),
        at(23, 11),
        name="Mario Prova",
        phone="+355690000001",
        reference="hold1",
        language="it",
    )


def test_a_booking_carries_the_name_the_phone_and_our_reference(pack: BusinessPack) -> None:
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            201, json={"status": "success", "data": {"uid": "u1", "status": "accepted"}}
        )

    _book(_scheduler(answer), pack)
    (request,) = seen
    assert (request.method, request.url.path) == ("POST", "/v2/bookings")
    assert request.headers["cal-api-version"] == BOOKINGS_VERSION
    assert json.loads(request.content) == {
        "start": "2026-10-23T08:00:00Z",
        "eventTypeId": 7441781,
        "attendee": {
            "name": "Mario Prova",
            "timeZone": "Europe/Tirane",
            "phoneNumber": "+355690000001",
        },
        "metadata": {"frontdesk": "hold1"},
    }


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(httpx2.Response(400, json={"status": "error"}), id="taken"),
        pytest.param(
            httpx2.Response(201, json={"status": "success", "data": {"status": "rejected"}}),
            id="rejected",
        ),
        pytest.param(httpx2.Response(201, json={"status": "success"}), id="no-data"),
    ],
)
def test_a_booking_cal_com_does_not_accept_raises(
    pack: BusinessPack, answer: httpx2.Response
) -> None:
    with pytest.raises(CalendarUnavailableError):
        _book(_scheduler(lambda _: answer), pack)


@pytest.mark.parametrize(("listed", "booked"), [("hold1", True), ("someone-else", False)])
def test_a_lost_answer_is_checked_by_our_reference(
    pack: BusinessPack, listed: str, booked: bool
) -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            raise httpx2.ReadTimeout("lost", request=request)
        bookings = [{"status": "accepted", "metadata": {"frontdesk": listed}}]
        return httpx2.Response(200, json={"status": "success", "data": bookings})

    if booked:
        _book(_scheduler(answer), pack)
    else:
        with pytest.raises(CalendarUnavailableError, match="may not be booked"):
            _book(_scheduler(answer), pack)


def test_a_server_error_is_checked_too(pack: BusinessPack) -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            return httpx2.Response(502)
        return httpx2.Response(503)

    with pytest.raises(CalendarUnavailableError, match="may not be booked"):
        _book(_scheduler(answer), pack)


def test_the_calendar_scheduler_writes_the_entry_reception_reads(
    pack: BusinessPack, tmp_path: Path
) -> None:
    path = tmp_path / "calendar.yaml"
    path.write_text('"7441781": {blocks: [], busy: []}\n', encoding="utf-8")
    calendar = FixtureCalendar(path)
    titles: list[dict[str, Any]] = []
    original = calendar.book

    def keep(*args: Any, **kwargs: Any) -> None:
        titles.append(kwargs)
        original(*args, **kwargs)

    calendar.book = keep  # type: ignore[method-assign]
    CalendarScheduler(calendar).book(
        pack,
        "dr-arben-demo",
        "orthopedics-visit",
        at(23, 10),
        at(23, 11),
        name="Mario Prova",
        phone="+355690000001",
        reference="hold1",
        language="it",
    )
    assert titles == [
        {
            "event_id": "hold1",
            "title": "Vizitë ortopedike — Mario Prova",  # the clinic's default language
            "description": "Telefoni: +355690000001\nRezervuar nga chat-i.",
        }
    ]
    assert calendar.booked["7441781"] == {"hold1": Interval(at(23, 10), at(23, 11))}
