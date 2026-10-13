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

from frontdesk.booking.calcom import (
    BOOKINGS_VERSION,
    EVENT_TYPES_VERSION,
    SCHEDULES_VERSION,
    SLOTS_VERSION,
    CalcomScheduler,
)
from frontdesk.booking.calendar import CalendarUnavailableError, FixtureCalendar
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.booking.slots import Interval
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Integrations

TIRANE = ZoneInfo("Europe/Tirane")
NOW = datetime(2026, 10, 20, 8, tzinfo=TIRANE)
Handler = Callable[[httpx2.Request], httpx2.Response]


def at(day: int, hour: int) -> datetime:
    """A day of October 2026; past the 31st, of November."""
    month, day = (10, day) if day <= 31 else (11, day - 31)
    return datetime(2026, month, day, hour, tzinfo=TIRANE)


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


def _book(scheduler: CalcomScheduler, pack: BusinessPack) -> str:
    return scheduler.book(
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


def test_the_hours_the_doctor_receives_come_from_the_event_type_s_schedule(
    pack: BusinessPack,
) -> None:
    # A full day is no closed day: the hours come from the availability set in cal.com, not from
    # the free slots (Albi, 2026-10-13). A date override replaces the week's hours; one of no
    # length marks the day off.
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.path == "/v2/event-types/7441781":
            return httpx2.Response(200, json={"data": {"id": 7441781, "scheduleId": 2487067}})
        schedule = {
            "timeZone": "Europe/Rome",
            "availability": [{"days": ["Monday"], "startTime": "14:00", "endTime": "18:00"}],
            "overrides": [
                {"date": "2026-10-23", "startTime": "09:00", "endTime": "17:00"},
                {"date": "2026-10-26", "startTime": "00:00", "endTime": "00:00"},
            ],
        }
        return httpx2.Response(200, json={"data": schedule})

    days = _scheduler(answer).receiving(pack, "dr-arben-demo", NOW)
    assert days == {
        at(23, 9).date(): (Interval(at(23, 9), at(23, 17)),),
        at(2 + 31, 14).date(): (Interval(at(2 + 31, 14), at(2 + 31, 18)),),
    }
    assert [(r.url.path, r.headers["cal-api-version"]) for r in seen] == [
        ("/v2/event-types/7441781", EVENT_TYPES_VERSION),
        ("/v2/schedules/2487067", SCHEDULES_VERSION),
    ]


def test_the_hours_of_a_fixture_calendar_are_its_blocks(pack: BusinessPack, tmp_path: Path) -> None:
    path = tmp_path / "calendar.yaml"
    path.write_text(
        '"7441781":\n  blocks:\n    - {start: "2026-10-23 09:00", end: "2026-10-23 11:00"}\n'
        '  busy:\n    - {start: "2026-10-23 09:00", end: "2026-10-23 11:00"}\n',
        encoding="utf-8",
    )
    days = CalendarScheduler(FixtureCalendar(path)).receiving(pack, "dr-arben-demo", NOW)
    assert days == {at(23, 9).date(): (Interval(at(23, 9), at(23, 11)),)}  # full, yet he receives


def test_a_booking_says_whether_the_clinic_must_confirm_it(pack: BusinessPack) -> None:
    # cal.com's "requires confirmation" (Albi, 2026-10-13): the booking waits, "pending".
    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(201, json={"status": "success", "data": {"status": "pending"}})

    assert _book(_scheduler(answer), pack) == "pending"


def test_the_appointments_to_come_are_read_page_by_page(pack: BusinessPack) -> None:
    # One appointment per person at a time: every booking to come counts, made in the chat or by
    # reception, waiting for confirmation too; a cancelled one does not.
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        first = request.url.params["skip"] == "0"
        items = (
            [
                {
                    "status": "accepted",
                    "start": at(23, 9).isoformat(),
                    "attendees": [{"phoneNumber": "+355691234567"}],
                    "metadata": {"frontdesk": "hold1"},
                },
                {
                    "status": "pending",
                    "start": at(26, 12).isoformat(),
                    "attendees": [{}],
                    "bookingFieldsResponses": {"attendeePhoneNumber": "+355690000002"},
                },
                {"status": "cancelled", "start": at(27, 14).isoformat(), "attendees": []},
            ]
            if first
            else [{"status": "accepted", "start": at(28, 9).isoformat(), "attendees": []}]
        )
        return httpx2.Response(200, json={"data": items, "pagination": {"hasNextPage": first}})

    found = _scheduler(answer).upcoming(pack, NOW)
    assert [(item.start, item.phone, item.reference) for item in found] == [
        (at(23, 9), "+355691234567", "hold1"),
        (at(26, 12), "+355690000002", None),
        (at(28, 9), None, None),
    ]
    assert [request.url.params["status"] for request in seen] == ["upcoming,unconfirmed"] * 2
    assert seen[0].url.params["eventTypeIds"] == "7441781"
