"""Google Calendar as the reader: two listings, blocks organised by the calendar, no event content
kept or logged (ADR-0003, amended for M4; security review of 2026-10-12)."""

import json
import logging
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx2
import pytest

from frontdesk.booking.calendar import CalendarUnavailableError, CalendarView, opening_intervals
from frontdesk.booking.google_calendar import (
    BLOCK_FIELDS,
    BUSY_FIELDS,
    MAX_PAGES,
    TIMEOUT_SECONDS,
    GoogleCalendar,
    calendar_http_client,
)
from frontdesk.booking.slots import Interval
from frontdesk.tenancy.pack import OpeningHours

TIRANE = ZoneInfo("Europe/Tirane")
HOURS = [OpeningHours(days="mon-fri", opens="09:00", closes="18:00")]
START = datetime(2026, 10, 19, 0, tzinfo=TIRANE)
END = datetime(2026, 11, 2, 0, tzinfo=TIRANE)
PATIENT = "Mario Rossi, controllo"
OWN = {"self": True}

Handler = Callable[[httpx2.Request], httpx2.Response]
Page = dict[str, Any]


def opening(day: date) -> list[Interval]:
    return opening_intervals(HOURS, day, TIRANE)


def at(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=TIRANE)


def _page(*items: dict[str, Any], page: str | None = None) -> Page:
    return {"items": list(items), **({"nextPageToken": page} if page else {})}


def _timed(event_id: str, day: int, start: int, end: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": event_id,
        "start": {"dateTime": at(day, start).isoformat()},
        "end": {"dateTime": at(day, end).isoformat()},
        **extra,
    }


def _reader(
    blocks: list[Page], busy: list[Page], seen: list[httpx2.Request] | None = None
) -> GoogleCalendar:
    pages = {"blocks": iter(blocks), "busy": iter(busy)}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        listing = "blocks" if "q" in request.url.params else "busy"
        return httpx2.Response(200, json=next(pages[listing]))

    return GoogleCalendar(lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(handler)))


def _read(reader: GoogleCalendar) -> CalendarView:
    return reader.read("cal-1", START, END, marker="Disponibile", zone=TIRANE, opening=opening)


def test_two_listings_one_with_the_marker_and_neither_with_notes() -> None:
    seen: list[httpx2.Request] = []
    _read(_reader([_page()], [_page()], seen))
    blocks, busy = seen
    window = {
        "singleEvents": "true",
        "timeMin": START.isoformat(),
        "timeMax": END.isoformat(),
        "timeZone": "Europe/Tirane",
        "maxResults": "250",
    }
    assert (blocks.method, busy.method) == ("GET", "GET")
    assert blocks.url.path == busy.url.path == "/calendar/v3/calendars/cal-1/events"
    assert dict(blocks.url.params) == {**window, "q": "Disponibile", "fields": BLOCK_FIELDS}
    assert dict(busy.url.params) == {**window, "fields": BUSY_FIELDS}
    assert "summary" not in BUSY_FIELDS
    assert "description" not in BLOCK_FIELDS + BUSY_FIELDS
    assert blocks.headers["authorization"] == "Bearer tok"


def test_only_ordinary_marker_events_of_the_calendar_are_blocks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    found = _page(
        _timed("b1", 23, 9, 13, summary=" disponibile ", organizer=OWN),
        _timed("a1", 23, 10, 11, summary=PATIENT, organizer=OWN),  # "Disponibile" in its notes
        _timed("x1", 26, 9, 13, summary="Disponibile", organizer={"self": False}),  # an invitation
        _timed("x2", 27, 9, 13, summary="Disponibile", organizer=OWN, status="cancelled"),
        _timed("x3", 28, 9, 13, summary="Disponibile", organizer=OWN, eventType="outOfOffice"),
    )
    view = _read(_reader([found], [_page()]))
    assert view.blocks == (Interval(at(23, 9), at(23, 13)),)
    assert PATIENT not in repr(view)
    assert "Rossi" not in caplog.text


def test_every_event_but_a_block_is_busy_whatever_its_show_as() -> None:
    blocks = _page(_timed("b1", 23, 9, 13, summary="Disponibile", organizer=OWN))
    busy = _page(
        _timed("b1", 23, 9, 13),  # the block itself, left as busy: not busy time
        _timed("a1", 23, 10, 11),
        _timed("a2", 23, 11, 12, transparency="transparent"),  # an appointment marked as free
        _timed("x1", 23, 12, 13, status="cancelled"),
        _timed("w1", 23, 9, 18, eventType="workingLocation"),
    )
    view = _read(_reader([blocks], [busy]))
    assert view.busy == (Interval(at(23, 10), at(23, 11)), Interval(at(23, 11), at(23, 12)))


def test_all_day_events_are_clipped_to_the_window() -> None:
    year = {"start": {"date": "2026-01-01"}, "end": {"date": "2027-01-01"}}
    blocks = _page({"id": "b1", "summary": "Disponibile", "organizer": OWN, **year})
    busy = _page({"id": "h1", "start": {"date": "2026-10-28"}, "end": {"date": "2026-10-29"}})
    view = _read(_reader([blocks], [busy]))
    # The opening hours of the weekdays from 19 October to 2 November, both included.
    assert len(view.blocks) == 11
    assert view.blocks[0] == Interval(at(19, 9), at(19, 18))
    assert view.busy == (Interval(at(28, 0), at(29, 0)),)


def test_every_page_is_read() -> None:
    first = _page(_timed("b1", 23, 9, 10, summary="Disponibile", organizer=OWN), page="p2")
    second = _page(_timed("b2", 24, 9, 10, summary="Disponibile", organizer=OWN))
    seen: list[httpx2.Request] = []
    view = _read(_reader([first, second], [_page()], seen))
    assert len(view.blocks) == 2
    assert seen[1].url.params["pageToken"] == "p2"


def test_endless_pages_stop_with_an_error() -> None:
    endless = [_page(page="again")] * (MAX_PAGES + 1)
    with pytest.raises(CalendarUnavailableError, match="too many pages"):
        _read(_reader(endless, [_page()]))


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        pytest.param(
            lambda _: httpx2.Response(403, json={"error": {}}), "status 403", id="forbidden"
        ),
        pytest.param(
            lambda _: httpx2.Response(404, json={"error": {}}), "status 404", id="not-found"
        ),
        pytest.param(
            lambda _: httpx2.Response(200, text="<html>"), "malformed response", id="not-json"
        ),
        pytest.param(
            lambda _: httpx2.Response(200, json=[]), "malformed response", id="not-an-object"
        ),
    ],
)
def test_a_calendar_that_cannot_be_read_raises_with_a_status_only(
    handler: Handler, message: str
) -> None:
    reader = GoogleCalendar(lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(handler)))
    with pytest.raises(CalendarUnavailableError, match=message):
        _read(reader)


def test_a_timeout_or_a_malformed_event_cannot_be_read() -> None:
    def timeout(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    slow = GoogleCalendar(lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(timeout)))
    with pytest.raises(CalendarUnavailableError, match="ReadTimeout"):
        _read(slow)
    no_end = _page(
        {"id": "b1", "summary": "Disponibile", "organizer": OWN, "start": {"date": "2026-10-23"}}
    )
    with pytest.raises(CalendarUnavailableError, match="malformed event: KeyError"):
        _read(_reader([no_end], [_page()]))


def test_a_token_that_cannot_be_had_cannot_read_the_calendar() -> None:
    def no_token() -> str:
        raise RuntimeError("no credentials")

    answer = httpx2.MockTransport(lambda _: httpx2.Response(200, json=_page()))
    reader = GoogleCalendar(no_token, httpx2.Client(transport=answer))
    with pytest.raises(CalendarUnavailableError, match="credentials: RuntimeError"):
        _read(reader)


def test_the_client_ignores_proxies_and_verifies_with_certifi() -> None:
    client = calendar_http_client()
    assert client.trust_env is False
    assert client.timeout.read == TIMEOUT_SECONDS


# --- Writing an appointment (M5) -------------------------------------------------------------------


def _writer(handler: Handler) -> GoogleCalendar:
    return GoogleCalendar(lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(handler)))


def _book(writer: GoogleCalendar) -> None:
    writer.book(
        "cal-1",
        at(23, 10),
        at(23, 11),
        event_id="abc123",
        title="Trattamento osteopatico — Mario Rossi",
        description="Telefono: +355691234567",
    )


def test_an_appointment_is_one_event_with_its_own_id() -> None:
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"id": "abc123"})

    _book(_writer(answer))
    (insert,) = seen
    assert (insert.method, insert.url.path) == ("POST", "/calendar/v3/calendars/cal-1/events")
    assert json.loads(insert.content) == {
        "id": "abc123",
        "summary": "Trattamento osteopatico — Mario Rossi",
        "description": "Telefono: +355691234567",
        "start": {"dateTime": at(23, 10).isoformat()},
        "end": {"dateTime": at(23, 11).isoformat()},
        "extendedProperties": {"private": {"frontdesk": "booking"}},
    }


def test_a_repeated_request_meets_its_own_id_and_is_the_same_appointment() -> None:
    _book(_writer(lambda _: httpx2.Response(409, json={"error": {}})))


@pytest.mark.parametrize("found", [200, 404])
def test_a_lost_answer_is_checked_by_the_event_id(found: int) -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            raise httpx2.ReadTimeout("lost", request=request)
        assert request.url.path.endswith("/events/abc123")
        return httpx2.Response(found, json={"id": "abc123", "status": "confirmed"})

    if found == 200:
        _book(_writer(answer))
    else:
        with pytest.raises(CalendarUnavailableError, match="may not be written"):
            _book(_writer(answer))


def test_a_server_error_is_checked_and_a_refusal_raises() -> None:
    def server_error(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            return httpx2.Response(503)
        return httpx2.Response(200, json={"id": "abc123", "status": "cancelled"})

    with pytest.raises(CalendarUnavailableError, match="may not be written"):
        _book(_writer(server_error))
    with pytest.raises(CalendarUnavailableError, match="status 403"):
        _book(_writer(lambda _: httpx2.Response(403, json={"error": {}})))
