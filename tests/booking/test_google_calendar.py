"""Google Calendar as the reader: two reads, marker events only, no event content kept or logged."""

import json
import logging
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx2
import pytest

from frontdesk.booking.calendar import CalendarUnavailableError, opening_intervals
from frontdesk.booking.google_calendar import (
    EVENT_FIELDS,
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

Handler = Callable[[httpx2.Request], httpx2.Response]


def opening(day: date) -> list[Interval]:
    return opening_intervals(HOURS, day, TIRANE)


def at(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=TIRANE)


def _events(*items: dict[str, Any], page: str | None = None) -> dict[str, Any]:
    return {"items": list(items), **({"nextPageToken": page} if page else {})}


def _timed(day: int, start: int, end: int, summary: str, **extra: str) -> dict[str, Any]:
    return {
        "summary": summary,
        "start": {"dateTime": at(day, start).isoformat()},
        "end": {"dateTime": at(day, end).isoformat()},
        **extra,
    }


def _busy(*pairs: tuple[datetime, datetime]) -> dict[str, Any]:
    busy = [{"start": a.isoformat(), "end": b.isoformat()} for a, b in pairs]
    return {"calendars": {"cal-1": {"busy": busy}}}


def _reader(
    events: list[dict[str, Any]], busy: dict[str, Any], seen: list[httpx2.Request]
) -> GoogleCalendar:
    pages = iter(events)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.path.endswith("/freeBusy"):
            return httpx2.Response(200, json=busy)
        return httpx2.Response(200, json=next(pages))

    return GoogleCalendar(lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(handler)))


def _read(reader: GoogleCalendar) -> Any:
    return reader.read("cal-1", START, END, marker="Disponibile", zone=TIRANE, opening=opening)


def test_two_reads_with_the_marker_query_and_the_fields_limited() -> None:
    seen: list[httpx2.Request] = []
    _read(_reader([_events()], _busy(), seen))
    events, freebusy = seen
    assert events.method == "GET"
    assert events.url.path == "/calendar/v3/calendars/cal-1/events"
    assert dict(events.url.params) == {
        "q": "Disponibile",
        "singleEvents": "true",
        "timeMin": START.isoformat(),
        "timeMax": END.isoformat(),
        "timeZone": "Europe/Tirane",
        "fields": EVENT_FIELDS,
        "maxResults": "250",
    }
    assert EVENT_FIELDS == "items(start,end,summary,status),nextPageToken"
    assert events.headers["authorization"] == "Bearer tok"
    assert freebusy.method == "POST"
    assert json.loads(freebusy.content) == {
        "timeMin": START.isoformat(),
        "timeMax": END.isoformat(),
        "timeZone": "Europe/Tirane",
        "items": [{"id": "cal-1"}],
    }


def test_only_marker_events_are_blocks_and_no_other_title_is_kept_or_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    events = _events(
        _timed(23, 9, 13, " disponibile "),
        _timed(23, 10, 11, PATIENT),  # the search also matches an appointment's notes
        _timed(26, 9, 13, "Disponibile", status="cancelled"),
        {"summary": "Disponibile", "start": {"date": "2026-10-27"}, "end": {"date": "2026-10-28"}},
    )
    view = _read(_reader([events], _busy((at(23, 10), at(23, 11))), []))
    assert view.blocks == (Interval(at(23, 9), at(23, 13)), Interval(at(27, 9), at(27, 18)))
    assert view.busy == (Interval(at(23, 10), at(23, 11)),)
    assert PATIENT not in repr(view)
    assert "Rossi" not in caplog.text


def test_every_page_of_events_is_read() -> None:
    first = _events(_timed(23, 9, 10, "Disponibile"), page="p2")
    second = _events(_timed(24, 9, 10, "Disponibile"))
    seen: list[httpx2.Request] = []
    view = _read(_reader([first, second], _busy(), seen))
    assert len(view.blocks) == 2
    assert seen[1].url.params["pageToken"] == "p2"


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
    ],
)
def test_a_calendar_that_cannot_be_read_raises_with_a_status_only(
    handler: Handler, message: str
) -> None:
    reader = GoogleCalendar(lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(handler)))
    with pytest.raises(CalendarUnavailableError, match=message):
        _read(reader)


def test_a_timeout_or_a_freebusy_error_cannot_be_read() -> None:
    def timeout(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    reader = GoogleCalendar(lambda: "tok", httpx2.Client(transport=httpx2.MockTransport(timeout)))
    with pytest.raises(CalendarUnavailableError, match="ReadTimeout"):
        _read(reader)
    refused = {"calendars": {"cal-1": {"errors": [{"domain": "global", "reason": "notFound"}]}}}
    with pytest.raises(CalendarUnavailableError, match="freeBusy: notFound"):
        _read(_reader([_events()], refused, []))


def test_a_token_that_cannot_be_had_cannot_read_the_calendar() -> None:
    def no_token() -> str:
        raise RuntimeError("no credentials")

    reader = GoogleCalendar(
        no_token, httpx2.Client(transport=httpx2.MockTransport(lambda _: httpx2.Response(200)))
    )
    with pytest.raises(CalendarUnavailableError, match="credentials: RuntimeError"):
        _read(reader)


def test_malformed_answers_cannot_be_read() -> None:
    no_end = {"items": [{"summary": "Disponibile", "start": {"dateTime": at(23, 9).isoformat()}}]}
    with pytest.raises(CalendarUnavailableError, match="malformed event: KeyError"):
        _read(_reader([no_end], _busy(), []))
    with pytest.raises(CalendarUnavailableError, match="freeBusy: no answer for the calendar"):
        _read(_reader([_events()], {"calendars": {}}, []))
    as_list = GoogleCalendar(
        lambda: "tok",
        httpx2.Client(transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=[]))),
    )
    with pytest.raises(CalendarUnavailableError, match="malformed response"):
        _read(as_list)


def test_endless_pages_stop_with_an_error() -> None:
    endless = [_events(page="again")] * (MAX_PAGES + 1)
    with pytest.raises(CalendarUnavailableError, match="too many pages"):
        _read(_reader(endless, _busy(), []))


def test_the_client_ignores_proxies_and_verifies_with_certifi() -> None:
    client = calendar_http_client()
    assert client.trust_env is False
    assert client.timeout.read == TIMEOUT_SECONDS
