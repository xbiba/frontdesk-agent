"""Google Calendar as the calendar reader (ADR-0003, amended for M4).

Two listings of the doctor's calendar, nothing else:
- the blocks: the events whose text holds the marker word, with their id, start, end, title,
  status, type and whether this calendar organises them. Only an ordinary event organised by this
  calendar and titled exactly with the marker is a block: an invitation from outside cannot open
  hours. The search also matches an appointment that holds the word in its notes, so any other
  title is dropped as soon as the response is read;
- the busy time: every other event, with only its id, start, end, status and type. No title or
  note is requested. Every event that is not a block is busy, whatever its "Show as": an appointment
  marked as free by mistake still takes its time (security review of 2026-10-12).
No event content is logged, and an error carries a status, never a response body. Ranges are
clipped to the window read, and at most MAX_PAGES pages are followed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, time, timedelta
from typing import Any, Final
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx2

from frontdesk.booking.calendar import CalendarUnavailableError, CalendarView, Opening
from frontdesk.booking.slots import Interval
from frontdesk.model_gateway.factory import tls_context
from frontdesk.model_gateway.google_tokens import TokenSource

API: Final = "https://www.googleapis.com/calendar/v3"
SCOPE: Final = "https://www.googleapis.com/auth/calendar.readonly"
BLOCK_FIELDS: Final = "items(id,start,end,summary,status,eventType,organizer(self)),nextPageToken"
BUSY_FIELDS: Final = "items(id,start,end,status,eventType),nextPageToken"
MAX_PAGES: Final = 10
TIMEOUT_SECONDS: Final = 10.0
_NOT_BUSY: Final = frozenset({"workingLocation"})  # where someone works, not an appointment


def calendar_http_client() -> httpx2.Client:
    """No proxy from the environment and certifi's certificates, as the model gateway's clients."""
    return httpx2.Client(
        trust_env=False,
        verify=tls_context(),
        timeout=httpx2.Timeout(TIMEOUT_SECONDS, connect=5.0),
    )


class GoogleCalendar:
    def __init__(self, token: TokenSource, client: httpx2.Client) -> None:
        self._token = token
        self._client = client

    def read(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        marker: str,
        zone: ZoneInfo,
        opening: Opening,
    ) -> CalendarView:
        try:
            headers = {"Authorization": f"Bearer {self._token()}"}
        except Exception as error:  # any credential failure; only its type is kept
            raise CalendarUnavailableError(f"credentials: {type(error).__name__}") from None
        url = f"{API}/calendars/{quote(calendar_id, safe='')}/events"
        window = {
            "singleEvents": "true",
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": zone.key,
            "maxResults": "250",
        }
        days = (start.astimezone(zone).date(), end.astimezone(zone).date())
        try:
            found = self._pages(url, {**window, "q": marker, "fields": BLOCK_FIELDS}, headers)
            block_items = [item for item in found if _is_block(item, marker)]
            blocks = tuple(
                interval
                for item in block_items
                for interval in _block_intervals(item, opening, days)
            )
            own = {item["id"] for item in block_items}
            busy = tuple(
                interval
                for item in self._pages(url, {**window, "fields": BUSY_FIELDS}, headers)
                if _is_busy(item, own)
                for interval in _busy_intervals(item, zone, days)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CalendarUnavailableError(f"malformed event: {type(error).__name__}") from None
        return CalendarView(blocks, busy)

    def _pages(
        self, url: str, params: dict[str, str], headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page: str | None = None
        for _ in range(MAX_PAGES):
            token = {"pageToken": page} if page else {}
            data = self._send(url, headers, params={**params, **token})
            items += data.get("items") or []
            page = data.get("nextPageToken")
            if not page:
                return items
        raise CalendarUnavailableError("too many pages of events")

    def _send(self, url: str, headers: dict[str, str], **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.get(url, headers=headers, **kwargs)
        except httpx2.HTTPError as error:  # timeouts and connection errors
            raise CalendarUnavailableError(type(error).__name__) from None
        if response.status_code >= 400:
            raise CalendarUnavailableError(f"status {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise CalendarUnavailableError("malformed response") from None
        if not isinstance(data, dict):
            raise CalendarUnavailableError("malformed response")
        return data


def _is_block(item: dict[str, Any], marker: str) -> bool:
    title = str(item.get("summary", "")).strip().casefold()
    return (
        item.get("status") != "cancelled"
        and item.get("eventType", "default") == "default"
        and (item.get("organizer") or {}).get("self") is True
        and title == marker.strip().casefold()
    )


def _is_busy(item: dict[str, Any], own: set[str]) -> bool:
    return (
        item.get("status") != "cancelled"
        and item.get("eventType", "default") not in _NOT_BUSY
        and item.get("id") not in own
    )


def _block_intervals(
    item: dict[str, Any], opening: Opening, days: tuple[date, date]
) -> list[Interval]:
    """A timed block as it is; an all-day one as the clinic's opening hours of each of its days."""
    if "dateTime" in item["start"]:
        return [_timed(item)]
    return [interval for day in _days(item, days) for interval in opening(day)]


def _busy_intervals(
    item: dict[str, Any], zone: ZoneInfo, days: tuple[date, date]
) -> list[Interval]:
    """A timed event as it is; an all-day one takes each of its whole days."""
    if "dateTime" in item["start"]:
        return [_timed(item)]
    return [
        Interval(_midnight(day, zone), _midnight(day + timedelta(days=1), zone))
        for day in _days(item, days)
    ]


def _timed(item: dict[str, Any]) -> Interval:
    return Interval(
        datetime.fromisoformat(item["start"]["dateTime"]),
        datetime.fromisoformat(item["end"]["dateTime"]),
    )


def _days(item: dict[str, Any], days: tuple[date, date]) -> Iterator[date]:
    """The days of an all-day event (its end date is exclusive), within the window read."""
    first = max(date.fromisoformat(item["start"]["date"]), days[0])
    last = min(date.fromisoformat(item["end"]["date"]), days[1] + timedelta(days=1))
    for offset in range((last - first).days):
        yield first + timedelta(days=offset)


def _midnight(day: date, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time(0), tzinfo=zone)
