"""Google Calendar as the calendar reader (ADR-0003, amended for M4).

Two reads, nothing else:
- the events whose text holds the marker word, with only their start, end, title and status;
- the busy intervals (freeBusy).
The search also matches an appointment that holds the word in its notes, so any title other than the
marker is dropped as soon as the response is read. No event content is logged, and an error carries
a status, never a response body.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timedelta
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
EVENT_FIELDS: Final = "items(start,end,summary,status),nextPageToken"
MAX_PAGES: Final = 10
TIMEOUT_SECONDS: Final = 10.0


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
        window = {"timeMin": start.isoformat(), "timeMax": end.isoformat(), "timeZone": zone.key}
        try:
            blocks = tuple(self._blocks(calendar_id, window, marker, opening, headers))
            busy = self._busy(calendar_id, window, headers)
        except (KeyError, TypeError, ValueError) as error:
            raise CalendarUnavailableError(f"malformed event: {type(error).__name__}") from None
        return CalendarView(blocks, busy)

    def _blocks(
        self,
        calendar_id: str,
        window: dict[str, str],
        marker: str,
        opening: Opening,
        headers: dict[str, str],
    ) -> Iterator[Interval]:
        url = f"{API}/calendars/{quote(calendar_id, safe='')}/events"
        params = {
            "q": marker,
            "singleEvents": "true",
            **window,
            "fields": EVENT_FIELDS,
            "maxResults": "250",
        }
        wanted = marker.strip().casefold()
        for item in self._pages(url, params, headers):
            title = str(item.get("summary", "")).strip().casefold()
            if item.get("status") != "cancelled" and title == wanted:
                yield from _intervals(item, opening)

    def _pages(
        self, url: str, params: dict[str, str], headers: dict[str, str]
    ) -> Iterator[dict[str, Any]]:
        page: str | None = None
        for _ in range(MAX_PAGES):
            data = self._send(
                "GET", url, headers, params={**params, **({"pageToken": page} if page else {})}
            )
            yield from data.get("items") or []
            page = data.get("nextPageToken")
            if not page:
                return
        raise CalendarUnavailableError("too many pages of events")

    def _busy(
        self, calendar_id: str, window: dict[str, str], headers: dict[str, str]
    ) -> tuple[Interval, ...]:
        body = {**window, "items": [{"id": calendar_id}]}
        data = self._send("POST", f"{API}/freeBusy", headers, json=body)
        entry = (data.get("calendars") or {}).get(calendar_id)
        if entry is None:
            raise CalendarUnavailableError("freeBusy: no answer for the calendar")
        if entry.get("errors"):
            raise CalendarUnavailableError(f"freeBusy: {entry['errors'][0].get('reason', 'error')}")
        return tuple(
            Interval(datetime.fromisoformat(item["start"]), datetime.fromisoformat(item["end"]))
            for item in entry.get("busy") or []
        )

    def _send(
        self, method: str, url: str, headers: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, url, headers=headers, **kwargs)
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


def _intervals(item: dict[str, Any], opening: Opening) -> list[Interval]:
    """A timed block as it is; an all-day one as the clinic's opening hours of each of its days."""
    start, end = item["start"], item["end"]
    if "dateTime" in start:
        return [
            Interval(
                datetime.fromisoformat(start["dateTime"]), datetime.fromisoformat(end["dateTime"])
            )
        ]
    first, last = date.fromisoformat(start["date"]), date.fromisoformat(end["date"])
    return [
        interval
        for offset in range((last - first).days)
        for interval in opening(first + timedelta(days=offset))
    ]
