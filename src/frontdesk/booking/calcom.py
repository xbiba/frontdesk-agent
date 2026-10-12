"""cal.com as the scheduler (Albi, 2026-10-12): it keeps each doctor's availability, checks the
clinic's Google Calendar for conflicts, and writes the appointments there.

Two calls: the free slots of the doctor's event type, and the booking, with the patient's name and
phone (cal.com's event type asks for a phone in international format; the email is optional). A
booking whose answer is lost or fails on the server is looked up by its reference before anything
is said. Errors carry a status, never a body; the API key never appears in one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx2

from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.slots import Slot
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Language

API: Final = "https://api.cal.com"
SLOTS_VERSION: Final = "2024-09-04"
BOOKINGS_VERSION: Final = "2026-02-25"
LIST_VERSION: Final = "2024-08-13"
_BOOKED: Final = frozenset({"accepted", "pending"})


class CalcomScheduler:
    def __init__(self, api_key: str, client: httpx2.Client, *, base_url: str = API) -> None:
        self._key = api_key
        self._client = client
        self._base = base_url.rstrip("/")

    def free_slots(
        self, pack: BusinessPack, doctor_id: str, service_id: str, now: datetime
    ) -> tuple[Slot, ...]:
        calendar = _calendar(pack)
        zone = ZoneInfo(pack.timezone)
        data = self._call(
            "GET",
            "/v2/slots",
            SLOTS_VERSION,
            params={
                "eventTypeId": calendar.calendars[doctor_id],
                "start": _utc(now),
                "end": _utc(now + timedelta(days=calendar.horizon_days)),
                "timeZone": pack.timezone,
                "format": "range",
            },
        )
        days = data.get("data")
        if not isinstance(days, dict):
            raise CalendarUnavailableError("malformed slots")
        try:
            slots = [
                Slot(
                    datetime.fromisoformat(item["start"]).astimezone(zone),
                    datetime.fromisoformat(item["end"]).astimezone(zone),
                )
                for day in days.values()
                for item in day
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise CalendarUnavailableError(f"malformed slot: {type(error).__name__}") from None
        return tuple(sorted(slots, key=lambda slot: slot.start))

    def book(
        self,
        pack: BusinessPack,
        doctor_id: str,
        service_id: str,
        start: datetime,
        end: datetime,
        *,
        name: str,
        phone: str,
        reference: str,
        language: Language,
    ) -> None:
        event_type = int(_calendar(pack).calendars[doctor_id])
        body = {
            "start": _utc(start),
            "eventTypeId": event_type,
            "attendee": {"name": name, "timeZone": pack.timezone, "phoneNumber": phone},
            "metadata": {"frontdesk": reference},
        }
        try:
            data = self._call("POST", "/v2/bookings", BOOKINGS_VERSION, json=body, unsure=True)
        except _UnsureError:
            self._confirm(event_type, start, end, reference)
            return
        booking = data.get("data")
        if not isinstance(booking, dict) or booking.get("status") not in _BOOKED:
            raise CalendarUnavailableError("the booking was not accepted")

    def _confirm(self, event_type: int, start: datetime, end: datetime, reference: str) -> None:
        """After a lost answer or a server error: the booking exists if cal.com lists it with our
        reference."""
        try:
            data = self._call(
                "GET",
                "/v2/bookings",
                LIST_VERSION,
                params={
                    "eventTypeId": event_type,
                    "afterStart": _utc(start - timedelta(minutes=1)),
                    "beforeEnd": _utc(end + timedelta(minutes=1)),
                },
            )
        except CalendarUnavailableError:
            raise CalendarUnavailableError("the appointment may not be booked") from None
        bookings = data.get("data")
        if not isinstance(bookings, list) or not any(
            isinstance(item, dict)
            and (item.get("metadata") or {}).get("frontdesk") == reference
            and item.get("status") in _BOOKED
            for item in bookings
        ):
            raise CalendarUnavailableError("the appointment may not be booked")

    def _call(
        self, method: str, path: str, version: str, *, unsure: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._key}", "cal-api-version": version}
        try:
            response = self._client.request(
                method, f"{self._base}{path}", headers=headers, **kwargs
            )
        except httpx2.HTTPError as error:  # timeouts and connection errors
            if unsure:
                raise _UnsureError from None
            raise CalendarUnavailableError(type(error).__name__) from None
        if response.status_code >= 500 and unsure:
            raise _UnsureError
        if response.status_code >= 400:
            raise CalendarUnavailableError(f"status {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise CalendarUnavailableError("malformed response") from None
        if not isinstance(data, dict):
            raise CalendarUnavailableError("malformed response")
        return data


class _UnsureError(Exception):
    """The request may have been carried out: its answer was lost, or the server failed."""


def _calendar(pack: BusinessPack) -> CalendarIntegration:
    if pack.integrations.calendar is None:
        raise ValueError(f"business '{pack.id}' has no calendar")
    return pack.integrations.calendar


def _utc(instant: datetime) -> str:
    return instant.astimezone(UTC).isoformat().replace("+00:00", "Z")
