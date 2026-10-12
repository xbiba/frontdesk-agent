"""`frontdesk calendar check`: read a doctor's free slots as the bot does (M4, M5 setup).

It prints counts and free times only, never an event's content, and shows whether the calendar's
sharing, the service account or the cal.com key work before any patient asks.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import typer

from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.slots import spread
from frontdesk.composition import build_scheduler
from frontdesk.settings import Settings
from frontdesk.tenancy.loader import load_pack


def check_calendar(settings: Settings, doctor: str, now: datetime) -> int:
    """0 when the free slots can be read, 1 when the deployment cannot read them, 2 when the
    calendar or cal.com refuses."""
    pack = load_pack(settings.require_business_pack())
    services = pack.bookable_doctors().get(doctor)
    calendar = pack.integrations.calendar
    if calendar is None or not services:
        typer.echo(
            f"error: '{doctor}' is not a bookable doctor with a calendar in the pack", err=True
        )
        return 1
    scheduler = build_scheduler(settings)
    if scheduler is None:
        typer.echo("error: FRONTDESK_CALENDAR_BACKEND is 'none' for this deployment", err=True)
        return 1
    try:
        slots = scheduler.free_slots(pack, doctor, services[0], now)
    except CalendarUnavailableError as error:
        typer.echo(f"calendar: cannot be read ({error})", err=True)
        return 2
    typer.echo(f"free slots of {doctor} in the next {calendar.horizon_days} days: {len(slots)}")
    for slot in spread(slots, ZoneInfo(pack.timezone), calendar.max_options):
        typer.echo(f"free: {slot.start:%Y-%m-%d %H:%M} ({services[0]})")
    return 0
