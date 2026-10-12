"""`frontdesk calendar check`: read a doctor's calendar as the bot does (M4 setup).

It prints counts and free times only, never an event's content, and shows whether the calendar's
sharing and the service account work before any patient asks.
"""

from __future__ import annotations

from datetime import datetime

import typer

from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.reply import read_calendar, slots_in
from frontdesk.composition import build_calendar
from frontdesk.settings import Settings
from frontdesk.tenancy.loader import load_pack


def check_calendar(settings: Settings, doctor: str, now: datetime) -> int:
    """0 when the calendar can be read, 1 when the deployment cannot read it, 2 when Google refuses."""
    pack = load_pack(settings.require_business_pack())
    services = pack.bookable_doctors().get(doctor)
    if pack.integrations.calendar is None or not services:
        typer.echo(
            f"error: '{doctor}' is not a bookable doctor with a calendar in the pack", err=True
        )
        return 1
    reader = build_calendar(settings)
    if reader is None:
        typer.echo("error: FRONTDESK_CALENDAR_BACKEND is 'none' for this deployment", err=True)
        return 1
    calendar = pack.integrations.calendar
    try:
        view = read_calendar(reader, pack, doctor, now)
    except CalendarUnavailableError as error:
        typer.echo(f"calendar: cannot be read ({error})", err=True)
        return 2
    typer.echo(
        f"calendar of {doctor}: {len(view.blocks)} '{calendar.availability_marker}' blocks and "
        f"{len(view.busy)} busy intervals in the next {calendar.horizon_days} days"
    )
    slots = slots_in(view, pack, services[0], now)
    for slot in slots:
        typer.echo(f"free: {slot.start:%Y-%m-%d %H:%M} ({services[0]})")
    if not slots:
        typer.echo("free: none")
    return 0
