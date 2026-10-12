"""`frontdesk bookings purge`: deletes the expired holds and the bookings whose slot has passed.

Each booking operation already deletes the expired holds of its business; this command does it on
a schedule too, so a name or a phone left by a booking that nobody finished is gone within its
hold's lifetime even when no one books for a while (M5 retention).
"""

from __future__ import annotations

from datetime import datetime

import typer

from frontdesk.booking.holds import purge
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.settings import Settings
from frontdesk.tenancy.loader import load_pack


def purge_bookings(settings: Settings, now: datetime) -> int:
    pack = load_pack(settings.require_business_pack())
    factory = make_session_factory(make_engine(settings.database_url))
    with business_session(factory, pack.id) as session:
        deleted = purge(session, now)
    typer.echo(f"booking holds deleted: {deleted}")
    return 0
