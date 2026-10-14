"""`frontdesk instagram ...` (M4): the ID the account's webhooks carry, its subscription to the
message webhooks, and the testers."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx2
import typer

from frontdesk.channels.inbox import Inbox
from frontdesk.channels.instagram import INSTAGRAM, account_user_id, subscribe
from frontdesk.channels.wiring import channel_secret
from frontdesk.db.engine import make_engine, make_session_factory
from frontdesk.settings import Settings
from frontdesk.tenancy.loader import load_pack


async def show_account(settings: Settings, *, client: httpx2.AsyncClient | None = None) -> int:
    token = channel_secret(settings, "instagram_token")
    async with client or httpx2.AsyncClient(timeout=10.0) as http:
        user_id, username = await account_user_id(http, token, settings.instagram_api_version)
    typer.echo(f"account id: {user_id}\nusername: {username}")
    return 0


async def subscribe_account(settings: Settings, *, client: httpx2.AsyncClient | None = None) -> int:
    token = channel_secret(settings, "instagram_token")
    async with client or httpx2.AsyncClient(timeout=10.0) as http:
        await subscribe(http, token, settings.instagram_api_version)
    typer.echo("the account is subscribed to messages")
    return 0


def show_testers(settings: Settings, *, clear: bool) -> int:
    pack = load_pack(settings.require_business_pack())
    inbox = Inbox(
        make_session_factory(make_engine(settings.database_url)),
        business_id=pack.id,
        channel=INSTAGRAM,
        account_id=settings.instagram_account_id or "",
        testers_only=True,
        pairing_code=None,
        now=lambda: datetime.now(UTC),
    )
    if clear:
        typer.echo(f"testers removed: {inbox.clear_testers()}")
    else:
        typer.echo(f"testers: {inbox.count_testers()}")
    return 0
