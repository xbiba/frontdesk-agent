"""`frontdesk instagram` (M4): the account's webhook ID, its subscription, the testers."""

from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from frontdesk.channels.inbox import Inbox
from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.cli.instagram import show_account, show_testers, subscribe_account
from frontdesk.cli.main import app
from frontdesk.db.engine import make_engine, make_session_factory
from frontdesk.settings import Settings

ACCOUNT = "17841400000000001"
NOW = datetime(2026, 10, 14, 9, tzinfo=UTC)


def test_the_instagram_commands_are_listed() -> None:
    result = CliRunner().invoke(app, ["instagram", "--help"])
    assert result.exit_code == 0
    for command in ("account", "subscribe", "testers"):
        assert command in result.output


async def test_the_account_command_prints_the_webhook_id(
    businesses_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"user_id": ACCOUNT, "username": "mcu_test"})

    settings = Settings(
        business_pack=businesses_dir / "demo-clinic", instagram_token=SecretStr("t")
    )
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    assert await show_account(settings, client=client) == 0
    assert capsys.readouterr().out == f"account id: {ACCOUNT}\nusername: mcu_test\n"


async def test_the_subscribe_command_subscribes_the_account(
    businesses_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"success": True})

    settings = Settings(
        business_pack=businesses_dir / "demo-clinic", instagram_token=SecretStr("t")
    )
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    assert await subscribe_account(settings, client=client) == 0
    assert capsys.readouterr().out == "the account is subscribed to messages\n"


@pytest.mark.db
def test_testers_are_counted_and_cleared(
    app_url: str, businesses_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inbox = Inbox(
        make_session_factory(make_engine(app_url)),
        business_id="demo-clinic",
        channel="instagram",
        account_id=ACCOUNT,
        testers_only=True,
        pairing_code="mcu-prova-1234",
        now=lambda: NOW,
    )
    inbox.accept([Inbound(ACCOUNT, "1111", InboundKind.TEXT, "pair", "mcu-prova-1234", NOW)])
    settings = Settings(business_pack=businesses_dir / "demo-clinic", database_url=app_url)
    assert show_testers(settings, clear=False) == 0
    assert capsys.readouterr().out == "testers: 1\n"
    assert show_testers(settings, clear=True) == 0
    assert capsys.readouterr().out == "testers removed: 1\n"
