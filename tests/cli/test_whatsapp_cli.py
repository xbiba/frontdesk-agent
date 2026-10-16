"""`frontdesk whatsapp` (W1): the booking request template, sent to Meta's approval."""

import json

import httpx2
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from frontdesk.channels.templates import booking_request_definition
from frontdesk.cli.main import app
from frontdesk.cli.whatsapp import send_template
from frontdesk.settings import Settings

WABA = "102938475610"


def _settings(**fields: object) -> Settings:
    return Settings(
        whatsapp_token=SecretStr("t0ken"),
        whatsapp_business_account_id=WABA,
        **fields,  # type: ignore[arg-type]
    )


def test_the_whatsapp_commands_are_listed() -> None:
    result = CliRunner().invoke(app, ["whatsapp", "--help"])
    assert result.exit_code == 0
    assert "template" in result.output


async def test_template_sends_the_definition_and_prints_meta_s_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"id": "8899", "status": "PENDING", "category": "UTILITY"})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    assert await send_template(_settings(), client=client) == 0
    (request,) = seen
    assert str(request.url) == f"https://graph.facebook.com/v25.0/{WABA}/message_templates"
    assert request.headers["authorization"] == "Bearer t0ken"
    assert json.loads(request.content) == booking_request_definition("it")
    output = capsys.readouterr().out
    assert output == "template frontdesk_booking_request (it): PENDING\n"
    assert "t0ken" not in output


async def test_the_template_goes_in_the_staff_language(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"id": "8899", "status": "PENDING"})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    assert await send_template(_settings(whatsapp_staff_language="sq"), client=client) == 0
    assert json.loads(seen[0].content) == booking_request_definition("sq")


async def test_a_refused_template_prints_meta_s_code_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        error = {"message": "Invalid parameter", "type": "OAuthException", "code": 100}
        return httpx2.Response(400, json={"error": error})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    assert await send_template(_settings(), client=client) == 1
    output = capsys.readouterr()
    assert "status 400, code 100" in output.err
    assert "t0ken" not in output.out + output.err
