"""`frontdesk whatsapp ...` (W1): the booking request template, sent to Meta's approval as soon as
WhatsApp is added to the app (Albi, 2026-10-17)."""

from __future__ import annotations

import httpx2
import typer

from frontdesk.channels.templates import BOOKING_REQUEST, booking_request_definition
from frontdesk.channels.whatsapp import GRAPH
from frontdesk.channels.wiring import channel_secret
from frontdesk.settings import Settings


async def send_template(settings: Settings, *, client: httpx2.AsyncClient | None = None) -> int:
    """Send the template in the staff's language; print Meta's status, or its error code."""
    account = settings.whatsapp_business_account_id
    if not account:
        typer.echo("error: FRONTDESK_WHATSAPP_BUSINESS_ACCOUNT_ID is not set", err=True)
        return 1
    language = settings.whatsapp_staff_language
    token = channel_secret(settings, "whatsapp_token")
    url = f"{GRAPH}/{settings.whatsapp_api_version}/{account}/message_templates"
    async with client or httpx2.AsyncClient(timeout=15.0) as http:
        response = await http.post(
            url,
            json=booking_request_definition(language),
            headers={"Authorization": f"Bearer {token}"},
        )
    answer = _json(response)
    if response.status_code != 200:
        error = answer.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        typer.echo(
            f"error: Meta refused the template: status {response.status_code}, code {code}",
            err=True,
        )
        return 1
    typer.echo(f"template {BOOKING_REQUEST} ({language}): {answer.get('status', 'sent')}")
    return 0


def _json(response: httpx2.Response) -> dict[str, object]:
    try:
        answer = response.json()
    except ValueError:
        return {}
    return answer if isinstance(answer, dict) else {}
