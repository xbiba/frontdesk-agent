"""`frontdesk email`: one pass of the email drafts (for a Cloud Run job that Cloud Scheduler
starts), its dry run, and the login that writes a development token file.

The pass needs the pack's `email:` section, a route approved for patient-derived text (the
deployment's FRONTDESK_MODEL_ROUTE), the mailbox (FRONTDESK_EMAIL_BACKEND) and the secret the
senders' hashes are keyed with (frontdesk-email-sender-key). The daily cap of the deployment holds
for every model request. Nothing it runs can send a message.
"""

from __future__ import annotations

import asyncio
import http.server
import threading
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import typer

from frontdesk.agent.prompts import read_prompt
from frontdesk.cli.runtime import fail
from frontdesk.composition import (
    Deployment,
    daily_capped,
    model_api_key,
    open_deployment,
)
from frontdesk.email.drafter import MAX_OUTPUT_TOKENS, Drafter
from frontdesk.email.gmail import SCOPES, GmailMailbox, gmail_http_client
from frontdesk.email.job import EmailPass, Outcome, PassResult
from frontdesk.email.ledger import MIN_SECRET_BYTES, PostgresLedger
from frontdesk.email.mailbox import FileMailbox, Mailbox
from frontdesk.model_gateway.costs import load_budget
from frontdesk.model_gateway.daily_cap import SpendCapReachedError
from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.google_tokens import google_token_source
from frontdesk.model_gateway.mail_tokens import (
    MailTokenError,
    delegated_token_source,
    exchange_code,
    login_url,
    read_client,
    user_token_source,
)
from frontdesk.model_gateway.routes import (
    CHAT_PROVIDERS,
    DataClass,
    UnapprovedRouteError,
    require_route,
)
from frontdesk.secrets import CLOUD_SCOPE, read_secret
from frontdesk.settings import EmailBackend, Settings, load_settings
from frontdesk.tenancy.pack import EmailChannel


def email_channel(deployment: Deployment) -> EmailChannel:
    if deployment.pack.email is None:
        raise fail(f"the pack of {deployment.pack.id} has no email section (email:)")
    return deployment.pack.email


def build_mailbox(settings: Settings, email: EmailChannel) -> Mailbox:
    """The mailbox of FRONTDESK_EMAIL_BACKEND."""
    if settings.email_backend is EmailBackend.FILE:
        if settings.email_file is None:
            raise fail("FRONTDESK_EMAIL_FILE is not set for the file mailbox")
        return FileMailbox.from_file(
            settings.email_file, out=settings.email_file.with_suffix(".out.yaml")
        )
    if settings.email_backend is EmailBackend.GMAIL:
        client = gmail_http_client()
        if settings.email_delegation_account:
            token = delegated_token_source(
                SCOPES,
                service_account=settings.email_delegation_account,
                subject=email.mailbox,
                allowed_subjects=frozenset({email.mailbox}),
                signer_token=google_token_source([CLOUD_SCOPE], impersonate=None),
                client=client,
            )
        elif settings.email_token_file is not None:
            token = user_token_source(settings.email_token_file, client=client)
        else:
            raise fail(
                "Gmail needs FRONTDESK_EMAIL_DELEGATION_ACCOUNT (domain-wide delegation) or "
                "FRONTDESK_EMAIL_TOKEN_FILE (development: `frontdesk email login`)"
            )
        return GmailMailbox(token, client)
    raise fail("FRONTDESK_EMAIL_BACKEND is not set: gmail or file")


def sender_secret(settings: Settings) -> bytes:
    """The secret the senders' hashes are keyed with; named, never shown, when missing."""
    found = read_secret(
        settings,
        "email_sender_key",
        token=google_token_source([CLOUD_SCOPE], impersonate=None),
        client=gmail_http_client(),
    )
    if not found or len(found.encode()) < MIN_SECRET_BYTES:
        raise fail(
            f"the secret frontdesk-email-sender-key (FRONTDESK_EMAIL_SENDER_KEY in development) "
            f"is missing or shorter than {MIN_SECRET_BYTES} bytes"
        )
    return found.encode()


def build_pass(deployment: Deployment, *, dry_run: bool) -> EmailPass:
    """The pass of this deployment: its mailbox, its ledger and, unless a dry run, its model under
    the daily cap."""
    settings = deployment.settings
    email = email_channel(deployment)
    mailbox = build_mailbox(settings, email)
    drafter, price = None, None
    if not dry_run:
        try:
            route = require_route(settings.model_route, DataClass.PATIENT_DERIVED)
        except UnapprovedRouteError as error:
            raise fail(str(error)) from error
        if route.provider not in CHAT_PROVIDERS:
            raise fail(f"route '{route.route_id}' calls no live model: the drafts need one")
        key = model_api_key(settings, route) if route.provider == "anthropic" else None
        model = build_live_model(route, gcp_project=settings.gcp_project, api_key=key)
        wrap = daily_capped(deployment, route)
        if wrap is not None:
            model = wrap(model, "agent")
        if settings.prices_file is not None:
            price = load_budget(settings.prices_file).price(route.route_id)
        drafter = Drafter(
            model,
            deployment.pack,
            email,
            prompt=read_prompt(deployment.pack.profile, "email"),
            today=deployment.today(),
            model_settings=live_model_settings(route, max_tokens=MAX_OUTPUT_TOKENS),
        )
    return EmailPass(
        mailbox=mailbox,
        ledger=PostgresLedger(deployment.session_factory, deployment.pack.id),
        drafter=drafter,
        pack=deployment.pack,
        now=deployment.now,
        price=price,
        stops=(SpendCapReachedError,),
        dry_run=dry_run,
        sender_secret=sender_secret(settings),
    )


def run_pass(env_file: Path, *, dry_run: bool) -> int:
    settings = load_settings(env_file)
    if settings.email_off and not dry_run:
        typer.echo("email drafts are off (FRONTDESK_EMAIL_OFF)")
        return 0
    deployment = open_deployment(settings)
    result = asyncio.run(build_pass(deployment, dry_run=dry_run).run())
    report(result, dry_run=dry_run)
    return 0 if result.stopped is None else 1


def report(result: PassResult, *, dry_run: bool) -> None:
    """One line a thread (its ID, the outcome, the flags), then the counts: no text, no address."""
    for item in result.threads:
        flags = f" [{', '.join(item.flags)}]" if item.flags else ""
        typer.echo(f"{item.thread_id} {item.outcome.value}{flags}")
    counts = {outcome.value: result.count(outcome) for outcome in Outcome if result.count(outcome)}
    cost = sum(item.usd for item in result.threads)
    head = "dry run: " if dry_run else ""
    typer.echo(
        f"{head}{len(result.threads)} threads {counts}, {cost:.4f} USD, purged {result.purged}"
    )
    if result.stopped is not None:
        typer.echo(f"error: the pass stopped: {result.stopped}", err=True)


def login(
    client_file: Path, token_file: Path, *, port: int = 0
) -> int:  # pragma: no cover - a browser
    """Open Google's consent page for gmail.modify, receive its answer on this Mac, write the token
    file. For the developer's own Gmail with an app in "Testing" (its tokens last 7 days)."""
    try:
        client_id, secret = read_client(client_file)
    except MailTokenError as error:
        raise fail(str(error)) from error
    answer: dict[str, str] = {}

    class Answer(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)
            answer.update({key: values[0] for key, values in query.items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"frontdesk: login received, you can close this page.")

        def log_message(self, *args: object) -> None:  # the code must not reach a log
            return

    server = http.server.HTTPServer(("127.0.0.1", port), Answer)
    pending = login_url(client_id, secret, SCOPES, f"http://127.0.0.1:{server.server_port}/")
    typer.echo(f"open this page and allow access:\n{pending.url}")
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    thread.join(timeout=300)
    server.server_close()
    if "code" not in answer:
        raise fail("no answer from Google's page within 5 minutes")
    try:
        exchange_code(
            pending, answer["code"], answer.get("state", ""), token_file, gmail_http_client()
        )
    except MailTokenError as error:
        raise fail(str(error)) from error
    typer.echo(f"token written to {token_file} ({datetime.now(UTC):%Y-%m-%d}; it lasts 7 days)")
    return 0
