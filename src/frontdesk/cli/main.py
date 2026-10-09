"""Command line. Every command acts on one deployment, selected by --env-file."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

import typer
from sqlalchemy.orm import Session

from frontdesk.cli.chat import run_chat
from frontdesk.composition import (
    Deployment,
    NoActiveKnowledgeError,
    build_turn_runtime,
    open_deployment,
)
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.db.engine import business_session
from frontdesk.db.migrate import upgrade_to_head
from frontdesk.knowledge.evaluation import (
    DatasetError,
    load_cases,
    render_markdown,
    run_retrieval_eval,
    to_json,
)
from frontdesk.knowledge.versions import (
    EmbedderMismatchError,
    KnowledgeStateError,
    VersionStatus,
    activate,
    approve,
    check_embedder,
    create_draft,
    reembed_active,
    review_report,
    rollback,
    version_status,
)
from frontdesk.settings import load_settings

DEMO_PREFIX: Final = "demo-"

app = typer.Typer(no_args_is_help=True, help="frontdesk: a messaging agent for small businesses.")
db_app = typer.Typer(no_args_is_help=True, help="Database migrations.")
knowledge_app = typer.Typer(no_args_is_help=True, help="Versioned, approved knowledge.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluations.")
app.add_typer(db_app, name="db")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(eval_app, name="eval")

EnvFile = Annotated[
    Path, typer.Option("--env-file", exists=True, dir_okay=False, help="The deployment's env file.")
]
VersionId = Annotated[uuid.UUID, typer.Option("--version", help="Knowledge version ID.")]
ReportPath = Annotated[
    Path | None, typer.Option("--output", dir_okay=False, help="Also write a JSON report here.")
]


@app.callback()
def _keep_output_clean() -> None:
    # Runs before every command, so before any agent does. pydantic-ai prints a first-run banner
    # on the first agent run; in a chat or demo transcript that is noise. A value the operator
    # has already set stays.
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def _deployment(env_file: Path) -> Deployment:
    return open_deployment(load_settings(env_file))


def _runtime(deployment: Deployment, env_file: Path) -> TurnRuntime:
    try:
        return build_turn_runtime(deployment)
    except NoActiveKnowledgeError as error:
        typer.echo(
            f"{error}. Draft, approve and activate a version first "
            f"(demo businesses: frontdesk knowledge bootstrap-demo --env-file {env_file}).",
            err=True,
        )
        raise typer.Exit(1) from error
    except EmbedderMismatchError as error:
        typer.echo(f"error: {error} --env-file {env_file}", err=True)
        raise typer.Exit(1) from error


@contextmanager
def _session(deployment: Deployment) -> Iterator[Session]:
    try:
        with business_session(deployment.session_factory, deployment.pack.id) as session:
            yield session
    except KnowledgeStateError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error


@db_app.command("upgrade")
def db_upgrade(env_file: EnvFile) -> None:
    """Apply all migrations to this deployment's database."""
    upgrade_to_head(load_settings(env_file).migration_database_url)
    typer.echo("database is at the latest migration")


@knowledge_app.command("draft")
def knowledge_draft(env_file: EnvFile) -> None:
    """Create a draft version from the business pack and print its review report."""
    deployment = _deployment(env_file)
    with _session(deployment) as session:
        draft = create_draft(session, deployment.pack)
        report = review_report(session, draft.version_id)
    label = "unchanged" if draft.unchanged else "draft"
    typer.echo(f"{label}: {draft.version_id}\n\n{report}")


@knowledge_app.command("report")
def knowledge_report(env_file: EnvFile, version: VersionId) -> None:
    """Print what a version changes compared with the active one."""
    deployment = _deployment(env_file)
    with _session(deployment) as session:
        typer.echo(review_report(session, version))


@knowledge_app.command("approve")
def knowledge_approve(
    env_file: EnvFile,
    version: VersionId,
    by: Annotated[str, typer.Option("--by", help="Who approved, as recorded.")],
    evidence: Annotated[str, typer.Option("--evidence", help="Proof of approval.")],
) -> None:
    """Record the business owner's approval of a draft."""
    deployment = _deployment(env_file)
    with _session(deployment) as session:
        approve(session, version, approved_by=by, evidence=evidence, now=datetime.now(UTC))
    typer.echo(f"approved: {version}")


@knowledge_app.command("activate")
def knowledge_activate(env_file: EnvFile, version: VersionId) -> None:
    """Embed and activate an approved version; the previous one is retired."""
    deployment = _deployment(env_file)
    with _session(deployment) as session:
        activate(session, version, deployment.embedder, now=datetime.now(UTC))
    typer.echo(f"active: {version}")


@knowledge_app.command("rollback")
def knowledge_rollback(env_file: EnvFile) -> None:
    """Re-activate the previous approved version."""
    deployment = _deployment(env_file)
    with _session(deployment) as session:
        restored = rollback(session, deployment.pack.id, deployment.embedder, now=datetime.now(UTC))
    typer.echo(f"active: {restored}")


@knowledge_app.command("reembed")
def knowledge_reembed(env_file: EnvFile) -> None:
    """Recompute the active version's vectors with this deployment's embedder."""
    deployment = _deployment(env_file)
    with _session(deployment) as session:
        count = reembed_active(session, deployment.pack.id, deployment.embedder)
    typer.echo(f"re-embedded {count} chunks with {deployment.embedder.identity}")


@knowledge_app.command("bootstrap-demo")
def knowledge_bootstrap_demo(env_file: EnvFile) -> None:
    """Draft, approve and activate a synthetic demo business in one step."""
    deployment = _deployment(env_file)
    if not deployment.pack.id.startswith(DEMO_PREFIX):
        typer.echo(
            "bootstrap-demo is only for demo businesses; real ones need a recorded approval",
            err=True,
        )
        raise typer.Exit(1)
    now = datetime.now(UTC)
    with _session(deployment) as session:
        version_id = create_draft(session, deployment.pack).version_id
        if version_status(session, version_id) is VersionStatus.DRAFT:
            approve(
                session,
                version_id,
                approved_by="demo bootstrap",
                evidence="synthetic data",
                now=now,
            )
        if version_status(session, version_id) is VersionStatus.APPROVED:
            activate(session, version_id, deployment.embedder, now=now)
    typer.echo(f"active: {version_id}")


@eval_app.command("retrieval")
def eval_retrieval(
    env_file: EnvFile,
    k: Annotated[int, typer.Option("--k", min=1, max=20)] = 5,
    output: ReportPath = None,
) -> None:
    """Run the business's labelled retrieval questions and print recall@k and MRR."""
    deployment = _deployment(env_file)
    try:
        cases = load_cases(deployment.settings.require_business_pack() / "evals" / "retrieval.yaml")
    except DatasetError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    with _session(deployment) as session:
        check_embedder(session, deployment.pack.id, deployment.embedder)
        report = run_retrieval_eval(
            session,
            business_id=deployment.pack.id,
            cases=cases,
            embedder=deployment.embedder,
            k=k,
            today=deployment.today(),
        )
    typer.echo(render_markdown(report))
    if output is not None:
        output.write_text(to_json(report), encoding="utf-8")


@app.command("chat")
def chat(
    env_file: EnvFile,
    trace: Annotated[bool, typer.Option("--trace/--no-trace")] = True,
) -> None:
    """Chat with this deployment's agent in the terminal."""
    deployment = _deployment(env_file)
    runtime = _runtime(deployment, env_file)
    typer.echo(f"{deployment.pack.name} ({deployment.pack.profile}) — type 'exit' to quit")
    asyncio.run(run_chat(runtime, read_line=_read_line, write=typer.echo, show_trace=trace))


def _read_line() -> str | None:
    line = sys.stdin.readline()
    return line or None
