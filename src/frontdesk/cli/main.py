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

from frontdesk.cli.calendar import check_calendar
from frontdesk.cli.chat import run_chat
from frontdesk.cli.experiment import eval_embedders
from frontdesk.cli.live import eval_report, live_turns, read_cases
from frontdesk.cli.live_chat import run_live_chat
from frontdesk.cli.runtime import build_runtime, without_knowledge
from frontdesk.composition import Deployment, NoActiveKnowledgeError, open_deployment
from frontdesk.conversation.evaluation import (
    render_turn_markdown,
    run_turn_eval,
    turn_report_json,
)
from frontdesk.db.engine import business_session
from frontdesk.db.migrate import upgrade_to_head
from frontdesk.knowledge.evaluation import (
    DatasetError,
    load_cases,
    render_markdown,
    run_retrieval_eval,
    to_json,
    unknown_relevant_documents,
)
from frontdesk.knowledge.versions import (
    KnowledgeStateError,
    VersionStatus,
    activate,
    active_version_id,
    approve,
    check_embedder,
    create_draft,
    reembed_active,
    review_report,
    rollback,
    version_status,
)
from frontdesk.model_gateway.routes import ROUTES
from frontdesk.settings import load_settings
from frontdesk.tenancy.pack import BusinessPack

DEMO_PREFIX: Final = "demo-"
DEMO_ISOLATION_LEVEL: Final = "T1"  # synthetic data only (ADR-0008); real clinics declare T3

# An uncaught error prints a traceback. With local variables in it, the frames that hold the
# Deployment would print the Settings: the database URL with its password. Typer's default is
# False; it is set here so that no change of that default can bring the locals back.
app = typer.Typer(
    no_args_is_help=True,
    help="frontdesk: a messaging agent for small businesses.",
    pretty_exceptions_show_locals=False,
)
db_app = typer.Typer(no_args_is_help=True, help="Database migrations.")
knowledge_app = typer.Typer(no_args_is_help=True, help="Versioned, approved knowledge.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluations.")
calendar_app = typer.Typer(no_args_is_help=True, help="The doctors' calendars.")
app.add_typer(db_app, name="db")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(eval_app, name="eval")
app.add_typer(calendar_app, name="calendar")

EnvFile = Annotated[
    Path, typer.Option("--env-file", exists=True, dir_okay=False, help="The deployment's env file.")
]
VersionId = Annotated[uuid.UUID, typer.Option("--version", help="Knowledge version ID.")]
ReportPath = Annotated[
    Path | None, typer.Option("--output", dir_okay=False, help="Also write a JSON report here.")
]
CasesPath = Annotated[
    Path | None,
    typer.Option(
        "--cases", dir_okay=False, help="The cases file (default: evals/ inside the business pack)."
    ),
]
TagOption = Annotated[
    list[str] | None, typer.Option("--tag", help="Only the cases with this tag (repeatable).")
]
LiveOption = Annotated[
    bool, typer.Option("--live", help="Run on a live model route: each call costs money.")
]
RouteOption = Annotated[
    str | None, typer.Option("--route", help="The live route to run on (with --live).")
]
RunDirOption = Annotated[
    Path | None,
    typer.Option("--run-dir", file_okay=False, help="Where a live run keeps its records."),
]
BudgetOption = Annotated[
    Path | None,
    typer.Option("--budget", exists=True, dir_okay=False, help="The phase's budget file."),
]
LimitOption = Annotated[
    int | None, typer.Option("--limit", min=1, help="Run at most this many cases now.")
]
YesOption = Annotated[bool, typer.Option("--yes", help="Start a live run without asking.")]


@app.callback()
def _keep_output_clean() -> None:
    # Runs before every command, so before any agent does. pydantic-ai prints a first-run banner
    # on the first agent run; in a chat or demo transcript that is noise. A value the operator
    # has already set stays.
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def _deployment(env_file: Path) -> Deployment:
    return open_deployment(load_settings(env_file))


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


def _why_not_a_demo(pack: BusinessPack) -> str | None:
    """Why `bootstrap-demo` may not approve this pack by itself, or None if it may.

    A demo business has both an id that starts with "demo-" and the isolation level T1, which is
    for synthetic data. A name alone proves nothing: a pack called "demo-..." that declares T2 or
    T3 holds real data, and its knowledge needs a recorded approval like any other.
    """
    if not pack.id.startswith(DEMO_PREFIX):
        return f"the id '{pack.id}' does not start with '{DEMO_PREFIX}'"
    if pack.isolation_level != DEMO_ISOLATION_LEVEL:
        return (
            f"'{pack.id}' declares isolation level {pack.isolation_level}, "
            f"and demo businesses declare {DEMO_ISOLATION_LEVEL}"
        )
    return None


@knowledge_app.command("bootstrap-demo")
def knowledge_bootstrap_demo(env_file: EnvFile) -> None:
    """Draft, approve and activate a synthetic demo business in one step."""
    deployment = _deployment(env_file)
    reason = _why_not_a_demo(deployment.pack)
    if reason is not None:
        typer.echo(
            f"bootstrap-demo is only for demo businesses: {reason}. A real business needs a "
            "recorded approval: frontdesk knowledge draft, approve and activate "
            f"--env-file {env_file}.",
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
    cases: CasesPath = None,
    min_similarity: Annotated[
        float | None,
        typer.Option(
            "--min-similarity",
            min=0.0,
            max=1.0,
            help="The vector side's relevance floor (default: FRONTDESK_MIN_RETRIEVAL_SIMILARITY).",
        ),
    ] = None,
) -> None:
    """Run the business's labelled retrieval questions and print recall@k and MRR.

    Exits 1, without a report, when the business has no active knowledge or a case names a
    document that the active version lacks: both would score zeros that mean nothing.
    """
    deployment = _deployment(env_file)
    dataset = cases or deployment.settings.require_business_pack() / "evals" / "retrieval.yaml"
    floor = (
        min_similarity
        if min_similarity is not None
        else deployment.settings.min_retrieval_similarity
    )
    try:
        retrieval_cases = load_cases(dataset)
    except DatasetError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    with _session(deployment) as session:
        check_embedder(session, deployment.pack.id, deployment.embedder)
        if active_version_id(session, deployment.pack.id) is None:
            raise without_knowledge(NoActiveKnowledgeError(deployment.pack.id), env_file)
        unknown = unknown_relevant_documents(
            session, business_id=deployment.pack.id, cases=retrieval_cases
        )
        if unknown:
            for case_id, key in unknown:
                typer.echo(
                    f"{dataset}: case '{case_id}': '{key}' is not a document of the active "
                    "knowledge version (a document key is its file name without .md)",
                    err=True,
                )
            raise typer.Exit(1)
        report = run_retrieval_eval(
            session,
            business_id=deployment.pack.id,
            cases=retrieval_cases,
            embedder=deployment.embedder,
            k=k,
            today=deployment.today(),
            min_similarity=floor,
        )
    typer.echo(render_markdown(report))
    if output is not None:
        output.write_text(to_json(report), encoding="utf-8")


@eval_app.command("turns")
def eval_turns(
    env_file: EnvFile,
    output: ReportPath = None,
    cases: CasesPath = None,
    tag: TagOption = None,
    live: LiveOption = False,
    route: RouteOption = None,
    run_dir: RunDirOption = None,
    budget: BudgetOption = None,
    limit: LimitOption = None,
    yes: YesOption = False,
) -> None:
    """Run the business's labelled conversation turns; exit 1 if any case fails.

    Offline by default, and free: a route that calls a model is refused without --live. With
    --live the run records every turn in --run-dir, resumes there after a stop, and writes the
    report and the review page (`frontdesk eval report` recomputes them with the labels).

    A live run exits 0 when it is done, 1 on an error (a budget another run holds included), 3 when
    the rate limit stopped it, 4 when the budget has no room, 5 when the model keeps failing, and
    130 on Ctrl-C. After a stop, the same command resumes the run.
    """
    tags = tag or []
    if live:
        if route is None or run_dir is None:
            raise _usage("--live needs --route and --run-dir")
        if output is not None:
            raise _usage("--output is for offline runs: a live run writes report.json in --run-dir")
        live_turns(
            env_file,
            route_id=route,
            cases_path=cases,
            run_dir=run_dir,
            budget_path=budget,
            limit=limit,
            yes=yes,
            tags=tags,
        )
        return
    if route is not None or run_dir is not None or budget is not None or limit is not None:
        raise _usage("--route, --run-dir, --budget and --limit are for --live runs")
    _offline_turns(env_file, output, cases, tags)


def _offline_turns(
    env_file: Path, output: Path | None, cases: Path | None, tags: list[str]
) -> None:
    """The free run: the deployment's own offline route, in memory, nothing recorded."""
    deployment = _deployment(env_file)
    configured = ROUTES.get(deployment.settings.model_route)
    if configured is not None and configured.provider != "offline":
        raise _usage(f"route '{configured.route_id}' calls a live model: add --live and a --budget")
    dataset = cases or deployment.settings.require_business_pack() / "evals" / "turns.yaml"
    turn_cases = read_cases(dataset, tags)
    runtime = build_runtime(deployment, env_file)
    report = asyncio.run(run_turn_eval(runtime, turn_cases, route=deployment.settings.model_route))
    typer.echo(render_turn_markdown(report))
    if output is not None:
        output.write_text(turn_report_json(report), encoding="utf-8")
    if not report.ok:
        raise typer.Exit(1)


@eval_app.command("report")
def eval_report_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", exists=True, file_okay=False, help="A live run's directory."),
    ],
    budget: BudgetOption = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="The tags the run was started with (repeatable)."),
    ] = None,
) -> None:
    """Judge a live run again with its labels.yaml; rewrite its report and review page."""
    eval_report(run_dir, budget, tag or [])


@eval_app.command("embedders")
def eval_embedders_command(
    env_file: EnvFile,
    cases: Annotated[
        Path, typer.Option("--cases", exists=True, dir_okay=False, help="The retrieval cases.")
    ],
    run_dir: Annotated[
        Path, typer.Option("--run-dir", file_okay=False, help="Where the results are kept.")
    ],
    budget: Annotated[
        Path, typer.Option("--budget", exists=True, dir_okay=False, help="The phase's budget file.")
    ],
    candidate: Annotated[
        list[str], typer.Option("--candidate", help="An embedding route to compare (repeatable).")
    ],
    yes: Annotated[bool, typer.Option("--yes", help="Start without asking.")] = False,
) -> None:
    """Compare the reference embedder with the candidates on the retrieval cases (M3 §4.5)."""
    eval_embedders(
        env_file,
        cases_path=cases,
        run_dir=run_dir,
        budget_path=budget,
        candidates=candidate,
        yes=yes,
    )


def _usage(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


@app.command("chat")
def chat(
    env_file: EnvFile,
    trace: Annotated[bool, typer.Option("--trace/--no-trace")] = True,
    live: Annotated[
        bool,
        typer.Option("--live", help="Answer with the real model: test messages only, and paid."),
    ] = False,
    budget: Annotated[
        Path | None,
        typer.Option("--budget", help="The budget file that pays for --live (required with it)."),
    ] = None,
) -> None:
    """Chat with this deployment's agent in the terminal.

    Offline by default. With --live the agent answers with the real model on its live route:
    every request is paid from --budget, which one session holds at a time, and only made-up
    test messages may be typed, never a real patient's data.
    """
    if budget is not None and not live:
        typer.echo("error: --budget only goes with --live", err=True)
        raise typer.Exit(1)
    if live:
        run_live_chat(env_file, budget, show_trace=trace, read_line=_read_line)
        return
    deployment = _deployment(env_file)
    runtime = build_runtime(deployment, env_file)
    typer.echo(f"{deployment.pack.name} ({deployment.pack.profile}) — type 'exit' to quit")
    asyncio.run(run_chat(runtime, read_line=_read_line, write=typer.echo, show_trace=trace))


def _read_line() -> str | None:
    line = sys.stdin.readline()
    return line or None


@calendar_app.command("check")
def calendar_check(
    env_file: EnvFile,
    doctor: Annotated[str, typer.Option("--doctor", help="The doctor's ID in the pack.")],
) -> None:
    """Read a doctor's calendar as the bot does: blocks, busy intervals and the first free slots.
    No event's content is printed."""
    raise typer.Exit(check_calendar(load_settings(env_file), doctor, datetime.now(UTC)))
