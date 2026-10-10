"""`frontdesk eval embedders`: the embedder experiment from the command line (M3 design §4.5).

It runs only on a scratch database (its name ends in `_experiment`), as that database's owner,
because each candidate gets a table of its own dimension there. The reference is the deployment's
embedder (the demo's local MiniLM); each candidate is a Vertex AI route approved for evaluation.
Results are saved per embedder, so a run that stopped (budget, rate limit) resumes with the next.

Like the other live commands it refuses to start while a proxy, certificate or billing variable is
set, and it holds the budget's ledger while it measures, so that no other live run spends meanwhile.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import typer
from sqlalchemy import make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.cli.live_common import (
    eur_left,
    refuse_environment_overrides,
    refuse_retired,
    spent_usd,
)
from frontdesk.cli.runtime import fail, unreachable
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.migrate import upgrade_to_head
from frontdesk.knowledge.embedder_experiment import (
    Candidate,
    CandidateResult,
    decide,
    render_experiment,
    result_from_json,
    result_json,
    run_candidate,
)
from frontdesk.knowledge.embedders import Embedder, build_embedder
from frontdesk.knowledge.evaluation import RetrievalCase, load_cases
from frontdesk.knowledge.versions import (
    VersionStatus,
    activate,
    approve,
    create_draft,
    version_status,
)
from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError, load_budget
from frontdesk.model_gateway.embeddings import EmbeddingRateLimitedError, VertexEmbedder
from frontdesk.model_gateway.routes import DataClass, UnapprovedRouteError, require_route
from frontdesk.settings import EmbeddingBackend, Settings, load_settings
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

SCRATCH_SUFFIX: Final = "_experiment"
REFERENCE: Final = "reference"


def vertex_candidate(route_id: str, settings: Settings, budget: Budget, run_id: str) -> Candidate:
    """A Vertex AI candidate: one embedder for passages, one for questions."""
    try:
        route = require_route(route_id, DataClass.EVALUATION)
    except UnapprovedRouteError as error:
        raise fail(str(error)) from error
    if route.provider != "google-vertex-embedding" or route.dimension is None:
        raise fail(f"route '{route_id}' is not an embedding route")
    if settings.gcp_project is None or not settings.gcp_project.strip():
        raise fail("FRONTDESK_GCP_PROJECT is required for the Vertex AI candidates")
    refuse_retired(route)
    try:
        budget.price(route_id)
    except BudgetError as error:
        raise fail(str(error)) from error
    gcp_project = settings.gcp_project

    def embedder(task: str) -> Embedder:
        return VertexEmbedder(
            route,
            gcp_project=gcp_project,
            task="RETRIEVAL_DOCUMENT" if task == "documents" else "RETRIEVAL_QUERY",
            budget=budget,
            run_id=run_id,
        )

    return Candidate(route_id, route.dimension, embedder("documents"), embedder("queries"))


def eval_embedders(
    env_file: Path,
    *,
    cases_path: Path,
    run_dir: Path,
    budget_path: Path,
    candidates: Sequence[str],
    yes: bool,
) -> None:
    settings = load_settings(env_file)
    _check_scratch(settings)
    refuse_environment_overrides()
    try:
        budget = load_budget(budget_path)
        cases = load_cases(cases_path)
        pack = load_pack(settings.require_business_pack())
        left = eur_left(budget, spent_usd(budget))
    except ValueError as error:  # a budget, a dataset or a pack: each names its file
        raise fail(str(error)) from error
    chosen = [vertex_candidate(route, settings, budget, run_dir.name) for route in candidates]
    reference = build_embedder(settings.embedding_backend)
    typer.echo(
        f"{len(cases)} questions, {len(chosen)} candidates; the budget has "
        f"{left:.2f} of {budget.config.cap_eur:.2f} EUR left"
    )
    if not yes and not typer.confirm("Start the experiment?", default=False):
        raise typer.Exit(1)
    _require_scratch_server(settings.migration_database_url)
    upgrade_to_head(settings.migration_database_url)
    factory = make_session_factory(make_engine(settings.migration_database_url))
    _activate(factory, pack, reference)
    run_dir.mkdir(parents=True, exist_ok=True)
    everyone = [Candidate(REFERENCE, reference.dimension, reference, reference), *chosen]
    try:
        with budget.exclusive():  # no live run spends from the same ledger meanwhile
            results = _run_all(factory, pack, cases, everyone, run_dir, budget)
    except BudgetError as error:
        raise fail(str(error)) from error
    decision = decide(results[0], results[1:])
    report = render_experiment(
        results, decision, usd_per_eur=budget.config.exchange_rate.usd_per_eur
    )
    (run_dir / "embedders.md").write_text(report, encoding="utf-8")
    typer.echo(report)


def _run_all(
    factory: sessionmaker[Session],
    pack: BusinessPack,
    cases: Sequence[RetrievalCase],
    everyone: Sequence[Candidate],
    run_dir: Path,
    budget: Budget,
) -> list[CandidateResult]:
    """Each embedder's results, read back when an earlier run saved them, else measured now."""
    results: list[CandidateResult] = []
    for candidate in everyone:
        saved = run_dir / f"{candidate.name}.json"
        if saved.exists():
            results.append(result_from_json(saved.read_text(encoding="utf-8")))
            continue
        try:
            with business_session(factory, pack.id) as session:
                result = run_candidate(
                    session,
                    business_id=pack.id,
                    cases=cases,
                    candidate=candidate,
                    today=datetime.now(UTC).date(),
                    spent_usd=budget.spent_usd,
                )
        except (BudgetExhaustedError, EmbeddingRateLimitedError) as error:
            raise fail(
                f"{error}: the results so far are kept; run the same command to resume"
            ) from error
        saved.write_text(result_json(result) + "\n", encoding="utf-8")
        results.append(result)
    return results


def _check_scratch(settings: Settings) -> None:
    database = make_url(settings.migration_database_url).database or ""
    if not database.endswith(SCRATCH_SUFFIX):
        raise fail(
            f"the experiment writes tables: it runs only on a database whose name ends in "
            f"'{SCRATCH_SUFFIX}', not '{database}' (FRONTDESK_MIGRATION_DATABASE_URL)"
        )
    if settings.embedding_backend is not EmbeddingBackend.FASTEMBED:
        raise fail(
            "the reference is the demo's local model: set FRONTDESK_EMBEDDING_BACKEND=fastembed"
        )


def _require_scratch_server(url: str) -> None:
    """Ask the server which database the URL reaches, and refuse unless it is a scratch one.

    `_check_scratch` reads the name from the URL, but a driver lets a query parameter name another
    database than the path does (`?dbname=`, `?service=`). The server cannot be mistaken, and
    nothing has been written when it answers.
    """
    engine = make_engine(url)
    try:
        with engine.connect() as connection:
            reached = connection.scalar(text("SELECT current_database()")) or ""
    except OperationalError as error:
        raise unreachable(url) from error
    finally:
        engine.dispose()
    if not reached.endswith(SCRATCH_SUFFIX):
        raise fail(
            f"the experiment writes tables: the server says this URL reaches '{reached}', whose "
            f"name does not end in '{SCRATCH_SUFFIX}' (FRONTDESK_MIGRATION_DATABASE_URL)"
        )


def _activate(factory: sessionmaker[Session], pack: BusinessPack, reference: Embedder) -> None:
    """The pack's knowledge, active in the scratch database: nothing here reaches a person."""
    now = datetime.now(UTC)
    with business_session(factory, pack.id) as session:
        version = create_draft(session, pack).version_id
        if version_status(session, version) is VersionStatus.DRAFT:
            approve(
                session,
                version,
                approved_by="embedder experiment",
                evidence="scratch database, approved pack",
                now=now,
            )
        if version_status(session, version) is VersionStatus.APPROVED:
            activate(session, version, reference, now=now)
