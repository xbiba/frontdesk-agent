"""`frontdesk eval embedders`: the embedder experiment from the command line (M3 design §4.5).

It runs only on a scratch database (its name ends in `_experiment`), as that database's owner,
because each candidate gets a table of its own dimension there. The reference is the deployment's
embedder (the demo's local MiniLM); each candidate is a Vertex AI route approved for evaluation.
Results are saved per embedder, so a run that stopped (budget, rate limit) resumes with the next.

Like the other live commands it refuses to start while a proxy, certificate or billing variable is
set, proves that the ledger takes a line before the first paid text, and holds the ledger while it
measures, so that no other live run spends meanwhile. Each stop it expects (the budget, a rate
limit, Vertex AI not answering or refusing, a ledger or a file it cannot use, a wrong input) ends in
one `error:` line and exit code 1, and Ctrl-C in `stopped:` and exit code 130; the results saved so
far are kept, and the same command resumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import typer
from sqlalchemy import make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.cli.live import EXIT_INTERRUPTED
from frontdesk.cli.live_common import (
    eur_left,
    prove_ledger,
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
from frontdesk.knowledge.evaluation import RetrievalCase, load_cases, unknown_relevant_documents
from frontdesk.knowledge.versions import (
    VersionStatus,
    activate,
    approve,
    create_draft,
    version_status,
)
from frontdesk.model_gateway.access import RouteAccessError
from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError, load_budget
from frontdesk.model_gateway.embeddings import EmbeddingRateLimitedError, VertexEmbedder
from frontdesk.model_gateway.routes import DataClass, UnapprovedRouteError, require_route
from frontdesk.settings import EmbeddingBackend, Settings, load_settings
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

SCRATCH_SUFFIX: Final = "_experiment"
REFERENCE: Final = "reference"
_RESUME: Final = "run the same command again to resume"


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


@dataclass(frozen=True)
class _Experiment:
    """What the command has read and built by the time it asks: nothing here has cost anything."""

    settings: Settings
    budget: Budget
    cases: Sequence[RetrievalCase]
    cases_path: Path
    pack: BusinessPack
    reference: Embedder
    chosen: Sequence[Candidate]
    run_dir: Path

    @property
    def everyone(self) -> list[Candidate]:
        """The reference first, then the candidates in the order given."""
        reference = Candidate(REFERENCE, self.reference.dimension, self.reference, self.reference)
        return [reference, *self.chosen]


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
    again = next((route for route in candidates if candidates.count(route) > 1), None)
    if again is not None:
        raise fail(f"candidate '{again}' is given more than once")
    chosen = [vertex_candidate(route, settings, budget, run_dir.name) for route in candidates]
    reference = build_embedder(settings.embedding_backend)
    typer.echo(
        f"{len(cases)} questions, {len(chosen)} candidates; the budget has "
        f"{left:.2f} of {budget.config.cap_eur:.2f} EUR left"
    )
    if not yes and not typer.confirm("Start the experiment?", default=False):
        raise typer.Exit(1)
    _require_scratch_server(settings.migration_database_url)
    experiment = _Experiment(settings, budget, cases, cases_path, pack, reference, chosen, run_dir)
    try:
        report = _conclude(experiment, _measure(experiment))
    except KeyboardInterrupt as stop:  # the request in flight has its line in the ledger
        typer.echo(f"stopped: {_RESUME}", err=True)
        raise typer.Exit(EXIT_INTERRUPTED) from stop
    except BudgetError as error:  # the ledger, or another run that holds it
        raise fail(str(error)) from error
    except OSError as error:  # a run directory or a file in it that cannot be made, read or written
        # A rename names the file it was moving first and the one it was moving it to second.
        raise fail(
            f"{error.filename2 or error.filename or run_dir}: {error.strerror or error}"
        ) from error
    typer.echo(report)


def _measure(experiment: _Experiment) -> list[CandidateResult]:
    """Everything after the question, under the hold on the budget's ledger.

    The ledger is proved before anything else is written: a ledger that cannot take a line would
    let the first paid text through uncounted.
    """
    settings, run_dir = experiment.settings, experiment.run_dir
    with experiment.budget.exclusive():  # no live run spends from the same ledger meanwhile
        first = experiment.chosen[0].name if experiment.chosen else REFERENCE
        prove_ledger(experiment.budget, run_id=run_dir.name, route_id=first)
        upgrade_to_head(settings.migration_database_url)
        factory = make_session_factory(make_engine(settings.migration_database_url))
        _activate(factory, experiment.pack, experiment.reference)
        _require_known_documents(factory, experiment)
        run_dir.mkdir(parents=True, exist_ok=True)
        return _run_all(
            factory,
            experiment.pack,
            experiment.cases,
            experiment.everyone,
            run_dir,
            experiment.budget,
        )


def _conclude(experiment: _Experiment, results: Sequence[CandidateResult]) -> str:
    """The decision of the rule on the results, as the report that is saved and shown."""
    decision = decide(results[0], results[1:])
    rate = experiment.budget.config.exchange_rate.usd_per_eur
    report = render_experiment(results, decision, usd_per_eur=rate)
    _save(experiment.run_dir / "embedders.md", report)
    return report


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
            results.append(_read_saved(saved))
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
        except RouteAccessError as error:  # one sentence that says what to do about Vertex AI
            raise fail(
                f"{error}; the results so far are kept; run the same command to resume"
            ) from error
        _save(saved, result_json(result) + "\n")
        results.append(result)
    return results


def _read_saved(path: Path) -> CandidateResult:
    try:
        return result_from_json(path.read_text(encoding="utf-8"))
    except (ValueError, LookupError, TypeError, AttributeError) as error:
        raise fail(
            f"{path}: is not a result of this experiment; delete it to measure that embedder again"
        ) from error


def _save(path: Path, text: str) -> None:
    """Write a file whole or not at all: a stop in the middle leaves the old file, or none, never
    half of one that the next run would trip on."""
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(text, encoding="utf-8")
    try:
        partial.replace(path)
    except OSError:
        with suppress(OSError):
            partial.unlink()
        raise


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


def _require_known_documents(factory: sessionmaker[Session], experiment: _Experiment) -> None:
    """Refuse a case whose relevant document is not one of the knowledge, as `eval retrieval` does:
    a mistyped key can never be found, so it would score 0 for every embedder without a word."""
    pack = experiment.pack
    with business_session(factory, pack.id) as session:
        unknown = unknown_relevant_documents(session, business_id=pack.id, cases=experiment.cases)
    for case_id, key in unknown:
        typer.echo(
            f"error: {experiment.cases_path}: case '{case_id}': '{key}' is not a document of the "
            "active knowledge version (a document key is its file name without .md)",
            err=True,
        )
    if unknown:
        raise typer.Exit(1)


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
