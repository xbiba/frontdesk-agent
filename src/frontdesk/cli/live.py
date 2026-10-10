"""Live runs from the command line: `eval turns --live` and `eval report` (M3 design §4.4, §5, §6).

A live run checks everything it can before its first paid call: the route is approved for
evaluation data, pinned and not retired; no proxy, certificate or billing variable is set; the
credentials, the API and the model answer a free request; the ledger takes a line; and the budget
has room. It shows the most the run could cost and asks before it starts. It holds the budget's
ledger from before its first file to after its last call, so that a second run is refused.

It exits 0 when it is done, 1 on an error (a budget another run holds included), 3 when the rate
limit stopped it, 4 when the budget has no room (or cannot bound a request), 5 when the model keeps
failing, and 130 on Ctrl-C. A stop keeps what is recorded: the same command resumes the run.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import typer

from frontdesk.cli.live_common import (
    check_access,
    eur_left,
    refuse_environment_overrides,
    refuse_retired,
    spent_usd,
)
from frontdesk.cli.runtime import build_runtime, fail
from frontdesk.composition import Deployment, open_deployment
from frontdesk.conversation.evaluation import TurnCase, TurnDatasetError, load_turn_cases
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.db.engine import business_session
from frontdesk.knowledge.versions import active_version_id
from frontdesk.live_eval.judging import LabelsError
from frontdesk.live_eval.records import (
    RunFileError,
    RunFiles,
    RunManifest,
    completed_cases,
    manifest_differences,
    read_manifest,
    read_turns,
    write_manifest,
)
from frontdesk.live_eval.report import criteria, summarize, write_run_reports
from frontdesk.live_eval.runner import RunProgress, run_live_eval
from frontdesk.model_gateway.costs import Budget, BudgetError, load_budget
from frontdesk.model_gateway.factory import pinned_location
from frontdesk.model_gateway.metering import (
    BREAKER_ATTEMPTS,
    MAX_INPUT_TOKENS_PER_REQUEST,
    STOP_BUDGET,
    STOP_LEDGER,
    STOP_RATE_LIMITED,
    STOP_UNAVAILABLE,
    Meter,
)
from frontdesk.model_gateway.routes import (
    CHAT_PROVIDERS,
    DataClass,
    ModelRoute,
    UnapprovedRouteError,
    require_route,
)
from frontdesk.settings import Settings, load_settings

REQUESTS_PER_TURN: Final = 4  # two for the classifier, two for the responder (RESPONDER_LIMITS)
EXIT_ERROR: Final = 1
EXIT_RATE_LIMITED: Final = 3
EXIT_BUDGET: Final = 4
EXIT_UNAVAILABLE: Final = 5
EXIT_INTERRUPTED: Final = 130
_RESUME: Final = "run the same command again to resume"
_NAMED_CASES: Final = 5  # the cases a stop names when it drops some; the rest are counted


@dataclass(frozen=True)
class _Run:
    """A live run that is open and has not started: everything it needs, nothing it has written."""

    settings: Settings
    route: ModelRoute
    run_id: str
    files: RunFiles
    cases: tuple[TurnCase, ...]
    runtime: TurnRuntime
    manifest: RunManifest
    resumed: bool  # the directory already holds this run
    budget: Budget | None
    meter: Meter | None  # only a live route has one


def select_cases(cases: Sequence[TurnCase], tags: Sequence[str]) -> list[TurnCase]:
    """The cases with any of `tags`; all of them when no tag is given."""
    return [case for case in cases if not tags or set(tags) & set(case.tags)]


def read_cases(path: Path, tags: Sequence[str]) -> list[TurnCase]:
    try:
        cases = select_cases(load_turn_cases(path), tags)
    except TurnDatasetError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    if not cases:
        raise fail(f"{path}: no case has the tags {', '.join(tags)}")
    return cases


def evaluation_route(route_id: str) -> ModelRoute:
    """The route, if evaluation data may reach it, it answers messages, its host is pinned, and
    Google has not retired its model."""
    try:
        route = require_route(route_id, DataClass.EVALUATION)
        if route.provider != "offline":
            if route.provider not in CHAT_PROVIDERS:
                raise UnapprovedRouteError(
                    f"model route '{route_id}' cannot answer messages: its provider is "
                    f"{route.provider}"
                )
            pinned_location(route)
    except UnapprovedRouteError as error:
        raise fail(str(error)) from error
    refuse_retired(route)
    return route


def live_turns(
    env_file: Path,
    *,
    route_id: str,
    cases_path: Path | None,
    run_dir: Path,
    budget_path: Path | None,
    limit: int | None,
    yes: bool,
    tags: Sequence[str],
) -> None:
    """`frontdesk eval turns --live`: the checks, the plan, the run, the reports, the exit code."""
    route = evaluation_route(route_id)
    live = route.provider != "offline"
    if live:
        refuse_environment_overrides()
    budget = _budget(budget_path, route) if live or budget_path else None
    settings = load_settings(env_file).model_copy(update={"model_route": route.route_id})
    # Held from before the first file is written to after the last call is paid: a run started
    # meanwhile would see room that this one is about to use, so it is refused instead.
    lock: AbstractContextManager[None] = budget.exclusive() if budget else nullcontext()
    try:
        with lock:
            run = _open(settings, env_file, route, budget, cases_path, run_dir, tags)
            progress = _start(run, limit=limit, yes=yes)
    except BudgetError as error:
        raise fail(str(error)) from error
    except OSError as error:  # the run directory cannot take what the run records
        raise fail(f"{error.filename or run_dir}: {error.strerror or error}") from error
    except KeyboardInterrupt as stop:
        typer.echo(f"stopped: {_RESUME}", err=True)
        raise typer.Exit(EXIT_INTERRUPTED) from stop
    _write_reports(run.files, run.manifest, run.cases, budget)
    typer.echo(f"cases recorded: {progress.done}/{progress.total} in {run_dir}")
    if progress.stopped is not None:  # any stop, whatever its name: never a success
        detail = run.meter.stop_detail if run.meter else None
        raise _stopped(progress.stopped, detail, progress.dropped)


def eval_report(run_dir: Path, budget_path: Path | None, tags: Sequence[str] = ()) -> None:
    files = RunFiles(run_dir)
    try:
        manifest = read_manifest(files)
    except RunFileError as error:
        raise fail(str(error)) from error
    cases_path = Path(manifest.cases_file)
    cases = read_cases(cases_path, tags)
    if _digest(cases_path) != manifest.cases_digest:
        raise fail(f"{cases_path} changed since the run: its expected answers are not those judged")
    chosen = budget_path or (Path(manifest.budget_file) if manifest.budget_file else None)
    budget = _budget(chosen, None) if chosen else None
    _write_reports(files, manifest, cases, budget)


def _budget(path: Path | None, route: ModelRoute | None) -> Budget:
    if path is None:
        raise fail("--budget is required for a live route: every live call counts against it")
    try:
        budget = load_budget(path)
        if route is not None and route.provider != "offline":
            budget.price(route.route_id)
    except BudgetError as error:
        raise fail(str(error)) from error
    return budget


def _open(
    settings: Settings,
    env_file: Path,
    route: ModelRoute,
    budget: Budget | None,
    cases_path: Path | None,
    run_dir: Path,
    tags: Sequence[str],
) -> _Run:
    """Everything the run needs, built and checked; nothing is written yet."""
    deployment = open_deployment(settings)
    dataset = cases_path or settings.require_business_pack() / "evals" / "turns.yaml"
    cases = read_cases(dataset, tags)
    run_id = run_dir.resolve().name
    meter = None
    if budget is not None and route.provider != "offline":
        spent_usd(budget)  # a ledger that cannot be read is found before anything else is asked
        meter = Meter(budget, route_id=route.route_id, run_id=run_id, breaker=BREAKER_ATTEMPTS)
    runtime = build_runtime(
        deployment,
        env_file,
        data_class=DataClass.EVALUATION,
        wrap_model=meter.wrap if meter is not None else None,
    )
    manifest = _manifest(deployment, runtime, route, dataset, run_id, budget)
    files = RunFiles(run_dir)
    return _Run(
        settings=settings,
        route=route,
        run_id=run_id,
        files=files,
        cases=tuple(cases),
        runtime=runtime,
        manifest=manifest,
        resumed=_holds_this_run(files, manifest),
        budget=budget,
        meter=meter,
    )


def _manifest(
    deployment: Deployment,
    runtime: TurnRuntime,
    route: ModelRoute,
    dataset: Path,
    run_id: str,
    budget: Budget | None,
) -> RunManifest:
    pack = deployment.pack
    with business_session(deployment.session_factory, pack.id) as session:
        version = active_version_id(session, pack.id)
    return RunManifest(
        run_id=run_id,
        business_id=pack.id,
        route=route.route_id,
        model_name=route.model_name,
        engine_commit=os.environ.get("FRONTDESK_ENGINE_COMMIT", "unknown"),
        cases_file=str(dataset),
        cases_digest=_digest(dataset),
        knowledge_version=str(version),
        classifier_prompt=runtime.classifier_prompt_version,
        responder_prompt=runtime.responder_prompt_version,
        fixed_texts={language: dict(texts) for language, texts in pack.fixed_texts.items()},
        budget_file=str(budget.path) if budget else None,
        min_support_similarity=deployment.settings.min_support_similarity,
        min_retrieval_similarity=deployment.settings.min_retrieval_similarity,
        started_at=datetime.now(UTC),
    )


def _holds_this_run(files: RunFiles, manifest: RunManifest) -> bool:
    """Whether the directory already holds this run. One that holds another is refused: a resumed
    run must be the same run. A new run writes its manifest later, once it is sure to start."""
    if not files.manifest.exists():
        return False
    try:
        differences = manifest_differences(read_manifest(files), manifest)
    except RunFileError as error:
        raise fail(str(error)) from error
    if differences:
        raise fail(
            f"{files.directory} holds another run (it differs in {', '.join(differences)}): "
            "choose a new --run-dir"
        )
    return True


def _start(run: _Run, *, limit: int | None, yes: bool) -> RunProgress:
    """The checks that need the network, the plan and the question, then the run itself."""
    pending = _pending(run, limit)
    if pending and run.meter is not None:
        check_access(run.route, run.settings.gcp_project)
    _show_plan(run, pending)
    if pending:
        if not yes and not typer.confirm("Start the run?", default=False):
            raise typer.Exit(EXIT_ERROR)
        _claim(run)  # before the ledger is touched: a refused run leaves no line in it
        if run.meter is not None:
            run.meter.check_ledger()  # the last gate before the first paid request
    return _run(run, limit)


def _run(run: _Run, limit: int | None) -> RunProgress:
    """The cases themselves. Ctrl-C ends the run with what it recorded, and says so."""
    try:
        return asyncio.run(
            run_live_eval(
                run.runtime,
                run.cases,
                files=run.files,
                run_id=run.run_id,
                meter=run.meter,
                limit=limit,
            )
        )
    except KeyboardInterrupt as stop:
        # The reports of an earlier invocation would be stale: bring them up to date first. A
        # report that cannot be written has said so (an error line); the exit code stays 130.
        with suppress(typer.Exit):
            _write_reports(run.files, run.manifest, run.cases, run.budget)
        typer.echo(f"stopped: {_RESUME}", err=True)
        raise typer.Exit(EXIT_INTERRUPTED) from stop


def _claim(run: _Run) -> None:
    """Claim the run directory: write the manifest of a new run (which creates the directory) and
    open the record file for appending, so that a directory that cannot take what the run records
    refuses it here, before a request is paid for a case that could not be written down."""
    if not run.resumed:
        write_manifest(run.files, run.manifest)
    with run.files.turns.open("a", encoding="utf-8"):
        pass


def _pending(run: _Run, limit: int | None) -> list[TurnCase]:
    try:
        done = set(completed_cases(read_turns(run.files)))
    except RunFileError as error:
        raise fail(str(error)) from error
    return [case for case in run.cases if case.id not in done][:limit]


def _show_plan(run: _Run, pending: Sequence[TurnCase]) -> None:
    turns = sum(len(case.conversation) for case in pending)
    route = run.route
    typer.echo(f"{route.route_id} ({route.model_name}): {len(pending)} cases, {turns} turns to run")
    if run.meter is None or run.budget is None:
        return
    price = run.budget.price(route.route_id)
    cap = max(route.output_limits.classifier, route.output_limits.responder)
    most = (
        turns
        * REQUESTS_PER_TURN
        * price.cost(input_units=MAX_INPUT_TOKENS_PER_REQUEST, output_units=cap)
    )
    left = eur_left(run.budget, spent_usd(run.budget))
    typer.echo(
        f"at most {run.budget.eur(most):.2f} EUR; the budget has "
        f"{left:.2f} of {run.budget.config.cap_eur:.2f} EUR left"
    )


def _stopped(reason: str, detail: str | None, dropped: Sequence[str] = ()) -> typer.Exit:
    """Say why the run stopped, and give the exit code a script can act on. Any reason is a stop:
    one this function does not know exits 1 and names itself. The cases the stop dropped from the
    evidence (see `run_live_eval`) are named on a line of their own."""
    cause = f" ({detail})" if detail else ""
    if reason == STOP_RATE_LIMITED:
        code, line = (
            EXIT_RATE_LIMITED,
            "stopped by the rate limit: run the same command later to resume",
        )
    elif reason == STOP_BUDGET:
        # The cause says whether the budget is spent or a request cannot be bounded.
        code, line = EXIT_BUDGET, f"stopped: {detail or 'the budget has no room for the next call'}"
    elif reason == STOP_UNAVAILABLE:
        resume = "run the same command later to resume"
        code, line = EXIT_UNAVAILABLE, f"stopped: the model keeps failing{cause}: {resume}"
    elif reason == STOP_LEDGER:
        counted = "stopped: the ledger can no longer count what is spent"
        code, line = EXIT_ERROR, f"{counted}{cause}: fix it, then {_RESUME}"
    else:
        code, line = EXIT_ERROR, f"stopped: {reason}{cause}"
    typer.echo(line, err=True)
    if dropped:
        typer.echo(_dropped_line(dropped), err=True)
    return typer.Exit(code)


def _dropped_line(dropped: Sequence[str]) -> str:
    names = ", ".join(dropped[:_NAMED_CASES])
    if len(dropped) > _NAMED_CASES:
        names += f" and {len(dropped) - _NAMED_CASES} more"
    if len(dropped) == 1:
        return (
            "1 case ended in a failure just before the stop and is not recorded: "
            f"the same command runs it again ({names})"
        )
    return (
        f"{len(dropped)} cases ended in a failure just before the stop and are not recorded: "
        f"the same command runs them again ({names})"
    )


def _write_reports(
    files: RunFiles, manifest: RunManifest, cases: Sequence[TurnCase], budget: Budget | None
) -> None:
    try:
        spent = budget.eur(spent_usd(budget)) if budget else None
        verdicts = write_run_reports(files, manifest, cases, budget=budget)
    except (LabelsError, RunFileError, BudgetError) as error:
        raise fail(str(error)) from error
    except OSError as error:  # a run directory that cannot take the reports
        raise fail(f"{error.filename or files.directory}: {error.strerror or error}") from error
    cap = budget.config.cap_eur if budget else None
    summaries = summarize(verdicts, languages=manifest.fixed_texts)
    complete = len(verdicts) == len(cases)
    for criterion in criteria(summaries, spent_eur=spent, cap_eur=cap, complete=complete):
        typer.echo(f"{criterion.status.upper():8} {criterion.name}: {criterion.value}")
    typer.echo(
        f"report: {files.directory / 'report.md'} · review: {files.directory / 'review.html'}"
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
