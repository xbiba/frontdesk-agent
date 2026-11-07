"""`frontdesk eval email`: the email drafts on invented emails (design §5), through the real pass
on a file mailbox, with the live drafting route and a judge route, every request metered against
the budget's ledger, and a limit of its own in US dollars.

It checks before it spends: both routes take evaluation data, the environment moves no request,
the key and the models answer Anthropic's free token count, the ledger takes a line. It writes
run.json, results.jsonl, report.json and report.md in the run directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import typer

from frontdesk.agent.prompts import read_prompt
from frontdesk.cli.live import evaluation_route
from frontdesk.cli.live_common import check_access, refuse_environment_overrides, route_api_key
from frontdesk.cli.runtime import fail
from frontdesk.email.drafter import MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS, Drafter
from frontdesk.email.evaluation import (
    CasesError,
    EmailEval,
    Judge,
    case_thread,
    load_email_cases,
    render_markdown,
    row_json,
    summary,
)
from frontdesk.email.job import EmailPass
from frontdesk.email.ledger import MemoryLedger
from frontdesk.email.mailbox import FileMailbox
from frontdesk.model_gateway.costs import BudgetError, BudgetExhaustedError, load_budget
from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.metering import BREAKER_ATTEMPTS, CallRecord, Meter, RunStoppedError
from frontdesk.model_gateway.routes import ModelRoute
from frontdesk.settings import load_settings
from frontdesk.tenancy.loader import PackError, load_pack

JUDGE_ROUTE: Final = "anthropic-opus-5-5"
JUDGE_MAX_TOKENS: Final = 4000
STOPS: Final = (RunStoppedError, BudgetExhaustedError, BudgetError)


def _model(route: ModelRoute, api_key: str | None, gcp_project: str | None) -> Any:
    return build_live_model(route, gcp_project=gcp_project, api_key=api_key)


def eval_email(
    env_file: Path,
    *,
    cases_path: Path,
    run_dir: Path,
    budget_path: Path,
    route_id: str,
    judge_route_id: str,
    max_usd: float,
    yes: bool,
) -> int:
    settings = load_settings(env_file)
    try:
        pack = load_pack(settings.require_business_pack())
        cases = load_email_cases(cases_path)
        budget = load_budget(budget_path)
    except (PackError, CasesError, BudgetError, ValueError) as error:
        raise fail(str(error)) from error
    if pack.email is None:
        raise fail(f"the pack of {pack.id} has no email section (email:)")
    # Every case drafts: an evaluation is not a mailbox in tester mode.
    email = pack.email.model_copy(update={"testers_only": False})
    pack = pack.model_copy(update={"email": email})
    route, judge_route = evaluation_route(route_id), evaluation_route(judge_route_id)
    refuse_environment_overrides()
    keys = {item.route_id: route_api_key(settings, item) for item in (route, judge_route)}
    for item in (route, judge_route):
        check_access(item, settings.gcp_project, keys[item.route_id])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise fail(f"{run_dir} is not empty: a run directory holds one run")
    typer.echo(
        f"{len(cases)} cases on {route.route_id}, judged by {judge_route.route_id}; "
        f"this run stops at {max_usd:.2f} USD"
    )
    if not yes and not typer.confirm("start the live run?"):
        return 1
    try:
        with budget.exclusive():
            meter = Meter(
                budget, route_id=route.route_id, run_id=run_dir.name, breaker=BREAKER_ATTEMPTS
            )
            meter.check_ledger()
            rows, meta = asyncio.run(
                _run(pack, cases, route, judge_route, keys, settings.gcp_project, meter, max_usd)
            )
    except BudgetError as error:
        raise fail(str(error)) from error
    meta.update(
        cases_file=str(cases_path),
        cases_digest=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        engine_commit=os.environ.get("FRONTDESK_ENGINE_COMMIT", "unknown"),
        business_id=pack.id,
        run_id=run_dir.name,
    )
    _write(run_dir, rows, meta)
    totals = summary(rows)
    typer.echo(
        f"clean {totals['clean']}/{totals['n']}, {meta['usd']:.4f} USD; written to {run_dir}"
    )
    if meta["stopped"]:
        typer.echo(f"error: the run stopped: {meta['stopped']}", err=True)
        return 1
    return 0


async def _run(
    pack: Any,
    cases: Any,
    route: ModelRoute,
    judge_route: ModelRoute,
    keys: dict[str, str | None],
    gcp_project: str | None,
    meter: Meter,
    max_usd: float,
) -> tuple[Any, dict[str, Any]]:
    started = datetime.now(UTC)
    calls: list[CallRecord] = []

    def spent() -> float:
        calls.extend(meter.drain())
        return sum(call.usd for call in calls)

    slow = replace(route, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    drafting = meter.wrap(_model(slow, keys[route.route_id], gcp_project), "agent")
    judging = meter.wrap(
        _model(judge_route, keys[judge_route.route_id], gcp_project),
        "judge",
        route_id=judge_route.route_id,
    )
    prompt = read_prompt(pack.profile, "email")
    today = started.date()
    drafter = Drafter(
        drafting,
        pack,
        pack.email,
        prompt=prompt,
        today=today,
        model_settings=live_model_settings(route, max_tokens=MAX_OUTPUT_TOKENS),
    )
    email_pass = EmailPass(
        mailbox=FileMailbox([case_thread(case) for case in cases]),
        ledger=MemoryLedger(),
        drafter=drafter,
        pack=pack,
        now=lambda: datetime.now(UTC),
        price=meter.price_of(route.route_id),
        stops=STOPS,
        sender_secret=secrets.token_bytes(32),  # this run's own: nothing is kept
    )
    judge = Judge(
        judging,
        pack,
        pack.email,
        today=today,
        model_settings=live_model_settings(judge_route, max_tokens=JUDGE_MAX_TOKENS),
    )
    evaluation = EmailEval(
        email_pass, judge, pack.email.todo, spent=spent, max_usd=max_usd, stops=STOPS
    )
    rows = await evaluation.run(cases)
    spent()
    draft_usd = sum(call.usd for call in calls if call.agent == "agent")
    judge_usd = sum(call.usd for call in calls if call.agent == "judge")
    meta = {
        "route": route.route_id,
        "model_name": route.model_name,
        "judge_route": judge_route.route_id,
        "judge_model": judge_route.model_name,
        "prompt_version": prompt.version,
        "started_at": started.isoformat(),
        "today": today.isoformat(),
        "usd": draft_usd + judge_usd,
        "draft_usd": draft_usd,
        "judge_usd": judge_usd,
        "requests": len(calls),
        "stopped": evaluation.stopped or meter.stop_detail,
    }
    return rows, meta


def _write(run_dir: Path, rows: Any, meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row_json(row), ensure_ascii=False) + "\n")
    report = {"summary": summary(rows), "meta": meta}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(rows, meta), encoding="utf-8")
