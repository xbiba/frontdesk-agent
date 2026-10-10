"""Run labelled conversations on a live route and record every turn (M3 design §4.4, §5, §6).

Cases already recorded are skipped, so a run that stopped resumes where it stopped: no call of a
finished case is made again. A case is written only once all its turns have run, so a case cut
short (by the budget, a rate limit, a ledger that failed or Ctrl-C) leaves no turn behind and
runs again in full.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

from frontdesk.conversation.evaluation import TurnCase, initial_state
from frontdesk.conversation.turn import RejectedDraft, TurnResult, TurnRuntime, handle_message
from frontdesk.live_eval.records import (
    CallRow,
    ClaimRow,
    DraftRow,
    RunFiles,
    TurnRecord,
    append_case,
    completed_cases,
    read_turns,
)
from frontdesk.model_gateway.metering import CallRecord, Meter


@dataclass(frozen=True)
class RunProgress:
    done: int  # cases recorded, this call's and earlier ones
    total: int
    stopped: str | None  # "budget", "rate_limited" or "ledger" when the run had to stop


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def run_live_eval(
    runtime: TurnRuntime,
    cases: Sequence[TurnCase],
    *,
    files: RunFiles,
    run_id: str,
    meter: Meter | None,
    limit: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], datetime] = _utc_now,
) -> RunProgress:
    """Run the cases not yet recorded, at most `limit` of them; drafts refused by a check are
    kept for the review. Without a meter (the offline route) nothing is reserved or recorded."""
    keeping = replace(runtime, keep_drafts=True)
    done = set(completed_cases(read_turns(files)))
    pending = [case for case in cases if case.id not in done]
    for case in pending[:limit]:
        records = await _run_case(keeping, case, run_id=run_id, meter=meter, clock=clock, now=now)
        if records is None:
            return RunProgress(len(done), len(cases), meter.stop_reason if meter else None)
        append_case(files, records)
        done.add(case.id)
    return RunProgress(len(done), len(cases), None)


async def _run_case(
    runtime: TurnRuntime,
    case: TurnCase,
    *,
    run_id: str,
    meter: Meter | None,
    clock: Callable[[], float],
    now: Callable[[], datetime],
) -> tuple[TurnRecord, ...] | None:
    """The turns of one case, or None when the meter stopped the run in the middle of it."""
    attempt = uuid.uuid4().hex[:12]
    state = initial_state(case)
    records: list[TurnRecord] = []
    if meter is not None:
        meter.drain()
    for number, message in enumerate(case.conversation, start=1):
        started = clock()
        result, state = await handle_message(message, state, runtime)
        seconds = clock() - started
        calls = meter.drain() if meter is not None else ()
        if meter is not None and meter.stop_reason is not None:
            return None
        records.append(
            _record(
                result,
                run_id=run_id,
                attempt=attempt,
                case=case,
                turn=number,
                message=message,
                calls=calls,
                seconds=seconds,
                at=now(),
            )
        )
    return tuple(records)


def _record(
    result: TurnResult,
    *,
    run_id: str,
    attempt: str,
    case: TurnCase,
    turn: int,
    message: str,
    calls: Sequence[CallRecord],
    seconds: float,
    at: datetime,
) -> TurnRecord:
    trace = result.trace
    return TurnRecord(
        run_id=run_id,
        attempt=attempt,
        case_id=case.id,
        turn=turn,
        turns=len(case.conversation),
        message=message,
        reply=result.text,
        language=result.language,
        outcome=result.outcome,
        rule=trace.rule,
        category=trace.category,
        intent=trace.intent,
        rejection=trace.rejection,
        check_reasons=trace.check_reasons,
        sources=trace.sources,
        draft=_draft_row(result.draft),
        calls=tuple(CallRow.model_validate(asdict(call)) for call in calls),
        seconds=seconds,
        at=at,
    )


def _draft_row(draft: RejectedDraft | None) -> DraftRow | None:
    if draft is None:
        return None
    claims = tuple(
        ClaimRow(text=c.text, evidence_id=c.evidence_id, quote=c.quote) for c in draft.claims
    )
    return DraftRow(action=draft.action, clarify=draft.clarify, claims=claims)
