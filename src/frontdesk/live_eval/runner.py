"""Run labelled conversations on a live route and record every turn (M3 design §4.4, §5, §6).

Cases already recorded are skipped, so a run that stopped resumes where it stopped: no call of a
finished case is made again. A case is written only once all its turns have run, so a case cut
short (by the budget, a rate limit, a ledger that failed or Ctrl-C) leaves no turn behind and
runs again in full.

A case that ends in the failure an outage causes (the turn crashed or timed out, or a model call
failed) is held back before it is written. The provider may be down: the breaker needs three failed
attempts to say so, and the cases that fail with fewer would stand in the evidence, where a resume
skips them and the report counts them. So the record waits for a later case that the provider
answered (a request of it came back with its usage, and the case ended in no such failure: then the
failure was that case's own), and is written at the normal end of the run; when the meter stops the
run it is dropped, and the same command runs the case again.

A case the rules decide (an emergency, clinical advice, personal data) makes no request and shows
nothing of the provider. While a failure waits, such a case waits behind it, so that the file keeps
the order of the cases, and it is written with the failure; it costs nothing to run again. Each case
is still written on its own. The offline route has no provider that can be down, and holds nothing.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Final

from frontdesk.conversation.behaviour import FAILURE_RULES
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

# A model call that failed: the rejections an outage leaves. The others (an id the catalogue does
# not hold, a question left blank) are the model's own mistakes and stay evidence wherever they fall.
OUTAGE_REJECTIONS: Final = frozenset({"classifier_error", "responder_error"})

# The cases written down for later, in the order of the run: the id and the turns of each.
_Held = Sequence[tuple[str, tuple[TurnRecord, ...]]]


@dataclass(frozen=True)
class RunProgress:
    done: int  # cases recorded, this call's and earlier ones
    total: int
    stopped: str | None  # why the meter stopped the run (`Meter.stop_reason`), or None
    # The cases that ended in an outage failure and were held back when the meter stopped the run:
    # they are not recorded, so the same command runs them again. The cases the rules decided that
    # waited behind them are dropped as well, and are not named: they ended in no failure.
    dropped: tuple[str, ...] = ()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def ends_in_outage_failure(records: Sequence[TurnRecord]) -> bool:
    """Whether a case ended with the failure an outage causes: its last turn crashed or timed out,
    or a model call of it failed. An answer the model got wrong is not one."""
    last = records[-1]
    return last.rule in FAILURE_RULES or last.rejection in OUTAGE_REJECTIONS


def answered_by_the_model(records: Sequence[TurnRecord]) -> bool:
    """Whether a request of the case came back from the provider with its usage. A case the rules
    decide (an emergency, clinical advice, personal data) makes no request, and a request that
    came back without usage is recorded as unconfirmed, with no tokens: neither shows that the
    provider answers."""
    return any(
        call.input_tokens + call.output_tokens > 0 for turn in records for call in turn.calls
    )


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
    kept for the review. Without a meter (the offline route) nothing is reserved or recorded.
    The records of cases that ended in an outage failure are held, and so are the cases the rules
    decide while a failure waits (see the module); the progress names the failures a stop dropped."""
    keeping = replace(runtime, keep_drafts=True)
    done = set(completed_cases(read_turns(files)))
    pending = [case for case in cases if case.id not in done]
    held: list[tuple[str, tuple[TurnRecord, ...]]] = []
    for case in pending[:limit]:
        records = await _run_case(keeping, case, run_id=run_id, meter=meter, clock=clock, now=now)
        if records is None:
            stopped = meter.stop_reason if meter else None
            return RunProgress(len(done), len(cases), stopped, _failures(held))
        if _is_held_back(records, held, meter):
            held.append((case.id, records))
            continue
        done |= _write([*held, (case.id, records)], files)
        held.clear()
    done |= _write(held, files)  # the run ended normally: nothing stopped it, nothing is lost
    return RunProgress(len(done), len(cases), None)


def _is_held_back(records: Sequence[TurnRecord], held: _Held, meter: Meter | None) -> bool:
    """Whether a case waits before it is written: it ended in an outage failure, or it shows
    nothing of the provider while a failure waits. Only a metered run has a provider to be down."""
    if meter is None:
        return False
    return ends_in_outage_failure(records) or (bool(held) and not answered_by_the_model(records))


def _failures(held: _Held) -> tuple[str, ...]:
    """The held cases that ended in an outage failure, in order."""
    return tuple(name for name, records in held if ends_in_outage_failure(records))


def _write(cases: _Held, files: RunFiles) -> set[str]:
    """Write each case on its own, in order; the ids written."""
    for _, records in cases:
        append_case(files, records)
    return {name for name, _ in cases}


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
