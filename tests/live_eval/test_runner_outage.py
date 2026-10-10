"""A case that ends in an outage failure is not evidence until a later case shows the provider
answering (M3 Task 8, fix round 1, review I-1).

An outage shows as fewer than three failed attempts in a case: a refusal or a reset (not retried), or
one case cut by the turn limit while the provider hangs. The breaker stops the run at the case that
fails the third time, and until then the earlier cases had been written to `turns.jsonl`, where a
resume treated them as finished and the report counted them. Now the record of such a case is held:
written when a later case that a model call answered completes without that kind of failure (the
provider answered, so the failure was the case's own), or when the run ends normally, and dropped
when the meter stops the run, so that the same command runs the case again. A case the rules decide
makes no request and shows nothing of the provider: while a failure waits, it waits behind it
(fix round 2, review N-1).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic_ai import ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval import runner
from frontdesk.live_eval.records import RunFiles, TurnRecord, read_turns
from frontdesk.live_eval.runner import answered_by_the_model, ends_in_outage_failure, run_live_eval
from frontdesk.tenancy.pack import BusinessPack
from tests.conversation.helpers import document_responder, make_runtime
from tests.live_eval.helpers import call, record
from tests.live_eval.rig import CASES, Activate, Rig
from tests.live_eval.turn_limit import hang

db = pytest.mark.db


def _reset() -> None:
    """Sent, and the connection was reset: not retried, so the case ends with rule=error."""
    raise httpx2.ReadError("connection reset by peer")


def _refused() -> None:
    """Google refuses the request: the classifier step falls back with rejection=classifier_error."""
    raise ModelHTTPError(403, "gemini-3.1-flash-lite", body={"error": {"code": 403}})


def _rate_limit() -> None:
    raise ModelHTTPError(429, "gemini-3.1-flash-lite", body={"error": {"code": 429}})


def _single_turn_cases(*case_ids: str) -> list[TurnCase]:
    return [
        TurnCase.model_validate(
            {"id": case_id, "message": "Quanto costa la fisioterapia?", "expect": {}}
        )
        for case_id in case_ids
    ]


def _rules_only_case(case_id: str) -> TurnCase:
    """An emergency: the rules answer it, and no request is made."""
    return TurnCase.model_validate({"id": case_id, "message": "Mi fa male il petto", "expect": {}})


def _rules(records: tuple[TurnRecord, ...]) -> list[tuple[str, int, str]]:
    return [(r.case_id, r.turn, r.rule) for r in records]


@pytest.fixture(autouse=True)
def _no_waiting_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frontdesk.model_gateway.metering.RETRY_WAITS", (0.0, 0.0, 0.0))


# --- Which cases are held ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "outage"),
    [
        pytest.param({"rule": "error"}, True, id="a-crash"),
        pytest.param({"rule": "timeout"}, True, id="a-timeout"),
        pytest.param({"rejection": "classifier_error"}, True, id="the-classifier-call-failed"),
        pytest.param({"rejection": "responder_error"}, True, id="the-responder-call-failed"),
        pytest.param({"rejection": "unknown_service_id"}, False, id="the-model-got-an-id-wrong"),
        pytest.param({"rejection": "empty_text"}, False, id="the-model-wrote-nothing"),
        pytest.param({"rejection": "intent_not_allowed"}, False, id="a-designed-handoff"),
        pytest.param({"rule": "emergency"}, False, id="an-emergency"),
        pytest.param({}, False, id="a-reply"),
    ],
)
def test_only_a_failure_of_the_call_or_the_turn_is_the_failure_an_outage_causes(
    fields: dict[str, Any], outage: bool
) -> None:
    assert ends_in_outage_failure([record("a", "ciao", **fields)]) is outage


def test_a_conversation_is_judged_by_its_last_turn() -> None:
    failed = record("a", "ciao", turn=1, turns=2, rule="error")
    answered = record("a", "ciao", turn=2, turns=2)
    assert ends_in_outage_failure([failed, answered]) is False  # it recovered within the case
    assert ends_in_outage_failure([answered, failed]) is True


# --- When a held case is written ---------------------------------------------------------------------


# (what the request does, the rule of the case it ends, its rejection)
RESET = (_reset, "error", None)
HANG = (hang, "timeout", None)  # never answers: the turn limit cuts it, the case is a timeout
REFUSED = (_refused, "pass", "classifier_error")
FAILURES = [
    pytest.param(RESET, id="a-reset"),
    pytest.param(HANG, id="a-case-cut-by-the-turn-limit"),
    pytest.param(REFUSED, id="a-refusal"),
]


@db
@pytest.mark.parametrize("failure", FAILURES)
async def test_a_case_that_failed_is_written_when_a_later_case_completes_cleanly_in_order(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    tmp_path: Path,
    failure: tuple[Callable[[], Any], str, str | None],
) -> None:
    action, rule, rejection = failure
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={1: action})

    progress = await rig.start()

    assert (progress.done, progress.total, progress.stopped, progress.dropped) == (3, 3, None, ())
    # "one" failed on its own: the provider answered the cases after it, so it is evidence, and it
    # comes before them in the file as it does in the cases.
    records = read_turns(rig.files)
    assert _rules(records) == [
        ("one", 1, rule),
        ("two", 1, "pass"),
        ("two", 2, "pass"),
        ("three", 1, "pass"),
    ]
    assert records[0].rejection == rejection


@db
async def test_a_case_that_failed_last_is_written_when_the_run_ends_normally(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={4: _reset})

    progress = await rig.start()  # "three" is the fourth request

    assert (progress.done, progress.stopped, progress.dropped) == (3, None, ())
    assert _rules(read_turns(rig.files))[-1] == ("three", 1, "error")


@db
async def test_a_case_that_failed_is_written_when_the_limit_ends_the_run(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={1: _reset})

    progress = await rig.start(limit=1)

    assert (progress.done, progress.stopped, progress.dropped) == (1, None, ())
    assert _rules(read_turns(rig.files)) == [("one", 1, "error")]


@db
async def test_every_case_is_written_on_its_own_whether_it_was_held_or_not(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[list[str]] = []
    real = runner.append_case

    def spy(files: RunFiles, records: Any) -> None:
        writes.append([each.case_id for each in records])
        real(files, records)

    monkeypatch.setattr(runner, "append_case", spy)
    cases = _single_turn_cases("a", "b", "c", "d")
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={1: _reset, 2: _reset})

    progress = await rig.start(cases)

    assert progress.done == 4
    # "a" and "b" were held and are written one by one before "c"; none shares a write.
    assert writes == [["a"], ["b"], ["c"], ["d"]]


# --- When a held case is dropped ---------------------------------------------------------------------


@db
async def test_the_breaker_drops_the_cases_held_and_the_same_command_runs_them_again(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    cases = _single_turn_cases("a", "b", "c")
    failing = {1: _reset, 2: _reset, 3: _reset}
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request=failing)

    stopped = await rig.start(cases, breaker=3)

    assert (stopped.done, stopped.total, stopped.stopped) == (0, 3, "unavailable")
    assert stopped.dropped == ("a", "b")  # they failed just before the stop: not evidence
    assert read_turns(rig.files) == ()  # and "c", which the stop cut, is not there either

    resumed = await rig.start(cases, breaker=3)  # the provider answers again

    assert (resumed.done, resumed.stopped, resumed.dropped) == (3, None, ())
    assert _rules(read_turns(rig.files)) == [("a", 1, "pass"), ("b", 1, "pass"), ("c", 1, "pass")]


STOPS = [
    pytest.param(failure, cause, id=f"{cause}-{name}")
    for name, failure in (("reset", RESET), ("hang", HANG), ("refusal", REFUSED))
    for cause in ("rate_limited", "budget", "ledger")
    # A refusal costs nothing: no spent budget can follow it at the very next request.
    if (name, cause) != ("refusal", "budget")
]


@db
@pytest.mark.parametrize(("failure", "cause"), STOPS)
async def test_whatever_stops_the_run_drops_the_cases_held(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    tmp_path: Path,
    cause: str,
    failure: tuple[Callable[[], Any], str, str | None],
) -> None:
    action, _, _ = failure
    on_request: dict[int, Callable[[], Any]] = {1: action}
    # Case "two" has two turns, requests 2 and 3: the stop comes while it runs.
    if cause == "rate_limited":
        on_request |= {2: _rate_limit, 3: _rate_limit, 4: _rate_limit, 5: _rate_limit}
    if cause == "ledger":

        def damage() -> None:
            with rig.budget.ledger_path.open("a", encoding="utf-8") as ledger:
                ledger.write("not json\n")

        on_request[2] = damage
    # A request that is cut off is recorded at its reserve of 0.038 USD (a cap of 0.06 USD).
    rig = Rig(
        clinic_pack,
        activate_pack(clinic_pack),
        tmp_path,
        on_request=on_request,
        cap_eur=0.048 if cause == "budget" else 10.0,
    )

    progress = await rig.start()

    assert (progress.done, progress.stopped) == (0, cause)
    assert progress.dropped == ("one",)
    assert read_turns(rig.files) == ()


@db
async def test_a_run_that_is_cancelled_loses_the_cases_held_and_runs_them_again(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    # Ctrl-C reaches the run as a cancellation: what was held back goes with it, like the case that
    # was away, and the same command runs both again.
    def cancelled() -> None:
        raise asyncio.CancelledError

    cases = _single_turn_cases("a", "b")
    rig = Rig(
        clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={1: _reset, 2: cancelled}
    )

    with pytest.raises(asyncio.CancelledError):
        await rig.start(cases)
    assert read_turns(rig.files) == ()

    resumed = await rig.start(cases)
    assert (resumed.done, resumed.stopped) == (2, None)
    assert _rules(read_turns(rig.files)) == [("a", 1, "pass"), ("b", 1, "pass")]


@db
async def test_a_case_the_model_got_wrong_is_no_outage_and_is_recorded_at_once(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    wrong = {"category": "administrative", "intent": "price", "service_ids": ["no-such-service"]}

    def stop() -> None:
        assert rig.meter is not None
        rig.meter.rate_limited()

    rig = Rig(
        clinic_pack,
        activate_pack(clinic_pack),
        tmp_path,
        on_request={1: lambda: wrong, 2: stop},
    )

    progress = await rig.start()

    assert (progress.done, progress.stopped, progress.dropped) == (1, "rate_limited", ())
    [only] = read_turns(rig.files)
    assert (only.case_id, only.rejection) == ("one", "unknown_service_id")


@db
async def test_a_run_that_nothing_stops_has_nothing_dropped(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path)
    progress = await rig.start(CASES)
    assert (progress.done, progress.stopped, progress.dropped) == (3, None, ())
    assert rig.requests == ["classifier"] * 4


# --- A case the rules decide says nothing about the provider (fix round 2, re-review N-1) ---------
#
# An emergency, clinical advice or personal data is answered by the rules: the case makes no
# request. It cannot show that the provider answers, so it does not release the failures held before
# it; it waits behind them, and is written with them when a case the model answered completes.


def test_a_call_that_came_back_with_usage_is_a_case_the_model_answered() -> None:
    answered = call(input_tokens=900, output_tokens=40, finish_reason="stop")
    cut_off = call(input_tokens=0, output_tokens=0, finish_reason="unconfirmed", model_version=None)
    assert answered_by_the_model([record("a", "ciao", calls=(answered,))]) is True
    assert answered_by_the_model([record("a", "ciao", calls=())]) is False  # the rules decided
    assert answered_by_the_model([record("a", "ciao", calls=(cut_off,))]) is False  # no usage
    assert answered_by_the_model([record("a", "ciao", calls=(cut_off, answered))]) is True


def test_a_conversation_is_answered_if_the_model_answered_any_of_its_turns() -> None:
    first = record("a", "ciao", turn=1, turns=2, calls=())
    second = record("a", "ciao", turn=2, turns=2, calls=(call(),))
    assert answered_by_the_model([first, second]) is True
    assert answered_by_the_model([first, record("a", "ciao", turn=2, turns=2, calls=())]) is False


@db
async def test_a_case_the_rules_decide_does_not_release_a_failure_held_before_it(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    # Probes H7 and H12: "a" fails, "e" is decided by the rules, and the run stops in "b".
    cases = [*_single_turn_cases("a"), _rules_only_case("e"), *_single_turn_cases("b")]
    on_request = {1: _reset, 2: _rate_limit, 3: _rate_limit, 4: _rate_limit, 5: _rate_limit}
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request=on_request)

    stopped = await rig.start(cases)

    assert (stopped.done, stopped.stopped) == (0, "rate_limited")
    assert stopped.dropped == ("a",)  # the failure is named; "e" is no failure
    assert read_turns(rig.files) == ()  # and "e" waited behind it: it is not recorded either

    resumed = await rig.start(cases)  # the provider answers again; "e" costs nothing to run again

    assert (resumed.done, resumed.stopped, resumed.dropped) == (3, None, ())
    assert _rules(read_turns(rig.files)) == [
        ("a", 1, "pass"),
        ("e", 1, "emergency"),
        ("b", 1, "pass"),
    ]


@db
async def test_a_case_the_model_answered_releases_the_failure_and_the_rules_only_cases_in_order(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    cases = [*_single_turn_cases("a"), _rules_only_case("e"), *_single_turn_cases("b")]
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={1: _reset})

    progress = await rig.start(cases)

    assert (progress.done, progress.stopped, progress.dropped) == (3, None, ())
    # "a" failed on its own (the model answered "b"): it is evidence, and "e" is written with it,
    # in the order of the cases.
    assert _rules(read_turns(rig.files)) == [
        ("a", 1, "error"),
        ("e", 1, "emergency"),
        ("b", 1, "pass"),
    ]


@db
async def test_failures_and_rules_only_cases_are_written_one_by_one_when_a_model_answer_comes(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[list[str]] = []
    real = runner.append_case

    def spy(files: RunFiles, records: Any) -> None:
        writes.append([each.case_id for each in records])
        real(files, records)

    monkeypatch.setattr(runner, "append_case", spy)
    cases = [
        *_single_turn_cases("a"),
        _rules_only_case("e1"),
        *_single_turn_cases("b"),
        _rules_only_case("e2"),
        *_single_turn_cases("c"),
    ]
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={1: _reset, 2: _reset})

    progress = await rig.start(cases)

    assert progress.done == 5
    # "a" and "b" failed, "c" is the first case the model answered: all five are written then, one
    # write for each case, in the order of the cases.
    assert writes == [["a"], ["e1"], ["b"], ["e2"], ["c"]]


@db
async def test_a_stop_names_the_failures_it_dropped_and_not_the_rules_only_cases_behind_them(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    cases = [
        *_single_turn_cases("a"),
        _rules_only_case("e"),
        *_single_turn_cases("b", "c"),
    ]
    rig = Rig(
        clinic_pack,
        activate_pack(clinic_pack),
        tmp_path,
        on_request={1: _reset, 2: _reset, 3: _reset},
    )

    stopped = await rig.start(cases, breaker=3)

    assert (stopped.done, stopped.stopped) == (0, "unavailable")
    assert stopped.dropped == ("a", "b")
    assert read_turns(rig.files) == ()


@db
async def test_a_case_the_rules_decide_is_written_at_once_when_no_failure_is_held(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    cases = [_rules_only_case("e"), *_single_turn_cases("b")]
    on_request = {1: _rate_limit, 2: _rate_limit, 3: _rate_limit, 4: _rate_limit}
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request=on_request)

    stopped = await rig.start(cases)

    assert (stopped.done, stopped.stopped, stopped.dropped) == (1, "rate_limited", ())
    assert _rules(read_turns(rig.files)) == [("e", 1, "emergency")]


@db
async def test_the_rules_only_cases_held_at_the_end_of_the_run_are_written_with_the_failure(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    cases = [*_single_turn_cases("a"), _rules_only_case("e")]
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={1: _reset})

    progress = await rig.start(cases)  # no model answer follows the failure: the run just ends

    assert (progress.done, progress.stopped, progress.dropped) == (2, None, ())
    assert _rules(read_turns(rig.files)) == [("a", 1, "error"), ("e", 1, "emergency")]


@db
async def test_the_offline_route_holds_nothing_back_it_can_have_no_outage(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    requests: list[str] = []

    def classifier(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append("classifier")
        if len(requests) == 1:
            raise httpx2.ReadError("connection reset by peer")  # the first case ends as an error
        raise asyncio.CancelledError  # and Ctrl-C reaches the second

    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), FunctionModel(classifier), document_responder()
    )
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()

    with pytest.raises(asyncio.CancelledError):
        await run_live_eval(
            runtime, _single_turn_cases("a", "b"), files=files, run_id="run-1", meter=None
        )

    # Without a meter there is no provider to be down: the failure of "a" is the pipeline's own,
    # and it was written before "b" was run.
    assert _rules(read_turns(files)) == [("a", 1, "error")]
