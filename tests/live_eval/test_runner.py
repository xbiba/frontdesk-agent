"""A live run with fake models and no network: records, stops and resumes (M3 design §4.4, §6)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.live_eval.records import RunFiles, completed_cases, read_turns
from frontdesk.live_eval.runner import run_live_eval
from frontdesk.model_gateway.costs import Budget
from frontdesk.model_gateway.metering import Meter
from frontdesk.tenancy.pack import BusinessPack
from tests.conversation.helpers import classifier_model, document_responder, make_runtime
from tests.model_gateway.budget_helpers import ROUTE, write_budget

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
PARKING = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "Is there free parking?",
}


def _cases(*specs: dict[str, Any]) -> list[TurnCase]:
    return [TurnCase.model_validate(spec) for spec in specs]


CASES = _cases(
    {"id": "one", "message": "Quanto costa la fisioterapia?", "expect": {"contains": ["30 EUR"]}},
    {
        "id": "two",
        "messages": ["Quanto costa la fisioterapia?", "E per favore?"],
        "expect": {"contains": ["30 EUR"]},
    },
    {"id": "three", "message": "Quanto costa la fisioterapia?", "expect": {}},
)


async def test_every_turn_is_recorded_with_its_calls_and_time(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    budget = write_budget(tmp_path)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        meter.wrap(classifier_model(PRICE), "classifier"),
        meter.wrap(document_responder(), "responder"),
    )
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    progress = await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=meter)
    assert (progress.done, progress.total, progress.stopped) == (3, 3, None)
    records = read_turns(files)
    assert [(r.case_id, r.turn, r.turns) for r in records] == [
        ("one", 1, 1),
        ("two", 1, 2),
        ("two", 2, 2),
        ("three", 1, 1),
    ]
    first = records[0]
    assert (first.outcome, first.rule, first.intent, first.language) == (
        "replied",
        "pass",
        "price",
        "it",
    )
    assert first.reply.startswith("Fisioterapia: 30 EUR")
    assert [call.agent for call in first.calls] == ["classifier"]
    assert first.usd > 0
    assert first.seconds >= 0
    assert budget.spent_usd() == pytest.approx(sum(r.usd for r in records))


async def test_a_resumed_run_makes_no_call_for_a_recorded_case(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    factory = activate_pack(clinic_pack)
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    prompts: list[str] = []  # one per classifier request
    runtime = make_runtime(
        clinic_pack, factory, classifier_model(PRICE, prompts), document_responder()
    )
    first = await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=None, limit=1)
    assert (first.done, len(prompts)) == (1, 1)
    second = await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=None)
    assert second.done == 3
    assert len(prompts) == 1 + 3  # "two" has two turns; "one" ran once only
    assert set(completed_cases(read_turns(files))) == {"one", "two", "three"}


async def test_the_budget_stops_the_run_and_the_cut_case_leaves_no_turn(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    # Each classifier request costs 0.03 USD and reserves 0.052 (32,000 tokens in, 2,000 out):
    # with a cap of 0.10 the third request no longer fits, in the middle of case "two".
    def classifier(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call = ToolCallPart(info.output_tools[0].name, PRICE)
        return ModelResponse(parts=[call], usage=RequestUsage(input_tokens=30_000))

    budget = write_budget(tmp_path, cap_eur=0.10, usd_per_eur=1.0)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        meter.wrap(FunctionModel(classifier), "classifier"),
        meter.wrap(document_responder(), "responder"),
    )
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    progress = await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=meter)
    assert (progress.done, progress.stopped) == (1, "budget")
    assert set(completed_cases(read_turns(files))) == {"one"}  # "two" was cut: none of it stays
    assert budget.spent_usd() == pytest.approx(0.06)  # the refused request cost nothing
    assert budget.spent_usd() <= budget.cap_usd


async def test_a_run_stopped_by_the_budget_resumes_under_a_larger_one_and_loses_nothing(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    requests: list[str] = []

    def classifier(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append("classifier")  # each costs 0.03 USD, as in the test above
        call = ToolCallPart(info.output_tools[0].name, PRICE)
        return ModelResponse(parts=[call], usage=RequestUsage(input_tokens=30_000))

    factory = activate_pack(clinic_pack)
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()

    def runtime_and_meter(budget: Budget) -> tuple[TurnRuntime, Meter]:
        meter = Meter(budget, route_id=ROUTE, run_id="run-1")
        model = meter.wrap(FunctionModel(classifier), "classifier")
        return make_runtime(clinic_pack, factory, model, document_responder()), meter

    small = write_budget(tmp_path, cap_eur=0.10, usd_per_eur=1.0)
    runtime, meter = runtime_and_meter(small)
    first = await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=meter)
    assert (first.done, first.stopped, len(requests)) == (1, "budget", 2)  # "two" was cut

    # A larger cap in the same budget file: the ledger beside it still holds the first run's spend.
    larger = write_budget(tmp_path, cap_eur=1.0, usd_per_eur=1.0)
    assert larger.spent_usd() == pytest.approx(0.06)
    runtime, meter = runtime_and_meter(larger)
    second = await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=meter)
    assert (second.done, second.total, second.stopped) == (3, 3, None)
    assert len(requests) == 2 + 2 + 1  # "one" was not run again; "two" ran in full, then "three"
    records = read_turns(files)
    assert [(r.case_id, r.turn) for r in records] == [
        ("one", 1),
        ("two", 1),
        ("two", 2),
        ("three", 1),
    ]
    # The ledger holds every request that was paid for, the cut case's first one included.
    assert larger.spent_usd() == pytest.approx(0.03 * len(requests))
    assert sum(r.usd for r in records) == pytest.approx(0.03 * (len(requests) - 1))


async def test_a_rate_limit_stops_the_run_after_the_recorded_cases(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    answers = [PRICE]

    def classifier(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not answers:
            raise ModelHTTPError(429, "gemini-3.1-flash-lite")
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answers.pop())])

    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        meter.wrap(FunctionModel(classifier), "classifier"),
        meter.wrap(document_responder(), "responder"),
    )
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    progress = await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=meter)
    assert (progress.done, progress.stopped) == (1, "rate_limited")
    assert [r.case_id for r in read_turns(files)] == ["one"]


async def test_a_refused_answer_is_recorded_for_the_review(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    def invented(passage_id: str, content: str) -> dict[str, str]:
        quote = "free parking spaces"
        return {
            "text": "There are fifty free parking spaces.",
            "evidence_id": passage_id,
            "quote": quote,
        }

    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING),
        document_responder(claim=invented),
    )
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    case = _cases({"id": "parking", "message": "Is there free parking?", "expect": {}})
    await run_live_eval(runtime, case, files=files, run_id="run-1", meter=None)
    [record] = read_turns(files)
    assert record.outcome == "handoff"
    assert "number_not_in_quote" in record.check_reasons
    assert record.draft is not None
    assert record.draft.claims[0].text == "There are fifty free parking spaces."


async def test_a_run_cancelled_in_the_middle_of_a_case_keeps_only_whole_cases(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    answers = [PRICE, PRICE]

    def classifier(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not answers:
            raise asyncio.CancelledError  # Ctrl-C reaches a running turn as a cancellation
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answers.pop())])

    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), FunctionModel(classifier), document_responder()
    )
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    with pytest.raises(asyncio.CancelledError):
        await run_live_eval(runtime, CASES, files=files, run_id="run-1", meter=None)
    assert set(completed_cases(read_turns(files))) == {"one"}  # "two" ran one turn of two
    assert [r.case_id for r in read_turns(files)] == ["one"]
