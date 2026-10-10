"""A run that can no longer count what it spends stops; what the stop cut runs again (M3 design §6).

Both tests reproduce the review's probes on the real turn code: a ledger that is damaged, or made
read-only, while the run is under way.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.records import RunFiles, read_turns
from frontdesk.live_eval.runner import RunProgress, run_live_eval
from frontdesk.model_gateway.metering import Meter
from frontdesk.tenancy.pack import BusinessPack
from tests.conversation.helpers import document_responder, make_runtime
from tests.model_gateway.budget_helpers import ROUTE, write_budget

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
CAP = {"max_tokens": 600}
CASES = [
    TurnCase.model_validate(spec)
    for spec in (
        {"id": "one", "message": "Quanto costa la fisioterapia?", "expect": {}},
        {
            "id": "two",
            "messages": ["Quanto costa la fisioterapia?", "E per favore?"],
            "expect": {},
        },
        {"id": "three", "message": "Quanto costa la fisioterapia?", "expect": {}},
    )
]
GARBAGE = "not json"


class Rig:
    """One run directory and one ledger, started again as often as a test needs. Each start has a
    new meter, as a new process would; the classifier counts its requests, and `on_request` runs
    something at the n-th one (it happens while that request is away)."""

    def __init__(
        self,
        pack: BusinessPack,
        factory: sessionmaker[Session],
        tmp_path: Path,
        on_request: dict[int, Callable[[], None]] | None = None,
    ) -> None:
        self.pack = pack
        self.factory = factory
        self.budget = write_budget(tmp_path)
        self.files = RunFiles(tmp_path / "run-1")
        self.files.directory.mkdir()
        self.requests: list[str] = []
        self._on_request = on_request or {}

    def _classifier(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append("classifier")
        if (action := self._on_request.get(len(self.requests))) is not None:
            action()
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, PRICE)])

    async def start(
        self, cases: list[TurnCase] = CASES, *, limit: int | None = None
    ) -> RunProgress:
        meter = Meter(self.budget, route_id=ROUTE, run_id="run-1")
        runtime = make_runtime(
            self.pack,
            self.factory,
            meter.wrap(FunctionModel(self._classifier), "classifier"),
            meter.wrap(document_responder(), "responder"),
            model_settings=CAP,
        )
        return await run_live_eval(
            runtime, cases, files=self.files, run_id="run-1", meter=meter, limit=limit
        )

    def ledger_lines(self) -> list[str]:
        return self.budget.ledger_path.read_text(encoding="utf-8").splitlines()


async def test_a_ledger_damaged_during_the_run_stops_it_and_the_cut_case_runs_again(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    def damage() -> None:
        with rig.budget.ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(GARBAGE + "\n")

    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={2: damage})
    first = await rig.start()
    assert (first.done, first.total, first.stopped) == (1, 3, "ledger")
    assert len(rig.requests) == 2  # nothing was sent once the damage could be seen
    assert [record.case_id for record in read_turns(rig.files)] == ["one"]  # "two" was cut

    repaired = [line for line in rig.ledger_lines() if line != GARBAGE]
    rig.budget.ledger_path.write_text("\n".join(repaired) + "\n", encoding="utf-8")
    second = await rig.start()
    assert (second.done, second.stopped) == (3, None)
    assert len(rig.requests) == 2 + 3  # "two" ran again in full (2 turns), then "three"
    records = read_turns(rig.files)
    assert [record.rule for record in records] == ["pass"] * len(records)  # no turn of a failure
    assert [(record.case_id, record.turn) for record in records] == [
        ("one", 1),
        ("two", 1),
        ("two", 2),
        ("three", 1),
    ]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only file")
async def test_a_ledger_made_read_only_during_the_run_stops_it_with_one_request_uncounted(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    def make_read_only() -> None:
        rig.budget.ledger_path.chmod(0o444)

    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path, on_request={2: make_read_only})
    try:
        first = await rig.start()
    finally:
        rig.budget.ledger_path.chmod(0o644)
    assert (first.done, first.total, first.stopped) == (1, 3, "ledger")
    # "two" asked once: the request was paid, but its line could not be written. Then the run
    # stopped; before, the three requests that followed were paid and uncounted too.
    assert len(rig.requests) == 2
    assert len(rig.ledger_lines()) == 1
    assert [record.case_id for record in read_turns(rig.files)] == ["one"]

    second = await rig.start()
    assert (second.done, second.stopped) == (3, None)
    assert len(rig.requests) == 2 + 3
    assert len(rig.ledger_lines()) == 1 + 3  # one request in all stays uncounted


async def test_a_case_whose_last_line_lost_its_newline_is_not_counted_and_runs_again(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    # The review's probe: the reader counted that case, the next append cut it, and the run
    # reported every case done while the file held one fewer.
    one, two, three = CASES
    rig = Rig(clinic_pack, activate_pack(clinic_pack), tmp_path)
    order = [one, three, two]
    first = await rig.start(order, limit=2)  # "one" and "three" are recorded
    assert (first.done, first.stopped, len(rig.requests)) == (2, None, 2)
    raw = rig.files.turns.read_bytes()
    rig.files.turns.write_bytes(raw.removesuffix(b"\n"))  # the write of "three" was cut at its end

    second = await rig.start(order)
    assert (second.done, second.total, second.stopped) == (3, 3, None)
    assert len(rig.requests) == 2 + 1 + 2  # "three" ran again, then "two" (2 turns)
    assert [(r.case_id, r.turn) for r in read_turns(rig.files)] == [
        ("one", 1),
        ("three", 1),
        ("two", 1),
        ("two", 2),
    ]  # every case the run reports is on disk
