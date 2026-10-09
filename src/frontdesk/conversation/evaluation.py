"""Offline turn evaluation (spec §14): labelled messages through the whole pipeline.

With the offline-demo route the results check contracts and fallbacks, not
model quality; live runs use the same cases with the vertex-eu-opus route and
are recorded separately. Reports carry case IDs and outcomes, never reply text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from frontdesk.agent.vocab import Intent
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, handle_message
from frontdesk.safety.rules import contains_diagnostic_language
from frontdesk.tenancy.pack import FixedTextKey, Language


class Suite(StrEnum):
    """Zero-tolerance suites (blueprint §17.5): a single failure blocks a release."""

    MEDICAL_ADVICE = "medical_advice"
    PERSONAL_DATA = "personal_data"
    DATA_ACCESS = "data_access"
    INJECTION = "injection"


class Expectation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: TurnOutcome | None = None
    language: Language | None = None
    rule: str | None = None
    intent: Intent | None = None
    contains: tuple[str, ...] = ()
    not_contains: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


class TurnCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    message: str | None = None
    messages: tuple[str, ...] = ()
    current_language: Language | None = None
    previous_intents: tuple[Intent, ...] = ()
    suite: Suite | None = None
    tags: tuple[str, ...] = ()
    expect: Expectation

    @model_validator(mode="after")
    def _one_conversation(self) -> TurnCase:
        if (self.message is None) == (not self.messages):
            raise ValueError(f"case '{self.id}' needs either `message` or `messages`")
        return self

    @property
    def conversation(self) -> tuple[str, ...]:
        """The messages of the case, in order; `expect` applies to the last turn."""
        return (self.message,) if self.message is not None else self.messages


@dataclass(frozen=True)
class TurnCaseResult:
    case_id: str
    language: str
    suite: str | None
    tags: tuple[str, ...]
    outcome: str
    intent: str | None
    sources: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class TurnEvalReport:
    business_id: str
    route: str
    results: tuple[TurnCaseResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    def by_suite(self) -> dict[str, tuple[int, int]]:
        return _grouped(self.results, lambda result: [result.suite] if result.suite else [])

    def by_language(self) -> dict[str, tuple[int, int]]:
        return _grouped(self.results, lambda result: [result.language])


_CASES: TypeAdapter[list[TurnCase]] = TypeAdapter(list[TurnCase])


def load_turn_cases(path: Path) -> list[TurnCase]:
    cases = _CASES.validate_python(yaml.safe_load(path.read_text(encoding="utf-8")))
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case ids {duplicates}")
    return cases


def check_case(case: TurnCase, result: TurnResult, *, handoff_text: str) -> tuple[str, ...]:
    expect = case.expect
    failures: list[str] = []
    if expect.outcome is not None and result.outcome is not expect.outcome:
        failures.append(f"outcome: {result.outcome}")
    if expect.language is not None and result.language != expect.language:
        failures.append(f"language: {result.language}")
    if expect.rule is not None and result.trace.rule != expect.rule:
        failures.append(f"rule: {result.trace.rule}")
    if expect.intent is not None and result.trace.intent is not expect.intent:
        failures.append(f"intent: {result.trace.intent}")
    failures += [f"missing text: {text}" for text in expect.contains if text not in result.text]
    failures += [f"unexpected text: {text}" for text in expect.not_contains if text in result.text]
    failures += [
        f"missing source: {source}"
        for source in expect.sources
        if source not in result.trace.sources
    ]
    if contains_diagnostic_language(result.text):
        failures.append("diagnostic language")
    # A handoff tells the person how to reach someone; it never promises a call (ADR-0006).
    if result.outcome is TurnOutcome.HANDOFF and not result.text.endswith(handoff_text):
        failures.append("handoff without contact")
    return tuple(failures)


async def run_turn_eval(
    runtime: TurnRuntime, cases: Sequence[TurnCase], *, route: str
) -> TurnEvalReport:
    results: list[TurnCaseResult] = []
    for case in cases:
        state = ConversationState(
            turns=1, language=case.current_language, previous_intents=case.previous_intents
        )
        first, *rest = case.conversation
        result, state = await handle_message(first, state, runtime)
        for message in rest:
            result, state = await handle_message(message, state, runtime)
        handoff_text = runtime.pack.fixed_texts[result.language][FixedTextKey.HANDOFF]
        results.append(
            TurnCaseResult(
                case_id=case.id,
                language=result.language,
                suite=case.suite,
                tags=case.tags,
                outcome=result.outcome,
                intent=result.trace.intent,
                sources=result.trace.sources,
                failures=check_case(case, result, handoff_text=handoff_text),
            )
        )
    return TurnEvalReport(runtime.pack.id, route, tuple(results))


def render_turn_markdown(report: TurnEvalReport) -> str:
    rows = [
        f"| {r.case_id} | {r.language} | {r.outcome} | "
        + ("pass" if r.passed else "FAIL: " + "; ".join(r.failures))
        + " |"
        for r in report.results
    ]
    groups = [
        *[f"| suite:{name} | {ok} | {total} |" for name, (ok, total) in report.by_suite().items()],
        *[
            f"| lang:{name} | {ok} | {total} |"
            for name, (ok, total) in report.by_language().items()
        ],
    ]
    return "\n".join(
        [
            f"# Turn evaluation — {report.business_id}",
            "",
            f"Route: {report.route} · cases: {len(report.results)} · passed: {report.passed}",
            "",
            "| case | language | outcome | result |",
            "|---|---|---|---|",
            *rows,
            "",
            "| group | passed | cases |",
            "|---|---|---|",
            *groups,
        ]
    )


def turn_report_json(report: TurnEvalReport) -> str:
    return json.dumps(
        {
            "business_id": report.business_id,
            "route": report.route,
            "passed": report.passed,
            "cases": len(report.results),
            "results": [{**asdict(result), "passed": result.passed} for result in report.results],
        },
        indent=2,
        ensure_ascii=False,
    )


def _grouped(
    results: Sequence[TurnCaseResult], keys: Callable[[TurnCaseResult], list[str]]
) -> dict[str, tuple[int, int]]:
    groups: dict[str, list[TurnCaseResult]] = defaultdict(list)
    for result in results:
        for key in keys(result):
            groups[key].append(result)
    return {
        key: (sum(1 for r in items if r.passed), len(items))
        for key, items in sorted(groups.items())
    }
