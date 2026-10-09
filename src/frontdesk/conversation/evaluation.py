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
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

from frontdesk.agent.vocab import Intent
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, handle_message
from frontdesk.safety.rules import contains_diagnostic_language
from frontdesk.tenancy.pack import FixedTextKey, Language


class TurnDatasetError(ValueError):
    """A turn dataset cannot be read, or some of its cases are invalid."""


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
    # A fresh case starts at turn 0, so its first reply carries the disclosure (spec §5); any
    # other case starts as a later turn.
    fresh: bool = False
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

    @model_validator(mode="after")
    def _fresh_has_no_history(self) -> TurnCase:
        if self.fresh and (self.current_language is not None or self.previous_intents):
            raise ValueError(f"case '{self.id}' is fresh, so it has no earlier language or intents")
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
        """At least one case ran and every case passed: a run that checked nothing is no pass."""
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    def by_suite(self) -> dict[str, tuple[int, int]]:
        return _grouped(self.results, lambda result: [result.suite] if result.suite else [])

    def by_language(self) -> dict[str, tuple[int, int]]:
        return _grouped(self.results, lambda result: [result.language])


def load_turn_cases(path: Path) -> list[TurnCase]:
    """The cases of a dataset file; every problem in it is reported at once, naming file and case."""
    cases: list[TurnCase] = []
    problems: list[str] = []
    positions: dict[str, int] = {}
    for position, item in enumerate(_read_items(path), start=1):
        try:
            case = TurnCase.model_validate(item)
        except ValidationError as error:
            problems.append(f"{path}: {_label(position, item)}: {_describe(error)}")
            continue
        if case.id in positions:
            problems.append(
                f"{path}: case '{case.id}': duplicate id, case #{positions[case.id]} has it too"
            )
            continue
        positions[case.id] = position
        cases.append(case)
    if problems:
        raise TurnDatasetError("\n".join(problems))
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
    failures += [
        f"missing text: {_one_line(text)}" for text in expect.contains if text not in result.text
    ]
    failures += [
        f"unexpected text: {_one_line(text)}" for text in expect.not_contains if text in result.text
    ]
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
            turns=0 if case.fresh else 1,
            language=case.current_language,
            previous_intents=case.previous_intents,
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


def _one_line(text: str) -> str:
    """A text from a dataset with its line breaks spelled out, so a reason never splits a row."""
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _read_items(path: Path) -> list[Any]:
    """The entries of a dataset file; a file that is not a non-empty list of them is an error."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TurnDatasetError(f"{path}: cannot be read ({error.strerror or error})") from error
    except UnicodeDecodeError as error:
        raise TurnDatasetError(f"{path}: not valid UTF-8 text (save the file as UTF-8)") from error
    except yaml.YAMLError as error:
        raise TurnDatasetError(f"{path}: invalid YAML{_where(error)}") from error
    if not raw:
        raise TurnDatasetError(f"{path}: no cases")
    if not isinstance(raw, list):
        raise TurnDatasetError(f"{path}: expected a list of cases, one '- id: ...' entry for each")
    return raw


def _where(error: yaml.YAMLError) -> str:
    """` at line L, column C: reason` for an error PyYAML placed in the text; else `: reason`."""
    if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
        mark = error.problem_mark
        return f" at line {mark.line + 1}, column {mark.column + 1}: {error.problem}"
    return ": " + " ".join(str(error).split())


def _label(position: int, item: object) -> str:
    """How an error names a case: by its id when it has a usable one, else by its position."""
    case_id = item.get("id") if isinstance(item, dict) else None
    if isinstance(case_id, str) and case_id.strip():
        return f"case '{case_id}'"
    return f"case #{position}"


def _describe(error: ValidationError) -> str:
    """The problems of one case, each with its field: `expect.outcome: Input should be ...`."""
    return "; ".join(
        _problem(problem) for problem in error.errors(include_url=False, include_input=False)
    )


def _problem(problem: ErrorDetails) -> str:
    field = ".".join(str(part) for part in problem["loc"])
    return f"{field}: {problem['msg']}" if field else problem["msg"]
