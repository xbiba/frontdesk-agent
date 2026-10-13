"""The files of a live run: one JSON line per turn, and the run's manifest (M3 design §4.4).

A run directory holds `run.json` (what the run is: route, cases, knowledge and prompt versions,
the fixed texts it was judged with), `turns.jsonl` (one line per turn, written a whole case at a
time), and the reports. The texts are synthetic questions and the bot's replies; no personal data.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from frontdesk.tenancy.pack import FixedTextKey, Language

MANIFEST = "run.json"
TURNS = "turns.jsonl"


class RunFileError(ValueError):
    """A run directory's file is missing, damaged, or belongs to another run."""


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CallRow(_Row):
    agent: str
    model_version: str | None
    input_tokens: int
    output_tokens: int
    thoughts_tokens: int
    finish_reason: str | None
    usd: float
    seconds: float


class ClaimRow(_Row):
    text: str
    evidence_id: str
    quote: str


class DraftRow(_Row):
    action: str
    clarify: str | None
    claims: tuple[ClaimRow, ...]


class TurnRecord(_Row):
    run_id: str
    attempt: str  # one run of the case; a case run again after a stop gets a new one
    case_id: str
    turn: int  # 1 for the first message of the case
    turns: int
    message: str
    reply: str
    language: Language
    outcome: str
    rule: str
    category: str | None
    intent: str | None
    rejection: str | None
    check_reasons: tuple[str, ...]
    sources: tuple[str, ...]
    draft: DraftRow | None
    calls: tuple[CallRow, ...]
    seconds: float
    at: datetime

    @property
    def usd(self) -> float:
        return sum(call.usd for call in self.calls)


class RunManifest(_Row):
    run_id: str
    business_id: str
    route: str
    model_name: str
    engine_commit: str
    cases_file: str
    cases_digest: str
    knowledge_version: str
    classifier_prompt: str
    responder_prompt: str
    # The reception agent's prompt, when it led the run (FRONTDESK_CONVERSATION=agent).
    reception_prompt: str | None = None
    fixed_texts: dict[Language, dict[FixedTextKey, str]]
    budget_file: str | None
    # The checks' thresholds (settings): a run judged with other ones is another run. A run made
    # before they were recorded reads None, so it cannot be resumed by a newer engine.
    min_support_similarity: float | None = None
    min_retrieval_similarity: float | None = None
    started_at: datetime


# What must match for a run to be resumed: the rest (the start time) may differ.
_IDENTITY = (
    "business_id",
    "route",
    "model_name",
    "engine_commit",
    "cases_digest",
    "knowledge_version",
    "classifier_prompt",
    "responder_prompt",
    "reception_prompt",
    "fixed_texts",
    "min_support_similarity",
    "min_retrieval_similarity",
)


@dataclass(frozen=True)
class RunFiles:
    directory: Path

    @property
    def manifest(self) -> Path:
        return self.directory / MANIFEST

    @property
    def turns(self) -> Path:
        return self.directory / TURNS


def write_manifest(files: RunFiles, manifest: RunManifest) -> None:
    files.directory.mkdir(parents=True, exist_ok=True)
    files.manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_manifest(files: RunFiles) -> RunManifest:
    try:
        return RunManifest.model_validate_json(files.manifest.read_text(encoding="utf-8"))
    except OSError as error:
        raise RunFileError(
            f"{files.manifest}: cannot be read ({error.strerror or error})"
        ) from error
    except ValidationError as error:
        raise RunFileError(f"{files.manifest}: not a run manifest") from error


def manifest_differences(recorded: RunManifest, current: RunManifest) -> list[str]:
    """The identity fields on which `current` differs from the run on disk."""
    return [name for name in _IDENTITY if getattr(recorded, name) != getattr(current, name)]


def append_case(files: RunFiles, records: Sequence[TurnRecord]) -> None:
    """Append the turns of one case in a single write and make them durable.

    A stop in the middle of an earlier write can leave a last line without its newline. It is cut
    off first, so that the first line of this case does not begin in the middle of it.
    """
    _cut_unfinished_line(files.turns)
    lines = "".join(record.model_dump_json() + "\n" for record in records)
    with files.turns.open("a", encoding="utf-8") as turns:
        turns.write(lines)
        turns.flush()
        os.fsync(turns.fileno())


def _cut_unfinished_line(path: Path) -> None:
    """Drop what follows the last newline, even when it reads as a whole record: `read_turns` does
    not count it either, so the case it belonged to runs again.

    The file is only shortened, in one step: there is no instant at which its finished cases are
    missing, whatever ends the process in the middle of it.
    """
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        os.truncate(path, data.rfind(b"\n") + 1)


def _lines(data: bytes) -> list[bytes]:
    """The finished lines of a JSON Lines file: the pieces that end in a newline.

    They are cut at newlines only: JSON lets U+2028, U+2029 and U+0085 stand raw inside a string,
    where `splitlines` would cut a valid line in two. What follows the last newline (nothing, when
    the file ends well) is a write that a stop cut short, not a line yet.
    """
    return data.split(b"\n")[:-1]


def read_turns(files: RunFiles) -> tuple[TurnRecord, ...]:
    """Every turn on disk. What follows the last newline, the write a stop cut short, is left out;
    a damaged line before it is an error."""
    if not files.turns.exists():
        return ()
    records: list[TurnRecord] = []
    for number, line in enumerate(_lines(files.turns.read_bytes()), start=1):
        if not line.strip():
            continue
        try:
            records.append(TurnRecord.model_validate_json(line))
        except ValidationError as error:
            raise RunFileError(f"{files.turns}: line {number} is not a turn record") from error
    return tuple(records)


def completed_cases(records: Iterable[TurnRecord]) -> dict[str, tuple[TurnRecord, ...]]:
    """For each case, the turns of its latest attempt that has all of them, in order."""
    attempts: dict[tuple[str, str], list[TurnRecord]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for record in records:
        key = (record.case_id, record.attempt)
        if key not in attempts:
            order.append(key)
        attempts[key].append(record)
    complete: dict[str, tuple[TurnRecord, ...]] = {}
    for key in order:
        turns = sorted(attempts[key], key=lambda record: record.turn)
        if [record.turn for record in turns] == list(range(1, turns[0].turns + 1)):
            complete[key[0]] = tuple(turns)
    return complete
