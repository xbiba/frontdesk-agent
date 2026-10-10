"""The run files: one run per directory, whole cases only."""

from __future__ import annotations

import io
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from frontdesk.live_eval.records import (
    RunFileError,
    RunFiles,
    RunManifest,
    append_case,
    completed_cases,
    manifest_differences,
    read_manifest,
    read_turns,
    write_manifest,
)
from tests.live_eval.helpers import AT, TEXTS, manifest, record


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"business_id": "demo-shop"}, "business_id"),
        ({"route": "offline-demo"}, "route"),
        ({"model_name": "gemini-3.1-flash"}, "model_name"),
        ({"engine_commit": "def5678"}, "engine_commit"),
        # The guard against resuming on a cases file that was edited since the first call.
        ({"cases_digest": "e" * 64}, "cases_digest"),
        ({"knowledge_version": "kv-2"}, "knowledge_version"),
        ({"classifier_prompt": "x" * 12}, "classifier_prompt"),
        ({"responder_prompt": "x" * 12}, "responder_prompt"),
        ({"fixed_texts": {"it": {**TEXTS, "handoff": "Chiama il 112."}}}, "fixed_texts"),
    ],
)
def test_a_run_resumes_only_with_the_same_knowledge_prompts_texts_route_and_engine(
    change: dict[str, object], field: str
) -> None:
    assert manifest_differences(manifest(), manifest(**change)) == [field]
    assert manifest_differences(manifest(), manifest(started_at=manifest().started_at)) == []


def test_turns_are_read_back_in_order(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    append_case(files, [record("a", "uno")])
    append_case(files, [record("b", "due", turn=1, turns=2), record("b", "tre", turn=2, turns=2)])
    assert [(r.case_id, r.turn) for r in read_turns(files)] == [("a", 1), ("b", 1), ("b", 2)]


def test_a_line_cut_by_a_stop_in_the_middle_of_a_write_is_left_out(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    append_case(files, [record("a", "uno")])
    with files.turns.open("a", encoding="utf-8") as turns:
        turns.write('{"run_id": "run-1", "case_id": "b", "tu')
    assert [r.case_id for r in read_turns(files)] == ["a"]


def test_a_damaged_line_before_the_last_is_an_error(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    files.turns.write_text("not json\n" + record("a", "uno").model_dump_json() + "\n")
    with pytest.raises(ValueError, match="line 1"):
        read_turns(files)


def test_blank_lines_are_not_turns(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    files.turns.write_text("\n" + record("a", "uno").model_dump_json() + "\n\n", encoding="utf-8")
    assert [r.case_id for r in read_turns(files)] == ["a"]


def test_only_the_latest_attempt_with_every_turn_counts(tmp_path: Path) -> None:
    cut = record("b", "due", attempt="first", turn=1, turns=2)
    whole = [
        record("b", "due", attempt="second", turn=1, turns=2),
        record("b", "tre", attempt="second", turn=2, turns=2),
    ]
    complete = completed_cases([cut, *whole])
    assert [r.attempt for r in complete["b"]] == ["second", "second"]
    assert "b" not in completed_cases([cut])


@pytest.mark.parametrize(
    "separator",
    [
        pytest.param("\N{LINE SEPARATOR}", id="line-separator"),
        pytest.param("\N{PARAGRAPH SEPARATOR}", id="paragraph-separator"),
        pytest.param("\N{NEXT LINE}", id="next-line"),
    ],
)
def test_a_message_with_a_unicode_line_break_stays_in_its_line(
    tmp_path: Path, separator: str
) -> None:
    # JSON allows these characters raw inside a string, but `str.splitlines` would cut at them.
    files = RunFiles(tmp_path)
    message = f"Is there{separator}parking?"
    append_case(files, [record("a", f"Yes{separator}there is", message=message)])
    append_case(files, [record("b", "due")])
    first, second = read_turns(files)
    assert (first.message, first.reply) == (message, f"Yes{separator}there is")
    assert second.case_id == "b"


def test_a_case_written_after_a_cut_line_does_not_begin_inside_it(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    append_case(files, [record("a", "uno")])
    with files.turns.open("a", encoding="utf-8") as turns:
        turns.write('{"run_id": "run-1", "case_id": "b", "tu')
    append_case(files, [record("b", "due", turn=1, turns=2), record("b", "tre", turn=2, turns=2)])
    assert [(r.case_id, r.turn) for r in read_turns(files)] == [("a", 1), ("b", 1), ("b", 2)]
    assert set(completed_cases(read_turns(files))) == {"a", "b"}


def test_a_line_cut_inside_a_character_is_left_out(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    append_case(files, [record("a", "uno")])
    cut = '{"run_id": "run-1", "reply": "caff\N{LATIN SMALL LETTER E WITH GRAVE}"'.encode()
    with files.turns.open("ab") as turns:
        turns.write(cut[:-2])  # ends after the first byte of the "è"
    assert [r.case_id for r in read_turns(files)] == ["a"]
    append_case(files, [record("b", "due")])
    assert [r.case_id for r in read_turns(files)] == ["a", "b"]


def test_a_manifest_is_read_back_as_it_was_written(tmp_path: Path) -> None:
    files = RunFiles(tmp_path / "evals" / "run-1")  # the directory is made when it is missing
    written = manifest(budget_file="runs/budget.yaml")
    write_manifest(files, written)
    assert files.manifest.name == "run.json"
    assert read_manifest(files) == written


def test_a_manifest_that_is_missing_or_not_a_manifest_is_reported_by_name(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    with pytest.raises(RunFileError, match=r"run\.json: cannot be read"):
        read_manifest(files)
    files.manifest.write_text('{"run_id": "run-1"}', encoding="utf-8")
    with pytest.raises(RunFileError, match=r"run\.json: not a run manifest"):
        read_manifest(files)


class Killed(BaseException):
    """The process dies at this instant: nothing after it runs."""


def test_a_kill_right_after_the_first_change_to_the_file_loses_no_finished_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cutting the unfinished line must never leave the file without its finished cases, however
    # the process ends in the middle of it: the file is the run's only record of results.
    files = RunFiles(tmp_path)
    for name in ("a", "b"):
        append_case(files, [record(name, name)])
    with files.turns.open("a", encoding="utf-8") as turns:
        turns.write('{"run_id": "run-1", "case_id": "c", "tu')
    finished = files.turns.read_bytes().rsplit(b"\n", 1)[0] + b"\n"
    real_open, real_truncate = io.open, os.truncate

    def dying_open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        handle = real_open(file, mode, *args, **kwargs)  # type: ignore[call-overload]
        if os.fspath(file) == os.fspath(files.turns) and "w" in mode:  # type: ignore[call-overload]
            handle.close()  # opening to write has just emptied the file
            raise Killed
        return handle

    def dying_truncate(path: object, length: int) -> None:
        real_truncate(path, length)  # type: ignore[arg-type]
        if os.fspath(path) == os.fspath(files.turns):  # type: ignore[call-overload]
            raise Killed

    with monkeypatch.context() as patched:
        patched.setattr(io, "open", dying_open)
        patched.setattr(os, "truncate", dying_truncate)
        with pytest.raises(Killed):
            append_case(files, [record("d", "quattro")])
    assert files.turns.read_bytes().startswith(finished)
    assert [r.case_id for r in read_turns(files)] == ["a", "b"]


MIXED = (
    "caff\N{LATIN SMALL LETTER E WITH GRAVE} \N{EURO SIGN} \N{GRINNING FACE} "
    "\N{LINE SEPARATOR} \N{PARAGRAPH SEPARATOR} \N{NEXT LINE}"
)


def test_a_write_cut_at_any_byte_keeps_the_finished_cases_and_the_next_case_is_not_lost(
    tmp_path: Path,
) -> None:
    # Every cut of a record that holds two- and four-byte characters and the three line breaks
    # that `splitlines` would take for ends of line, the whole record without its newline included.
    finished = [record("a", "uno"), record("b", "due")]
    line = record("c", MIXED, message=MIXED).model_dump_json().encode("utf-8")
    prefix = b"".join(r.model_dump_json().encode("utf-8") + b"\n" for r in finished)
    files = RunFiles(tmp_path)
    assert len(line) > 400  # many cuts, several of them inside a character
    for cut in range(len(line) + 1):
        files.turns.write_bytes(prefix + line[:cut])
        assert [r.case_id for r in read_turns(files)] == ["a", "b"], cut
        append_case(files, [record("d", "quattro")])
        assert files.turns.read_bytes().startswith(prefix), cut
        assert [r.case_id for r in read_turns(files)] == ["a", "b", "d"], cut


def test_a_whole_record_without_its_newline_is_not_yet_a_line(tmp_path: Path) -> None:
    # The reader and the appender must agree: what the next append cuts was never counted.
    files = RunFiles(tmp_path)
    append_case(files, [record("a", "uno")])
    with files.turns.open("a", encoding="utf-8") as turns:
        turns.write(record("b", "due").model_dump_json())  # a write cut before its last byte
    assert [r.case_id for r in read_turns(files)] == ["a"]
    assert set(completed_cases(read_turns(files))) == {"a"}
    append_case(files, [record("c", "tre")])
    assert [r.case_id for r in read_turns(files)] == ["a", "c"]
    assert set(completed_cases(read_turns(files))) == {"a", "c"}  # "b" runs again


def test_a_damaged_line_is_an_error_even_as_the_last_one_with_its_newline(tmp_path: Path) -> None:
    # A stop can only leave a line without its newline; a damaged line that has one is damage of
    # another kind, and the next append would bury it in the middle of the file.
    files = RunFiles(tmp_path)
    files.turns.write_text(record("a", "uno").model_dump_json() + "\nnot json\n", encoding="utf-8")
    with pytest.raises(RunFileError, match="line 2"):
        read_turns(files)


class _Counted:
    """A text file that lists each string written to it."""

    def __init__(self, handle: Any, writes: list[str]) -> None:
        self._handle = handle
        self._writes = writes

    def write(self, data: str) -> int:
        self._writes.append(data)
        return int(self._handle.write(data))

    def __enter__(self) -> _Counted:
        self._handle.__enter__()
        return self

    def __exit__(self, *error: object) -> object:
        return self._handle.__exit__(*error)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def test_a_case_is_appended_in_a_single_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One write for the case, not one for each line: a stop between two writes would leave half a
    # case that looks finished. The final bytes alone cannot tell the two apart.
    files = RunFiles(tmp_path)
    writes: list[str] = []
    real_open = Path.open

    def counting(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, mode, *args, **kwargs)
        return _Counted(handle, writes) if self == files.turns and "a" in mode else handle

    monkeypatch.setattr(Path, "open", counting)
    append_case(files, [record("b", "due", turn=1, turns=2), record("b", "tre", turn=2, turns=2)])
    assert len(writes) == 1
    assert writes[0].count("\n") == 2  # both turns, whole, in that one write
    assert [(r.case_id, r.turn) for r in read_turns(files)] == [("b", 1), ("b", 2)]


# A different value for each field of the manifest.
CHANGES: dict[str, object] = {
    "run_id": "run-2",
    "business_id": "demo-shop",
    "route": "offline-demo",
    "model_name": "gemini-3.1-flash",
    "engine_commit": "def5678",
    "cases_file": "evals/moved.yaml",
    "cases_digest": "e" * 64,
    "knowledge_version": "kv-2",
    "classifier_prompt": "x" * 12,
    "responder_prompt": "x" * 12,
    "fixed_texts": {"it": {**TEXTS, "handoff": "Chiama il 112."}},
    "budget_file": "runs/budget.yaml",
    "started_at": AT + timedelta(days=1),
}
# What makes a run this run and no other. The rest (its name, where the cases file lies, which
# budget it was started with, when) may differ when it resumes.
IDENTIFYING = {
    "business_id",
    "route",
    "model_name",
    "engine_commit",
    "cases_digest",
    "knowledge_version",
    "classifier_prompt",
    "responder_prompt",
    "fixed_texts",
}


def test_every_field_of_the_manifest_is_decided_to_identify_the_run_or_not() -> None:
    assert set(CHANGES) == set(RunManifest.model_fields)  # a new field needs a decision here
    for name, value in CHANGES.items():
        differences = manifest_differences(manifest(), manifest(**{name: value}))
        assert differences == ([name] if name in IDENTIFYING else []), name
