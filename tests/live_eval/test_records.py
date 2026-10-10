"""The run files: one run per directory, whole cases only."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontdesk.live_eval.records import (
    RunFileError,
    RunFiles,
    append_case,
    completed_cases,
    manifest_differences,
    read_manifest,
    read_turns,
    write_manifest,
)
from tests.live_eval.helpers import TEXTS, manifest, record


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"knowledge_version": "kv-2"}, "knowledge_version"),
        ({"responder_prompt": "x" * 12}, "responder_prompt"),
        ({"fixed_texts": {"it": {**TEXTS, "handoff": "Chiama il 112."}}}, "fixed_texts"),
        ({"route": "offline-demo"}, "route"),
        ({"engine_commit": "def5678"}, "engine_commit"),
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
