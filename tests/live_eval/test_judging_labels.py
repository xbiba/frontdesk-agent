"""The labels file is read from a person's hands: every way it can be wrong says where."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontdesk.live_eval.judging import Label, LabelsError, load_labels
from tests.live_eval.helpers import case

CASES = [case("parking"), case("hours"), case("price-list")]


def written(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_every_field_of_a_label_is_read(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        "parking:\n  refusal: right\nhours:\n  answer: wrong\n  invented_fact: true\n"
        '  note: "twelve hours"\nprice-list:\n  answer: right\n',
    )
    assert load_labels(path, CASES) == {
        "parking": Label(refusal="right"),
        "hours": Label(answer="wrong", invented_fact=True, note="twelve hours"),
        "price-list": Label(answer="right"),
    }


@pytest.mark.parametrize("text", ["", "\n", "# nothing reviewed yet\n", "{}\n"])
def test_a_file_with_no_label_in_it_gives_none(tmp_path: Path, text: str) -> None:
    assert load_labels(written(tmp_path, text), CASES) == {}


def test_a_case_with_nothing_under_it_is_an_empty_label(tmp_path: Path) -> None:
    assert load_labels(written(tmp_path, "parking:\n"), CASES) == {"parking": Label()}


@pytest.mark.parametrize("value", ["false", "0", "wrong", "[]", '""'])
def test_a_label_that_is_not_a_mapping_is_refused_naming_the_case(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(LabelsError, match="case 'parking'"):
        load_labels(written(tmp_path, f"parking: {value}\n"), CASES)


def test_a_field_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LabelsError, match=r"case 'parking': Extra inputs are not permitted"):
        load_labels(written(tmp_path, "parking:\n  verdict: wrong\n"), CASES)


def test_every_problem_is_reported_at_once(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        "parking:\n  answer: maybe\nnobody:\n  answer: right\nhours:\n  invented_fact: true\n",
    )
    with pytest.raises(LabelsError) as raised:
        load_labels(path, CASES)
    lines = str(raised.value).splitlines()
    assert len(lines) == 3
    assert f"{path}: case 'parking'" in lines[0]
    assert lines[1] == f"{path}: 'nobody' is not a case of this run"
    assert f"{path}: case 'hours': " in lines[2]
    assert "invented_fact needs answer: wrong" in lines[2]


def test_invalid_yaml_says_where_it_stopped(tmp_path: Path) -> None:
    # The line and the column are counted from 1, as an editor shows them.
    path = written(tmp_path, "parking:\n  refusal: wrong\n note: x\n")
    with pytest.raises(LabelsError) as raised:
        load_labels(path, CASES)
    assert str(raised.value).startswith(f"{path}: invalid YAML at line 3, column 2: ")
    assert "block" in str(raised.value)  # and what the parser found there


def test_invalid_yaml_without_a_position_is_still_named(tmp_path: Path) -> None:
    # A control character is refused by the reader before any line is parsed: no mark to quote.
    path = tmp_path / "labels.yaml"
    path.write_bytes(b"parking:\n  note: a\x00b\n")
    with pytest.raises(LabelsError) as raised:
        load_labels(path, CASES)
    assert str(raised.value) == f"{path}: invalid YAML"


def test_a_file_that_is_not_utf8_asks_to_be_saved_as_utf8(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_bytes('parking:\n  note: "è"\n'.encode("latin-1"))
    with pytest.raises(LabelsError, match=r"labels\.yaml: not valid UTF-8 text"):
        load_labels(path, CASES)


def test_a_labels_path_that_cannot_be_read_is_a_labels_error(tmp_path: Path) -> None:
    directory = tmp_path / "labels.yaml"
    directory.mkdir()
    with pytest.raises(LabelsError, match=r"labels\.yaml: cannot be read \(Is a directory\)"):
        load_labels(directory, CASES)


@pytest.mark.parametrize("text", ["- parking\n", "just words\n", "3\n", "[]\n", "0\n", "false\n"])
def test_a_file_that_is_not_a_mapping_is_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(LabelsError, match=r"expected a mapping of case id to label"):
        load_labels(written(tmp_path, text), CASES)
