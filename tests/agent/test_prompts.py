"""Prompt files: one read gives the text and its version, and a blank file is refused."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from frontdesk.agent import prompts
from frontdesk.agent.prompts import load_prompt, prompt_version, read_prompt
from frontdesk.tenancy.pack import Profile

PROFILE = Profile.RECEPTIONIST


def _write(
    root: Path, monkeypatch: pytest.MonkeyPatch, content: bytes, name: str = "classifier"
) -> Path:
    path = root / PROFILE.value / "prompts" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    monkeypatch.setattr(prompts, "AGENTS_DIR", root)
    return path


def test_one_read_gives_the_text_and_its_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, monkeypatch, b"hello\n")
    text, version = read_prompt(PROFILE, "classifier")
    assert text == "hello\n"
    assert version == "5891b5b522d5"
    assert version == hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    assert version == prompt_version(PROFILE, "classifier")


def test_the_version_follows_the_file_on_every_call_so_nothing_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, monkeypatch, b"hello\n")
    assert read_prompt(PROFILE, "classifier").version == "5891b5b522d5"
    path.write_bytes(b"hello!\n")
    assert read_prompt(PROFILE, "classifier").version == "c8a31cb076b2"


def test_the_version_is_that_of_the_text_the_model_sees_not_of_the_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, monkeypatch, b"one\ntwo\n")
    unix = read_prompt(PROFILE, "classifier")
    path.write_bytes(b"one\r\ntwo\r\n")  # the same prompt checked out with Windows line endings
    windows = read_prompt(PROFILE, "classifier")
    assert windows == unix
    assert windows.text == "one\ntwo\n"


def test_a_byte_order_mark_is_not_part_of_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, monkeypatch, b"hello\n")
    plain = read_prompt(PROFILE, "classifier")
    path.write_bytes(b"\xef\xbb\xbfhello\n")
    marked = read_prompt(PROFILE, "classifier")
    assert marked == plain
    assert "\ufeff" not in marked.text


@pytest.mark.parametrize("content", [b"", b"\n", b"  \t\r\n\n", b"\xef\xbb\xbf\n"])
@pytest.mark.parametrize("read", [load_prompt, read_prompt, prompt_version])
def test_a_blank_prompt_is_refused_and_the_error_names_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read: Callable[[Profile, str], object],
    content: bytes,
) -> None:
    _write(tmp_path, monkeypatch, content, name="responder")
    with pytest.raises(ValueError, match=r"receptionist/prompts/responder\.md"):
        read(PROFILE, "responder")


def test_a_missing_prompt_is_a_file_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts, "AGENTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        read_prompt(PROFILE, "classifier")


@pytest.mark.parametrize("profile", list(Profile))
@pytest.mark.parametrize("name", ["classifier", "responder"])
def test_the_shipped_prompts_are_not_blank(profile: Profile, name: str) -> None:
    text, version = read_prompt(profile, name)
    assert text.strip()
    assert len(version) == 12
