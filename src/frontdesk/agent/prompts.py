"""Prompts are versioned files under agents/<profile>/prompts.

A prompt is read once, when an agent is built (`read_prompt`): the agent keeps that text and the
version recorded for it comes from the same read. Nothing here caches, so a file edited later is
read afresh by the next build and by nothing else.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

from frontdesk.tenancy.pack import Profile

AGENTS_DIR = Path(__file__).resolve().parents[3] / "agents"


class Prompt(NamedTuple):
    text: str
    version: str


def _version(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_prompt(profile: Profile, name: str) -> str:
    """The text of one prompt file, as the model sees it.

    The file is UTF-8 and a byte-order mark is dropped. Line endings are normalised, so a checkout
    with Windows line endings gives the same text and the same version. A blank file is refused:
    an agent built from it would have no static instructions at all.
    """
    text = (AGENTS_DIR / profile.value / "prompts" / f"{name}.md").read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"prompt file is blank: agents/{profile.value}/prompts/{name}.md")
    return text


def read_prompt(profile: Profile, name: str) -> Prompt:
    """The text of a prompt and its version, from one read."""
    text = load_prompt(profile, name)
    return Prompt(text, _version(text))


def prompt_version(profile: Profile, name: str) -> str:
    """The first 12 hex characters of the SHA-256 of the prompt text, read from disk now."""
    return read_prompt(profile, name).version


def instructions_for(profile: Profile, name: str, given: str | None) -> str:
    """The instructions of an agent: `given` when the caller has read the prompt, else the file."""
    text = load_prompt(profile, name) if given is None else given
    if not text.strip():
        raise ValueError(f"the {name} instructions are blank")
    return text
