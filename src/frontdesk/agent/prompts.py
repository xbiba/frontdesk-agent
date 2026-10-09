"""Prompts are versioned files under agents/<profile>/prompts (blueprint §22.6)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from frontdesk.tenancy.pack import Profile

AGENTS_DIR = Path(__file__).resolve().parents[3] / "agents"


def load_prompt(profile: Profile, name: str) -> str:
    return (AGENTS_DIR / profile.value / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def prompt_version(profile: Profile, name: str) -> str:
    return hashlib.sha256(load_prompt(profile, name).encode("utf-8")).hexdigest()[:12]
