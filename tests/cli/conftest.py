"""The fake Gemini of the live chat tests, for every test module of this directory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tests.cli.live_chat_helpers import fake_gemini


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], list[str]]:
    """Puts a fake Gemini where `build_live_model` would build the real one, so that it takes the
    same `wrap_model` path. Returns the roles of the requests it is sent."""

    def install(classification: dict[str, Any]) -> list[str]:
        sent: list[str] = []
        model = fake_gemini(classification, sent)
        monkeypatch.setattr(
            "frontdesk.composition.build_live_model", lambda route, *, gcp_project: model
        )
        return sent

    return install
