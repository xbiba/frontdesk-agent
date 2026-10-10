"""The fake Gemini of the live chat tests, for every test module of this directory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from tests.cli.live_chat_helpers import Failure, Install, fake_gemini


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> Install:
    """Puts a fake Gemini where `build_live_model` would build the real one, so that it takes the
    same `wrap_model` path. Returns the roles of the requests it is sent; `failing` makes a role
    raise (see `fake_gemini`)."""

    def install(
        classification: dict[str, Any], *, failing: Mapping[str, Failure] | None = None
    ) -> list[str]:
        sent: list[str] = []
        model = fake_gemini(classification, sent, failing)
        monkeypatch.setattr(
            "frontdesk.composition.build_live_model", lambda route, *, gcp_project: model
        )
        return sent

    return install
