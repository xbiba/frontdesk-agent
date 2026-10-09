"""Run by test_environment_isolation.py in a process of its own; pytest does not collect it."""

from __future__ import annotations

import os

from frontdesk.settings import Settings


def test_no_frontdesk_variable_reaches_a_test() -> None:
    assert sorted(name for name in os.environ if name.startswith("FRONTDESK_")) == []
    # The built-in defaults: a deployment's own database, route and pack are not in play.
    settings = Settings()
    assert settings.database_url.endswith("@localhost:5432/frontdesk")
    assert settings.model_route == "offline-demo"
    assert settings.business_pack is None
