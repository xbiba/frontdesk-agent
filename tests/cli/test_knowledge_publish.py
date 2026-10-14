"""`frontdesk knowledge publish` (M4): the pack's knowledge, approved elsewhere, drafted, approved
and activated in one step; a second run finds it active."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from frontdesk.cli.publish import publish_knowledge
from frontdesk.settings import Settings

pytestmark = pytest.mark.db
NOW = datetime(2026, 10, 14, 9, tzinfo=UTC)


def test_publish_activates_the_pack_s_knowledge_once(
    app_url: str, businesses_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(business_pack=businesses_dir / "demo-clinic", database_url=app_url)
    evidence = "approvals/demo: synthetic"
    assert publish_knowledge(settings, by="test", evidence=evidence, now=NOW) == 0
    assert capsys.readouterr().out.startswith("active: ")
    assert publish_knowledge(settings, by="test", evidence=evidence, now=NOW) == 0
    assert capsys.readouterr().out.startswith("already active: ")
