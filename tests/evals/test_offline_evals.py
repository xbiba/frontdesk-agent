from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.composition import build_turn_runtime, open_deployment
from frontdesk.conversation.evaluation import load_turn_cases, render_turn_markdown, run_turn_eval
from frontdesk.settings import EmbeddingBackend, Settings
from frontdesk.tenancy.loader import load_pack

pytestmark = [pytest.mark.db, pytest.mark.eval]
TODAY = date(2026, 10, 7)


@pytest.mark.parametrize("business", ["demo-clinic", "demo-shop"])
async def test_offline_turn_evaluation_is_green(
    business: str,
    businesses_dir: Path,
    database: tuple[str, str],
    activate_pack: Callable[..., sessionmaker[Session]],
) -> None:
    owner_url, app_url = database
    pack_dir = businesses_dir / business
    activate_pack(load_pack(pack_dir))
    settings = Settings(
        business_pack=pack_dir,
        database_url=app_url,
        migration_database_url=owner_url,
        model_route="offline-demo",
        embedding_backend=EmbeddingBackend.HASH,
    )
    runtime = build_turn_runtime(open_deployment(settings), today=lambda: TODAY)
    cases = load_turn_cases(pack_dir / "evals" / "turns.yaml")
    report = await run_turn_eval(runtime, cases, route=settings.model_route)
    assert report.ok, render_turn_markdown(report)
