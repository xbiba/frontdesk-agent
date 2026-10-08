import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.db


def test_baseline_installs_pgvector(migrated_url: str) -> None:
    engine = create_engine(migrated_url)
    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one()
    engine.dispose()
    assert version


def test_app_role_can_connect(app_url: str) -> None:
    engine = create_engine(app_url)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT current_user")).scalar_one() == "frontdesk_app"
    engine.dispose()
