import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from frontdesk.api.app import create_app
from frontdesk.settings import Settings


@pytest.mark.db
def test_health_is_ok_when_database_answers(app_url: str) -> None:
    client = TestClient(create_app(Settings(), engine=create_engine(app_url)))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_unavailable_without_database() -> None:
    engine = create_engine(
        "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none",
        connect_args={"connect_timeout": 1},
    )
    client = TestClient(create_app(Settings(), engine=engine))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
