from pathlib import Path

import pytest
from pydantic import ValidationError

from frontdesk.settings import EmbeddingBackend, InstagramMode, Settings, load_settings


def test_defaults_are_offline_and_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "FRONTDESK_MODEL_ROUTE",
        "FRONTDESK_TELEMETRY_CAPTURE_CONTENT",
        "FRONTDESK_EMBEDDING_BACKEND",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings()
    assert settings.model_route == "offline-demo"
    assert settings.telemetry_capture_content is False
    assert settings.embedding_backend is EmbeddingBackend.HASH
    assert settings.min_support_similarity == 0.5


def test_the_retrieval_floor_is_off_unless_it_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTDESK_MIN_RETRIEVAL_SIMILARITY", raising=False)
    assert Settings().min_retrieval_similarity == 0.0


def test_the_retrieval_floor_is_read_from_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRONTDESK_MIN_RETRIEVAL_SIMILARITY", "0.35")
    assert Settings().min_retrieval_similarity == 0.35


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        pytest.param("1.5", "less than or equal to 1", id="above-one"),
        pytest.param("-0.1", "greater than or equal to 0", id="below-zero"),
    ],
)
def test_a_retrieval_floor_outside_zero_to_one_is_refused(
    monkeypatch: pytest.MonkeyPatch, value: str, reason: str
) -> None:
    monkeypatch.setenv("FRONTDESK_MIN_RETRIEVAL_SIMILARITY", value)
    with pytest.raises(ValidationError, match=rf"min_retrieval_similarity[\s\S]*{reason}"):
        Settings()


def test_env_file_selects_one_business(tmp_path: Path) -> None:
    env_file = tmp_path / "demo.env"
    env_file.write_text(
        "FRONTDESK_BUSINESS_PACK=businesses/demo-clinic\nFRONTDESK_MODEL_ROUTE=offline-demo\n"
    )
    settings = load_settings(env_file)
    assert settings.require_business_pack() == Path("businesses/demo-clinic")


def test_load_settings_without_a_file_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRONTDESK_BUSINESS_PACK", "businesses/demo-shop")
    assert load_settings().require_business_pack() == Path("businesses/demo-shop")


def test_missing_business_pack_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTDESK_BUSINESS_PACK", raising=False)
    with pytest.raises(ValueError, match="FRONTDESK_BUSINESS_PACK"):
        Settings().require_business_pack()


def test_the_instagram_channel_starts_with_testers_and_switched_on() -> None:
    settings = Settings()
    assert settings.instagram_mode is InstagramMode.TESTERS
    assert settings.instagram_off is False
    assert settings.instagram_api_version == "v25.0"
    assert settings.tasks_queue is None
