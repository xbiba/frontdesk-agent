"""Process settings. One process serves exactly one business (ADR-0008)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingBackend(StrEnum):
    HASH = "hash"
    FASTEMBED = "fastembed"


class CalendarBackend(StrEnum):
    """Where the doctors' free slots are read (ADR-0003): nowhere, a YAML fixture for tests and
    evaluations, or Google Calendar."""

    NONE = "none"
    FIXTURE = "fixture"
    GOOGLE = "google"
    CALCOM = "calcom"  # cal.com keeps the availability and writes to Google Calendar


class Conversation(StrEnum):
    """Who leads a conversation: the pipeline (rules, classifier, code's replies) or the reception
    agent, a model with the clinic's facts and tools that code checks (Albi, 2026-10-12)."""

    PIPELINE = "pipeline"
    AGENT = "agent"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRONTDESK_", extra="ignore")

    business_pack: Path | None = None
    database_url: str = "postgresql+psycopg://frontdesk_app:frontdesk_app@localhost:5432/frontdesk"
    migration_database_url: str = (
        "postgresql+psycopg://frontdesk:frontdesk@localhost:5432/frontdesk"
    )
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    model_route: str = "offline-demo"
    gcp_project: str | None = None
    # The reception agent leads the conversation on live routes; offline stays the pipeline.
    conversation: Conversation = Conversation.PIPELINE
    embedding_backend: EmbeddingBackend = EmbeddingBackend.HASH
    min_support_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    # The vector side of retrieval keeps chunks at least this similar to the query (P47). 0 keeps
    # the nearest ones whatever their similarity; each embedder is tuned with its own value.
    min_retrieval_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    calendar_backend: CalendarBackend = CalendarBackend.NONE
    calendar_fixture: Path | None = None
    # The service account the Mac impersonates with its ADC; unset on Cloud Run, which runs as it.
    calendar_impersonate: str | None = None
    # cal.com's API key: in the environment only, never in a committed env file.
    calcom_api_key: SecretStr | None = None
    calcom_base_url: str = "https://api.cal.com"
    telemetry_capture_content: bool = False
    log_level: str = "INFO"

    def require_business_pack(self) -> Path:
        if self.business_pack is None:
            raise ValueError("FRONTDESK_BUSINESS_PACK is not set for this deployment")
        return self.business_pack


def load_settings(env_file: Path | None = None) -> Settings:
    if env_file is None:
        return Settings()
    return Settings(_env_file=env_file)
