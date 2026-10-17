"""Process settings. One process serves exactly one business (ADR-0008)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingBackend(StrEnum):
    HASH = "hash"
    FASTEMBED = "fastembed"
    NONE = "none"  # a small pack's knowledge rides in the prompt: nothing is searched (2026-10-16)


class CalendarBackend(StrEnum):
    """Where the doctors' free slots are read (ADR-0003): nowhere, a YAML fixture for tests and
    evaluations, or Google Calendar."""

    NONE = "none"
    FIXTURE = "fixture"
    GOOGLE = "google"
    CALCOM = "calcom"  # cal.com keeps the availability and writes to Google Calendar


class SecretsBackend(StrEnum):
    """Where the engine reads its secrets (Albi, 2026-10-13): the environment in development, Google
    Secret Manager in production (frontdesk.secrets)."""

    ENV = "env"
    GCP = "gcp"


class Conversation(StrEnum):
    """Who leads a conversation: the pipeline (rules, classifier, code's replies) or the reception
    agent, a model with the clinic's facts and tools that code checks (Albi, 2026-10-12)."""

    PIPELINE = "pipeline"
    AGENT = "agent"


class InstagramMode(StrEnum):
    """Whom the Instagram channel answers (M4 design D4): its testers, or everyone."""

    TESTERS = "testers"
    LIVE = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRONTDESK_", extra="ignore")

    business_pack: Path | None = None
    database_url: str = "postgresql+psycopg://frontdesk_app:frontdesk_app@localhost:5432/frontdesk"
    migration_database_url: str = (
        "postgresql+psycopg://frontdesk:frontdesk@localhost:5432/frontdesk"
    )
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
    secrets_backend: SecretsBackend = SecretsBackend.ENV
    secrets_project: str | None = None  # the project of Secret Manager; else gcp_project
    calcom_base_url: str = "https://api.cal.com"
    telemetry_capture_content: bool = False
    # Production's guards on the model (Albi, 2026-10-13): a cap on each day's spend, priced from
    # a prices file (the live runs' budget format), and a switch that turns the model off.
    daily_cap_eur: float | None = Field(default=None, gt=0)
    prices_file: Path | None = None
    model_off: bool = False
    # Instagram (M4, ADR-0011): the account its webhooks name, whom it answers, its switch (D4,
    # D13). Token, app secret, verify token and pairing code come from the environment in
    # development only, like cal.com's key; from Secret Manager in production.
    instagram_account_id: str | None = None
    instagram_mode: InstagramMode = InstagramMode.TESTERS
    instagram_off: bool = False
    instagram_api_version: str = "v25.0"
    instagram_token: SecretStr | None = None
    instagram_app_secret: SecretStr | None = None
    instagram_verify_token: SecretStr | None = None
    instagram_pairing_code: SecretStr | None = None
    # WhatsApp for the reception (W1): the business account the booking request template lives in,
    # the bot's number, the language of every text to the staff, Italian unless reception prefers
    # Albanian (Albi, 2026-10-17), and the switch. The secrets, the staff list among them, come
    # from Secret Manager in production.
    whatsapp_business_account_id: str | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_api_version: str = "v25.0"
    whatsapp_staff_language: Literal["it", "sq"] = "it"
    whatsapp_off: bool = False
    whatsapp_token: SecretStr | None = None
    whatsapp_app_secret: SecretStr | None = None
    whatsapp_verify_token: SecretStr | None = None
    whatsapp_staff: SecretStr | None = None
    # Cloud Tasks (ADR-0011): the queue, the worker it calls, the identity it calls the worker with.
    tasks_queue: str | None = None
    worker_url: str | None = None
    tasks_invoker: str | None = None
    log_level: str = "INFO"

    def require_business_pack(self) -> Path:
        if self.business_pack is None:
            raise ValueError("FRONTDESK_BUSINESS_PACK is not set for this deployment")
        return self.business_pack


def load_settings(env_file: Path | None = None) -> Settings:
    if env_file is None:
        return Settings()
    return Settings(_env_file=env_file)
