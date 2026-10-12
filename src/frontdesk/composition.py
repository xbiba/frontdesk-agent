"""Composition root: one deployment, one business (ADR-0008).

Integrations are wired only when the business pack enables them: a clinic never
builds the order store, and nothing here initialises Calendar. A deployment
refuses to start when its active knowledge was embedded by another model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic_ai import Agent
from pydantic_ai.models import Model
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classifier import build_classifier
from frontdesk.agent.demo_models import (
    demo_classifier_model,
    demo_reception_model,
    demo_responder_model,
)
from frontdesk.agent.prompts import read_prompt
from frontdesk.agent.responder import build_responder
from frontdesk.agent.tools import ToolServices, build_toolset
from frontdesk.booking.calcom import CalcomScheduler
from frontdesk.booking.calendar import FixtureCalendar
from frontdesk.booking.google_calendar import SCOPE as CALENDAR_SCOPE
from frontdesk.booking.google_calendar import WRITE_SCOPE as CALENDAR_WRITE_SCOPE
from frontdesk.booking.google_calendar import GoogleCalendar, calendar_http_client
from frontdesk.booking.scheduler import CalendarScheduler, Scheduler
from frontdesk.conversation.reception import Desk, ReceptionReply, build_reception
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.embedders import Embedder, build_embedder
from frontdesk.knowledge.facts import load_catalog
from frontdesk.knowledge.versions import check_embedder
from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.google_tokens import google_token_source
from frontdesk.model_gateway.hedging import HedgedModel
from frontdesk.model_gateway.routes import (
    CHAT_PROVIDERS,
    DataClass,
    UnapprovedRouteError,
    require_route,
)
from frontdesk.settings import CalendarBackend, Conversation, Settings
from frontdesk.shop.orders import OrderStore
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

AgentRole = Literal["classifier", "responder", "agent"]
ModelWrapper = Callable[[Model, AgentRole], Model]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _unwrapped(model: Model, role: AgentRole) -> Model:
    return model


@dataclass(frozen=True)
class Deployment:
    """What one process serves: a business, its database and its embedder.

    `now` is the deployment's clock (an aware time); tests replace it with a fixed one.
    """

    settings: Settings
    pack: BusinessPack
    session_factory: sessionmaker[Session]
    embedder: Embedder
    now: Callable[[], datetime] = _utc_now

    def today(self) -> date:
        """The calendar date in the business's own time zone, not the server's.

        Prices and documents are valid for business days: at 00:30 in Tirane it is already the
        next day there, whichever date the server shows.
        """
        return self.now().astimezone(ZoneInfo(self.pack.timezone)).date()


class NoActiveKnowledgeError(RuntimeError):
    """The business has no approved, active knowledge version yet."""

    def __init__(self, business_id: str) -> None:
        super().__init__(f"business '{business_id}' has no active knowledge version")


def open_deployment(settings: Settings) -> Deployment:
    pack = load_pack(settings.require_business_pack())
    factory = make_session_factory(make_engine(settings.database_url))
    return Deployment(settings, pack, factory, build_embedder(settings.embedding_backend))


def build_turn_runtime(
    deployment: Deployment,
    *,
    today: Callable[[], date] | None = None,
    data_class: DataClass = DataClass.PATIENT_DERIVED,
    wrap_model: ModelWrapper | None = None,
) -> TurnRuntime:
    """The pipeline for this deployment's business.

    `today` defaults to the business-local date (`Deployment.today`); a caller may fix it. Each
    prompt is read once, here: the agents run that text and the runtime records its version, so a
    turn never reads a prompt file. A missing or blank prompt stops the build.

    Conversations carry patient-derived text, so the route must be approved for it unless the
    caller says the messages are evaluation data (the live evaluation does). `wrap_model` lets the
    live evaluation meter each agent's requests; it receives the model and the agent's role.
    """
    pack = deployment.pack
    # The route is checked first: a route the data may not reach, or one that answers no message,
    # stops the build before anything else is opened.
    route = require_route(deployment.settings.model_route, data_class)
    if route.provider != "offline" and route.provider not in CHAT_PROVIDERS:
        # An embedding route is approved for evaluation data too, yet it has no chat model.
        raise UnapprovedRouteError(
            f"model route '{route.route_id}' cannot answer messages: its provider is "
            f"{route.provider}"
        )
    business_today = today or deployment.today
    classifier_prompt = read_prompt(pack.profile, "classifier")
    responder_prompt = read_prompt(pack.profile, "responder")
    with business_session(deployment.session_factory, pack.id) as session:
        catalog = load_catalog(session, pack.id)
        check_embedder(session, pack.id, deployment.embedder)
    if catalog is None:
        raise NoActiveKnowledgeError(pack.id)
    services = ToolServices(
        session_factory=deployment.session_factory,
        embedder=deployment.embedder,
        today=business_today,
        min_similarity=deployment.settings.min_retrieval_similarity,
    )
    toolset = build_toolset(pack.profile, services)
    # The pipeline leads unless the deployment asks for the agent (FRONTDESK_CONVERSATION=agent).
    reception: Agent[Desk, ReceptionReply] | None = None
    reception_version: str | None = None
    if route.provider == "offline":
        classifier = build_classifier(
            demo_classifier_model(catalog), pack.profile, instructions=classifier_prompt.text
        )
        responder = build_responder(
            demo_responder_model(), pack.profile, toolset, instructions=responder_prompt.text
        )
        if deployment.settings.conversation is Conversation.AGENT:
            # The offline path of a deployment the agent leads: the rules, then the agent's stand-in.
            reception_prompt = read_prompt(pack.profile, "reception")
            reception = build_reception(
                demo_reception_model(), pack.profile, instructions=reception_prompt.text
            )
            reception_version = reception_prompt.version
    else:
        model = build_live_model(route, gcp_project=deployment.settings.gcp_project)
        wrap = wrap_model or _unwrapped
        classifier = build_classifier(
            wrap(model, "classifier"),
            pack.profile,
            model_settings=live_model_settings(route, max_tokens=route.output_limits.classifier),
            instructions=classifier_prompt.text,
        )
        responder = build_responder(
            wrap(model, "responder"),
            pack.profile,
            toolset,
            model_settings=live_model_settings(route, max_tokens=route.output_limits.responder),
            instructions=responder_prompt.text,
        )
        if deployment.settings.conversation is Conversation.AGENT:
            reception_prompt = read_prompt(pack.profile, "reception")
            reception = build_reception(
                HedgedModel(wrap(model, "agent")),  # a stuck request gets a twin (Albi, 2026-10-12)
                pack.profile,
                model_settings=live_model_settings(route, max_tokens=route.output_limits.agent),
                instructions=reception_prompt.text,
            )
            reception_version = reception_prompt.version
    return TurnRuntime(
        pack=pack,
        classifier=classifier,
        responder=responder,
        classifier_prompt_version=classifier_prompt.version,
        responder_prompt_version=responder_prompt.version,
        reception=reception,
        reception_prompt_version=reception_version,
        session_factory=deployment.session_factory,
        embedder=deployment.embedder,
        today=business_today,
        min_support_similarity=deployment.settings.min_support_similarity,
        min_retrieval_similarity=deployment.settings.min_retrieval_similarity,
        capture_content=deployment.settings.telemetry_capture_content,
        orders=OrderStore(tuple(pack.orders)) if pack.features.order_status else None,
        now=deployment.now,
        scheduler=(
            build_scheduler(deployment.settings, write=pack.integrations.calendar.chat_booking)
            if pack.integrations.calendar
            else None
        ),
    )


def build_scheduler(settings: Settings, *, write: bool = False) -> Scheduler | None:
    """Where the bookable doctors' free slots come from and where appointments go (ADR-0003).
    Google's credentials are found at the first read, so a deployment builds without them; cal.com
    needs its API key in the environment."""
    if settings.calendar_backend is CalendarBackend.FIXTURE:
        if settings.calendar_fixture is None:
            raise ValueError("FRONTDESK_CALENDAR_FIXTURE is not set for the fixture calendar")
        return CalendarScheduler(FixtureCalendar(settings.calendar_fixture))
    if settings.calendar_backend is CalendarBackend.GOOGLE:
        scope = CALENDAR_WRITE_SCOPE if write else CALENDAR_SCOPE  # writing only to book (M5)
        token = google_token_source([scope], impersonate=settings.calendar_impersonate)
        return CalendarScheduler(GoogleCalendar(token, calendar_http_client()))
    if settings.calendar_backend is CalendarBackend.CALCOM:
        if settings.calcom_api_key is None:
            raise ValueError("FRONTDESK_CALCOM_API_KEY is not set for the cal.com scheduler")
        key = settings.calcom_api_key.get_secret_value()
        return CalcomScheduler(key, calendar_http_client(), base_url=settings.calcom_base_url)
    return None
