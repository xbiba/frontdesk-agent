"""The channels of one deployment (M4, ADR-0011; WhatsApp W1): the webhook's part and the worker's
part, built from the settings. The webhook's part loads nothing of the model, the embedder or the
turn. WhatsApp is the reception's: its numbers come from the staff list, and it runs no model. The
site's chat (2026-11-07) answers anyone on the sites the pack lists, with its own limits."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx2

from frontdesk.channels.dispatch import CloudTasksDispatcher, Dispatcher
from frontdesk.channels.inbox import Inbox
from frontdesk.channels.instagram import INSTAGRAM, InstagramWebhook
from frontdesk.channels.messages import WebhookReader
from frontdesk.channels.staff import Staff, read_staff
from frontdesk.channels.web import WEB, WebReplies
from frontdesk.channels.whatsapp import WHATSAPP, WhatsAppWebhook
from frontdesk.db.engine import make_engine, make_session_factory
from frontdesk.model_gateway.google_tokens import google_token_source
from frontdesk.model_gateway.routes import DataClass, require_route
from frontdesk.secrets import CLOUD_SCOPE, SECRET_IDS, read_secret
from frontdesk.settings import InstagramMode, Settings
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack, ChannelPolicy, WebChannel

if TYPE_CHECKING:
    from frontdesk.channels.requests import Requests
    from frontdesk.channels.runner import Runner
    from frontdesk.composition import Deployment
    from frontdesk.conversation.state import ConversationState
    from frontdesk.conversation.turn import TurnResult


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Ingress:
    """What the webhook needs (M4 design §4 steps 1-3)."""

    webhook: WebhookReader
    inbox: Inbox
    dispatcher: Dispatcher
    verify_token: str
    off: bool


@dataclass(frozen=True)
class WebIngress:
    """What the site's routes need: the pack's web section and contacts, the inbox, the replies."""

    channel: WebChannel
    pack: BusinessPack
    inbox: Inbox
    replies: WebReplies
    dispatcher: Dispatcher
    off: bool


@dataclass(frozen=True)
class Worker:
    """What the worker needs, by channel: the runner of each conversation and the inbox for the
    schedules; and the booking requests the staff decides on WhatsApp (W1), if it is on."""

    runners: Mapping[str, Runner]
    inbox: Mapping[str, Inbox]
    requests: Requests | None
    dispatcher: Dispatcher
    purge_bookings: Callable[[], int]


def channel_secret(settings: Settings, name: str) -> str:
    """A secret the channel cannot run without (frontdesk.secrets), read once at start."""
    with httpx2.Client(timeout=10.0) as client:
        value = read_secret(
            settings,
            name,
            token=google_token_source([CLOUD_SCOPE], impersonate=None),
            client=client,
        )
    if not value:
        variable = f"FRONTDESK_{name.upper()}"
        raise ValueError(f"the secret {SECRET_IDS[name]} is missing ({variable} in development)")
    return value


def instagram_policy(pack: BusinessPack) -> ChannelPolicy:
    if pack.channels.instagram is None:
        raise ValueError(f"the pack of {pack.id} does not enable Instagram (channels.instagram)")
    return pack.channels.instagram


def build_ingress(settings: Settings) -> Ingress:
    pack = load_pack(settings.require_business_pack())
    instagram_policy(pack)
    testers = settings.instagram_mode is InstagramMode.TESTERS
    if not testers:
        # Live mode stores patients' messages: refused here too when the worker could not answer
        # them (final review of M4, M6).
        require_route(settings.model_route, DataClass.PATIENT_DERIVED)
    inbox = Inbox(
        make_session_factory(make_engine(settings.database_url)),
        business_id=pack.id,
        channel=INSTAGRAM,
        account_id=_account(settings),
        testers_only=testers,
        pairing_code=channel_secret(settings, "instagram_pairing_code") if testers else None,
        now=_utc_now,
    )
    return Ingress(
        webhook=InstagramWebhook(channel_secret(settings, "instagram_app_secret")),
        inbox=inbox,
        dispatcher=_dispatcher(settings),
        verify_token=channel_secret(settings, "instagram_verify_token"),
        off=settings.instagram_off,
    )


def build_ingresses(settings: Settings) -> tuple[Ingress, ...]:
    """Instagram's ingress when the pack lists the channel, and WhatsApp's when it lists that one
    (W1). A pack with neither starts with no webhook route: a deployment of email drafts or the site
    widget only (2026-11-07)."""
    pack = load_pack(settings.require_business_pack())
    found: tuple[Ingress, ...] = ()
    if pack.channels.instagram is not None:
        found = (build_ingress(settings),)
    if pack.channels.whatsapp is None:
        return found
    staff = _staff(settings)
    inbox = Inbox(
        make_session_factory(make_engine(settings.database_url)),
        business_id=pack.id,
        channel=WHATSAPP,
        account_id=_phone_number(settings),
        testers_only=False,
        pairing_code=None,
        now=_utc_now,
        allowed_users=frozenset(member.user_id for member in staff),
    )
    whatsapp = Ingress(
        webhook=WhatsAppWebhook(channel_secret(settings, "whatsapp_app_secret")),
        inbox=inbox,
        dispatcher=found[0].dispatcher if found else _dispatcher(settings),
        verify_token=channel_secret(settings, "whatsapp_verify_token"),
        off=settings.whatsapp_off,
    )
    return (*found, whatsapp)


def build_web_ingress(
    settings: Settings, *, dispatcher: Dispatcher | None = None
) -> WebIngress | None:
    """The site's ingress when the pack lists the channel (`channels.web`). In tester mode it
    answers only the sessions that wrote Instagram's pairing code (design §4)."""
    pack = load_pack(settings.require_business_pack())
    if pack.channels.web is None:
        return None
    testers = settings.web_mode is InstagramMode.TESTERS
    if not testers:
        require_route(settings.model_route, DataClass.PATIENT_DERIVED)
    factory = make_session_factory(make_engine(settings.database_url))
    inbox = Inbox(
        factory,
        business_id=pack.id,
        channel=WEB,
        account_id=pack.id,
        testers_only=testers,
        pairing_code=channel_secret(settings, "instagram_pairing_code") if testers else None,
        now=_utc_now,
    )
    return WebIngress(
        channel=pack.channels.web,
        pack=pack,
        inbox=inbox,
        replies=WebReplies(factory, business_id=pack.id),
        dispatcher=dispatcher or _dispatcher(settings),
        off=settings.web_off,
    )


def build_web_worker(
    settings: Settings, deployment: Deployment, *, data_class: DataClass | None = None
) -> tuple[Runner, Inbox]:
    """The site's runner and inbox: the pack narrowed by the web policy (no booking, no order),
    the web Sender, the limits of design §3. Testers write test messages; everyone else's are
    patient data, unless the caller says otherwise (a try on this computer)."""
    from frontdesk.channels.policy import narrow
    from frontdesk.channels.runner import ChannelLimits, Runner
    from frontdesk.channels.web import MESSAGES_PER_SESSION, REPEATS, TURNS_A_DAY, WebSender
    from frontdesk.composition import build_turn_runtime
    from frontdesk.conversation.turn import handle_message

    web = deployment.pack.channels.web
    if web is None:
        raise ValueError(
            f"the pack of {deployment.pack.id} does not enable the site (channels.web)"
        )
    testers = settings.web_mode is InstagramMode.TESTERS
    if data_class is None:
        data_class = DataClass.EVALUATION if testers else DataClass.PATIENT_DERIVED
    pack = narrow(deployment.pack, web.policy())
    runtime = build_turn_runtime(replace(deployment, pack=pack), data_class=data_class)

    async def turn(text: str, state: ConversationState) -> tuple[TurnResult, ConversationState]:
        return await handle_message(text, state, runtime)

    runner = Runner(
        deployment.session_factory,
        pack=pack,
        channel=WEB,
        sender=WebSender(),
        turn=turn,
        now=deployment.now,
        off=settings.web_off,
        limits=ChannelLimits(
            per_conversation=MESSAGES_PER_SESSION, repeats=REPEATS, turns_a_day=TURNS_A_DAY
        ),
    )
    inbox = Inbox(
        deployment.session_factory,
        business_id=pack.id,
        channel=WEB,
        account_id=pack.id,
        testers_only=testers,
        pairing_code=None,
        now=deployment.now,
    )
    return runner, inbox


def build_worker(settings: Settings) -> Worker:
    # Only the worker loads the model, the embedder and the engine's turn.
    from frontdesk.booking.holds import purge
    from frontdesk.channels.instagram import InstagramSender, PageToken
    from frontdesk.channels.policy import narrow
    from frontdesk.channels.requests import Requests, only_buttons
    from frontdesk.channels.runner import Runner
    from frontdesk.channels.whatsapp import WhatsAppSender
    from frontdesk.composition import build_turn_runtime, open_deployment
    from frontdesk.conversation.turn import handle_message
    from frontdesk.db.engine import business_session

    deployment = open_deployment(settings)
    policy = deployment.pack.channels.instagram
    pack = deployment.pack if policy is None else narrow(deployment.pack, policy)
    testers = settings.instagram_mode is InstagramMode.TESTERS
    # Testers write test messages; everyone else's are patient data, refused by an unapproved route.
    data_class = DataClass.EVALUATION if testers else DataClass.PATIENT_DERIVED
    runtime = build_turn_runtime(replace(deployment, pack=pack), data_class=data_class)
    client = httpx2.AsyncClient(timeout=httpx2.Timeout(15.0, connect=5.0))

    async def turn(text: str, state: ConversationState) -> tuple[TurnResult, ConversationState]:
        return await handle_message(text, state, runtime)

    def purge_bookings() -> int:
        with business_session(deployment.session_factory, pack.id) as session:
            return purge(session, deployment.now())

    dispatcher = _dispatcher(settings)
    runners: dict[str, Runner] = {}
    inboxes: dict[str, Inbox] = {}
    if policy is not None:  # Instagram is optional (2026-11-07)
        account, page = _account(settings), instagram_page(settings)
        token = PageToken(
            client,
            channel_secret(settings, "instagram_token"),
            page_id=page,
            api_version=settings.instagram_api_version,
        )
        sender = InstagramSender(
            token.get, client, page_id=page, api_version=settings.instagram_api_version
        )
        runners[INSTAGRAM] = Runner(
            deployment.session_factory,
            pack=pack,
            channel=INSTAGRAM,
            sender=sender,
            turn=turn,
            now=deployment.now,
            off=settings.instagram_off,
        )
        inboxes[INSTAGRAM] = Inbox(
            deployment.session_factory,
            business_id=pack.id,
            channel=INSTAGRAM,
            account_id=account,
            testers_only=testers,
            pairing_code=None,
            now=deployment.now,
        )
    requests = None
    if deployment.pack.channels.whatsapp is not None:
        if runtime.scheduler is None:
            raise ValueError("WhatsApp's booking requests need the pack's calendar")
        staff = _staff(settings)
        staff_inbox = Inbox(
            deployment.session_factory,
            business_id=pack.id,
            channel=WHATSAPP,
            account_id=_phone_number(settings),
            testers_only=False,
            pairing_code=None,
            now=deployment.now,
            allowed_users=frozenset(member.user_id for member in staff),
        )
        requests = Requests(
            deployment.session_factory,
            pack=pack,
            scheduler=runtime.scheduler,
            staff=staff,
            whatsapp=staff_inbox,
            dispatcher=dispatcher,
            now=deployment.now,
            language=settings.whatsapp_staff_language,
            patient_off=settings.instagram_off,
        )
        whatsapp_token = channel_secret(settings, "whatsapp_token")
        runners[WHATSAPP] = Runner(
            deployment.session_factory,
            pack=pack,
            channel=WHATSAPP,
            sender=WhatsAppSender(
                lambda: whatsapp_token,
                client,
                phone_number_id=_phone_number(settings),
                api_version=settings.whatsapp_api_version,
            ),
            turn=turn,  # never called: the staff's thread has its fixed reply in W1
            now=deployment.now,
            off=settings.whatsapp_off,
            buttons=requests.decide,
            fixed_reply=only_buttons(settings.whatsapp_staff_language),
        )
        inboxes[WHATSAPP] = staff_inbox
    if deployment.pack.channels.web is not None:
        runners[WEB], inboxes[WEB] = build_web_worker(settings, deployment)
    return Worker(
        runners=runners,
        inbox=inboxes,
        requests=requests,
        dispatcher=dispatcher,
        purge_bookings=purge_bookings,
    )


def _staff(settings: Settings) -> tuple[Staff, ...]:
    """The staff list from its secret: a bad one stops the build, naming the entry."""
    return read_staff(channel_secret(settings, "whatsapp_staff"))


def _phone_number(settings: Settings) -> str:
    if not settings.whatsapp_phone_number_id:
        raise ValueError(
            "FRONTDESK_WHATSAPP_PHONE_NUMBER_ID is not set: Meta's WhatsApp > API Setup shows it"
        )
    return settings.whatsapp_phone_number_id


def instagram_page(settings: Settings) -> str:
    if not settings.instagram_page_id:
        raise ValueError(
            "FRONTDESK_INSTAGRAM_PAGE_ID is not set: `frontdesk instagram account` prints it"
        )
    return settings.instagram_page_id


def _account(settings: Settings) -> str:
    if not settings.instagram_account_id:
        raise ValueError(
            "FRONTDESK_INSTAGRAM_ACCOUNT_ID is not set: `frontdesk instagram account` prints it"
        )
    return settings.instagram_account_id


def _dispatcher(settings: Settings) -> CloudTasksDispatcher:
    if not (settings.tasks_queue and settings.worker_url and settings.tasks_invoker):
        raise ValueError(
            "the channel needs FRONTDESK_TASKS_QUEUE, FRONTDESK_WORKER_URL and "
            "FRONTDESK_TASKS_INVOKER"
        )
    return CloudTasksDispatcher(
        queue=settings.tasks_queue,
        worker_url=settings.worker_url,
        invoker=settings.tasks_invoker,
        token=google_token_source([CLOUD_SCOPE], impersonate=None),
        client=httpx2.AsyncClient(timeout=10.0),
        now=_utc_now,
    )
