from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.vocab import ClarifyField
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnRuntime, handle_message
from frontdesk.db.engine import business_session
from frontdesk.db.models import OrderLookupAttempts
from frontdesk.knowledge.evidence import Evidence
from frontdesk.shop.attempts import MAX_ORDER_LOOKUPS
from frontdesk.shop.orders import OrderStore
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey, Language
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
ORDER = {"category": "administrative", "intent": "order_status"}
RIGHT = "Ordine ORD-10427, CAP 20121"
WRONG = "Ordine ORD-10427, CAP 00184"


class BrokenStore(OrderStore):
    def lookup(
        self, reference: str, postal_code: str, *, language: Language, now: datetime
    ) -> Evidence | None:
        raise RuntimeError("order system unavailable")


def _runtime(
    pack: BusinessPack, factory: sessionmaker[Session], prompts: list[str] | None = None
) -> TurnRuntime:
    runtime = make_runtime(pack, factory, classifier_model(ORDER), document_responder(prompts))
    return replace(runtime, orders=OrderStore(tuple(pack.orders)))


def _no_information(pack: BusinessPack) -> str:
    texts = pack.fixed_texts["it"]
    return f"{texts[FixedTextKey.NO_INFORMATION]}\n\n{texts[FixedTextKey.HANDOFF]}"


async def test_order_status_is_written_from_the_order(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    runtime = _runtime(shop_pack, activate_pack(shop_pack), prompts)
    result, _ = await handle_message(
        "Dov'è il mio ordine ORD-10427? CAP 20121", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text.startswith("Ordine ORD-10427: spedito. Consegna prevista: 2026-10-10")
    assert prompts == []


async def test_missing_details_get_a_question_and_use_no_attempt(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = _runtime(shop_pack, activate_pack(shop_pack))
    state = ConversationState(turns=1)
    for _ in range(5):
        result, state = await handle_message("Dov'è il mio ordine?", state, runtime)
        assert result.outcome is TurnOutcome.ASKED
        assert state.pending_clarification is ClarifyField.ORDER_DETAILS
    result, _ = await handle_message("Ordine ORD-10427, CAP 20121", state, runtime)
    assert result.outcome is TurnOutcome.REPLIED


async def test_wrong_postal_code_looks_like_an_unknown_order(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = _runtime(shop_pack, activate_pack(shop_pack))
    wrong, _ = await handle_message(
        "Ordine ORD-10427, CAP 00184", ConversationState(turns=1), runtime
    )
    unknown, _ = await handle_message(
        "Ordine ORD-99999, CAP 20121", ConversationState(turns=1), runtime
    )
    assert wrong.text == unknown.text == _no_information(shop_pack)


async def test_attempts_survive_failures_and_run_out(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(shop_pack)
    working = _runtime(shop_pack, factory)
    broken = replace(working, orders=BrokenStore(tuple(shop_pack.orders)))
    state = ConversationState(turns=1)
    failed, state = await handle_message("Ordine ORD-10427, CAP 00184", state, broken)
    assert failed.trace.rule == "error"
    for _ in range(2):
        result, state = await handle_message("Ordine ORD-10427, CAP 00184", state, working)
        assert result.text == _no_information(shop_pack)
    exhausted, _ = await handle_message("Ordine ORD-10427, CAP 20121", state, working)
    assert exhausted.outcome is TurnOutcome.HANDOFF
    assert exhausted.trace.rejection == "order_lookups_exhausted"


def _used(factory: sessionmaker[Session], pack: BusinessPack, state: ConversationState) -> int:
    """The lookups the table has counted for the conversation."""
    with business_session(factory, pack.id) as session:
        row = session.get(OrderLookupAttempts, (pack.id, state.conversation_id))
        return 0 if row is None else row.used


async def test_the_reply_follows_the_language_and_leaves_out_a_missing_delivery_date(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = _runtime(shop_pack, activate_pack(shop_pack))
    result, _ = await handle_message(
        "Where is my order ORD-10458? Postal code 00184", ConversationState(turns=1), runtime
    )
    assert result.language == "en"
    assert result.text == "Order ORD-10458: being prepared"


async def test_the_trace_counts_the_live_evidence_without_listing_its_id(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = _runtime(shop_pack, activate_pack(shop_pack))
    found, _ = await handle_message(RIGHT, ConversationState(turns=1), runtime)
    assert (found.trace.evidence_ids, found.trace.live_evidence) == ((), 1)
    assert found.trace.sources == ("order:ORD-10427",)


async def test_a_wrong_postal_code_and_an_unknown_order_leave_the_same_trace(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = _runtime(shop_pack, activate_pack(shop_pack))
    wrong, _ = await handle_message(WRONG, ConversationState(turns=1), runtime)
    unknown, _ = await handle_message(
        "Ordine ORD-99999, CAP 20121", ConversationState(turns=1), runtime
    )
    assert wrong.trace == unknown.trace
    assert (wrong.trace.live_evidence, wrong.trace.sources) == (0, ())


@pytest.mark.parametrize(
    ("message", "outcome"),
    [
        pytest.param(
            "Dov'è l'ordine ORD-10427? Doveva arrivare il 10.10.2026",
            TurnOutcome.ASKED,
            id="dotted-date-and-no-postal-code",
        ),
        pytest.param(
            "Dov'è l'ordine ORD-10427? Doveva arrivare il 10/10/2026",
            TurnOutcome.ASKED,
            id="slashed-date-and-no-postal-code",
        ),
        pytest.param(
            "Ordine ORD-10427, CAP 20121, consegna 10/10/2026",
            TurnOutcome.REPLIED,
            id="slashed-date-and-postal-code",
        ),
    ],
)
async def test_a_date_in_the_message_is_not_taken_for_the_postal_code(
    activate_pack: Activate, shop_pack: BusinessPack, message: str, outcome: TurnOutcome
) -> None:
    factory = activate_pack(shop_pack)
    runtime = _runtime(shop_pack, factory)
    result, state = await handle_message(message, ConversationState(turns=1), runtime)
    assert result.outcome is outcome
    assert _used(factory, shop_pack, state) == (1 if outcome is TurnOutcome.REPLIED else 0)


@pytest.mark.parametrize(
    ("reference", "postal_code"),
    [
        pytest.param("ORD-10427\n", "20121", id="reference-with-trailing-newline"),
        pytest.param("ORD-10427", "20121\n", id="postal-code-with-trailing-newline"),
        pytest.param("ORD-10427 ", "20121", id="padded-reference"),
        pytest.param("ord-10427", "20121", id="lower-case-reference"),
        pytest.param("ORD-1", "20121", id="short-reference"),
        pytest.param("ORD-10427", "123456789", id="long-postal-code"),
        pytest.param(
            "ORD-10427", "".join(chr(0x0660 + int(d)) for d in "20121"), id="other-digits"
        ),
    ],
)
async def test_details_that_are_not_well_formed_count_as_missing_and_use_no_attempt(
    activate_pack: Activate,
    shop_pack: BusinessPack,
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
    postal_code: str,
) -> None:
    # The extraction only yields well-formed details; this is what the check is for if it ever
    # lets something else through. The patched function ignores the message it is given.
    details = [(reference, postal_code)]
    monkeypatch.setattr(
        "frontdesk.agent.classification.extract_order_details", lambda text: details[0]
    )
    factory = activate_pack(shop_pack)
    runtime = _runtime(shop_pack, factory)
    state = ConversationState(turns=1)
    for _ in range(MAX_ORDER_LOOKUPS + 1):  # more often than the attempts allow
        result, state = await handle_message(RIGHT, state, runtime)
        assert result.outcome is TurnOutcome.ASKED
        assert result.trace.clarify is ClarifyField.ORDER_DETAILS
    assert _used(factory, shop_pack, state) == 0
    details[0] = ("ORD-10427", "20121")
    result, state = await handle_message(RIGHT, state, runtime)
    assert result.outcome is TurnOutcome.REPLIED


async def test_without_an_order_system_the_person_goes_to_reception(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(shop_pack)
    runtime = make_runtime(shop_pack, factory, classifier_model(ORDER), document_responder())
    assert runtime.orders is None
    result, state = await handle_message(RIGHT, ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == shop_pack.fixed_texts["it"][FixedTextKey.HANDOFF]
    assert _used(factory, shop_pack, state) == 0


async def test_every_conversation_has_its_own_attempts(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = _runtime(shop_pack, activate_pack(shop_pack))
    first, second = ConversationState(turns=1), ConversationState(turns=1)
    for _ in range(MAX_ORDER_LOOKUPS):
        _, first = await handle_message(WRONG, first, runtime)
    refused, first = await handle_message(RIGHT, first, runtime)
    allowed, second = await handle_message(RIGHT, second, runtime)
    assert refused.trace.rejection == "order_lookups_exhausted"
    assert allowed.outcome is TurnOutcome.REPLIED
    # The limit holds for the rest of the conversation, even for the right postal code.
    refused_again, _ = await handle_message(RIGHT, first, runtime)
    assert refused_again.trace.rejection == "order_lookups_exhausted"
    assert refused_again.text == shop_pack.fixed_texts["it"][FixedTextKey.HANDOFF]


@pytest.mark.parametrize(
    ("error", "rule"),
    [
        pytest.param(RuntimeError("order system unavailable"), "error", id="error"),
        pytest.param(TimeoutError(), "timeout", id="timeout"),
    ],
)
async def test_an_error_or_timeout_after_the_attempt_keeps_it_used(
    activate_pack: Activate,
    shop_pack: BusinessPack,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    rule: str,
) -> None:
    def failing(
        self: OrderStore, reference: str, postal_code: str, *, language: Language, now: datetime
    ) -> Evidence | None:
        raise error

    monkeypatch.setattr(OrderStore, "lookup", failing)
    factory = activate_pack(shop_pack)
    runtime = _runtime(shop_pack, factory)
    result, state = await handle_message(RIGHT, ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rule == rule
    assert _used(factory, shop_pack, state) == 1


async def test_the_attempt_is_stamped_with_the_runtime_clock(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(shop_pack)
    stamp = datetime(2026, 10, 7, 9, 30, tzinfo=UTC)
    runtime = replace(_runtime(shop_pack, factory), now=lambda: stamp)
    result, state = await handle_message(RIGHT, ConversationState(turns=1), runtime)
    with business_session(factory, shop_pack.id) as session:
        row = session.get(OrderLookupAttempts, (shop_pack.id, state.conversation_id))
        assert row is not None
        assert row.updated_at == stamp
    assert result.outcome is TurnOutcome.REPLIED
