from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.evidence import EvidenceKind
from frontdesk.shop.attempts import MAX_ORDER_LOOKUPS, consume_order_lookup
from frontdesk.shop.orders import OrderStore
from frontdesk.tenancy.pack import BusinessPack, Language
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def test_lookup_returns_live_evidence(shop_pack: BusinessPack) -> None:
    store = OrderStore(tuple(shop_pack.orders))
    evidence = store.lookup("ORD-10427", "20121", language="it", now=NOW)
    assert evidence is not None
    assert evidence.kind is EvidenceKind.LIVE
    assert evidence.source_ref == "order:ORD-10427"
    assert evidence.content == "Ordine ORD-10427: spedito. Consegna prevista: 2026-10-10"


def test_wrong_postal_code_and_unknown_order_look_the_same(shop_pack: BusinessPack) -> None:
    store = OrderStore(tuple(shop_pack.orders))
    assert store.lookup("ORD-10427", "00184", language="it", now=NOW) is None
    assert store.lookup("ORD-99999", "20121", language="it", now=NOW) is None


@pytest.mark.db
def test_attempts_are_bounded_per_conversation_and_business(app_url: str) -> None:
    factory: sessionmaker[Session] = make_session_factory(make_engine(app_url))

    def consume(business_id: str, conversation_id: str) -> bool:
        with business_session(factory, business_id) as session:
            return consume_order_lookup(
                session, business_id=business_id, conversation_id=conversation_id, now=NOW
            )

    assert [consume("demo-shop", "c1") for _ in range(MAX_ORDER_LOOKUPS + 1)] == [
        True,
        True,
        True,
        False,
    ]
    assert consume("demo-shop", "c2") is True
    assert consume("other-shop", "c1") is True


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        pytest.param("it", "Ordine ORD-10427: spedito. Consegna prevista: 2026-10-10", id="it"),
        pytest.param("en", "Order ORD-10427: shipped. Expected delivery: 2026-10-10", id="en"),
        # The order has no Albanian status: its first one is written with Albanian labels.
        pytest.param("sq", "Porosia ORD-10427: spedito. Dorëzimi i pritur: 2026-10-10", id="sq"),
    ],
)
def test_the_text_is_written_in_the_language_of_the_conversation(
    shop_pack: BusinessPack, language: Language, expected: str
) -> None:
    store = OrderStore(tuple(shop_pack.orders))
    evidence = store.lookup("ORD-10427", "20121", language=language, now=NOW)
    assert evidence is not None
    assert evidence.content == expected


def test_an_order_without_a_delivery_date_has_no_delivery_line(shop_pack: BusinessPack) -> None:
    store = OrderStore(tuple(shop_pack.orders))
    evidence = store.lookup("ORD-10458", "00184", language="en", now=NOW)
    assert evidence is not None
    assert evidence.content == "Order ORD-10458: being prepared"
    assert evidence.values == {"status": "being prepared"}


def test_the_evidence_values_and_version_come_from_the_record_and_the_lookup_time(
    shop_pack: BusinessPack,
) -> None:
    store = OrderStore(tuple(shop_pack.orders))
    evidence = store.lookup("ORD-10427", "20121", language="en", now=NOW)
    assert evidence is not None
    assert evidence.values == {"status": "shipped", "eta": "2026-10-10"}
    assert evidence.version == "2026-10-07T09:00:00+00:00"


@pytest.mark.parametrize(
    ("reference", "postal_code"),
    [
        pytest.param("ORD-10427\n", "20121", id="reference-with-trailing-newline"),
        pytest.param("ORD-10427", "20121\n", id="postal-code-with-trailing-newline"),
        pytest.param("ord-10427", "20121", id="lower-case-reference"),
        pytest.param(" ORD-10427", "20121", id="padded-reference"),
        pytest.param("ORD-10427", "", id="empty-postal-code"),
    ],
)
def test_only_the_exact_reference_and_postal_code_find_an_order(
    shop_pack: BusinessPack, reference: str, postal_code: str
) -> None:
    store = OrderStore(tuple(shop_pack.orders))
    assert store.lookup(reference, postal_code, language="it", now=NOW) is None


def test_the_runtime_clock_defaults_to_the_current_time_in_utc(shop_pack: BusinessPack) -> None:
    unbound: sessionmaker[Session] = sessionmaker(expire_on_commit=False)  # no database needed
    runtime = make_runtime(shop_pack, unbound, classifier_model(None), document_responder())
    now = runtime.now()
    assert now.utcoffset() == timedelta(0)
    assert abs(datetime.now(UTC) - now) < timedelta(seconds=5)
