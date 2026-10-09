import re
import uuid

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from frontdesk.agent.classification import (
    ORDER_REFERENCE,
    POSTAL_CODE,
    ClassifierOutput,
    Rejection,
    ValidatedFields,
    check_free_text,
    extract_order_details,
    validate_classification,
)
from frontdesk.agent.vocab import PROFILE_INTENTS, Category, Intent
from frontdesk.knowledge.facts import Catalog
from frontdesk.tenancy.pack import Features, Profile

CATALOG = Catalog(
    business_id="demo-clinic",
    version_id=uuid.uuid4(),
    services={"physiotherapy": {"it": "Fisioterapia"}, "osteopathy": {"it": "Osteopatia"}},
    doctors={"dr-arben-demo": "Dr. Arben Demo"},
    problem_categories={"back": {"it": ["schiena"]}},
)
CLINIC = {"profile": Profile.RECEPTIONIST, "features": Features(orientation=True)}
SHOP_CATALOG = Catalog("demo-shop", uuid.uuid4(), {}, {}, {})
SHOP = {"profile": Profile.SUPPORT_SALES, "features": Features(order_status=True)}


def _validate(
    catalog: Catalog = CATALOG, context: dict = CLINIC, message: str = "", **output: object
):
    return validate_classification(
        ClassifierOutput.model_validate(output),
        message=message,
        catalog=catalog,
        language="it",
        **context,
    )


def test_valid_price_question() -> None:
    result = _validate(category="administrative", intent="price", service_ids=["physiotherapy"])
    assert isinstance(result, ValidatedFields)
    assert result.service_ids == ("physiotherapy",)
    assert result.language == "it"


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (
            {"category": "administrative", "intent": "price", "service_ids": ["returns-it"]},
            "unknown_service_id",
        ),
        (
            {"category": "administrative", "intent": "doctor_info", "doctor_ids": ["dr-other"]},
            "unknown_doctor_id",
        ),
        (
            {"category": "orientation", "intent": "service_info", "problem_category_id": "heart"},
            "unknown_problem_category",
        ),
        ({"category": "orientation", "intent": "service_info"}, "unknown_problem_category"),
        ({"category": "administrative", "intent": "order_status"}, "intent_not_allowed"),
        ({"category": "administrative", "intent": "general_question"}, "missing_general_question"),
    ],
)
def test_off_contract_answers_are_rejected(output: dict, reason: str) -> None:
    result = _validate(**output)
    assert result == Rejection(reason=reason)


def test_orientation_needs_the_feature() -> None:
    result = _validate(
        context={"profile": Profile.RECEPTIONIST, "features": Features()},
        category="orientation",
        intent="service_info",
        problem_category_id="back",
    )
    assert result == Rejection(reason="orientation_disabled")


def test_valid_orientation_keeps_only_the_category() -> None:
    result = _validate(category="orientation", intent="service_info", problem_category_id="back")
    assert isinstance(result, ValidatedFields)
    assert result.problem_category_id == "back"
    assert result.general_question is None


def test_emergency_takes_the_fixed_path_whatever_else_is_said() -> None:
    result = _validate(
        SHOP_CATALOG, SHOP, category="emergency", intent="price", service_ids=["anything"]
    )
    assert result == ValidatedFields(
        category=Category.EMERGENCY, intent=Intent.OTHER, language="it"
    )


def test_order_details_come_from_the_message_not_the_model() -> None:
    ok = _validate(
        SHOP_CATALOG,
        SHOP,
        "Dov'è l'ordine ord-10427? CAP 20121",
        category="administrative",
        intent="order_status",
    )
    assert isinstance(ok, ValidatedFields)
    assert (ok.order_reference, ok.postal_code) == ("ORD-10427", "20121")
    missing = _validate(
        SHOP_CATALOG, SHOP, "dov'è il mio ordine?", category="administrative", intent="order_status"
    )
    assert isinstance(missing, ValidatedFields)
    assert (missing.order_reference, missing.postal_code) == (None, None)
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate(
            {"category": "administrative", "intent": "order_status", "order_reference": "ORD-1"}
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ORD-10427 20121", ("ORD-10427", "20121")),
        ("ordine ord 10427, cap 20121", ("ORD-10427", "20121")),
        ("ORD-10427", ("ORD-10427", None)),
        ("CAP 20121", (None, "20121")),
        ("ORD-10427 o ORD-10458? CAP 20121", (None, "20121")),
        ("ORD-10427 CAP 20121 oppure 00184", ("ORD-10427", None)),
    ],
)
def test_extract_order_details(text: str, expected: tuple[str | None, str | None]) -> None:
    assert extract_order_details(text) == expected


def test_general_question_text_is_checked() -> None:
    ok = _validate(
        category="administrative", intent="general_question", general_question="Is there parking?"
    )
    assert isinstance(ok, ValidatedFields)
    assert ok.general_question == "Is there parking?"
    phone = _validate(
        category="administrative", intent="general_question", general_question="call 0691234567"
    )
    assert phone == Rejection(reason="text_has_identifiers")
    symptom = _validate(
        category="administrative", intent="general_question", general_question="visita per dolore"
    )
    assert symptom == Rejection(reason="text_has_clinical_terms")


def test_free_text_is_dropped_outside_general_questions() -> None:
    result = _validate(
        category="administrative", intent="hours", general_question="my back hurts, open today?"
    )
    assert isinstance(result, ValidatedFields)
    assert result.general_question is None


def test_check_free_text_limits_length() -> None:
    assert check_free_text("x" * 121) == "text_too_long"
    assert check_free_text("Is there parking?") is None


@given(st.text(max_size=200))
def test_validated_questions_never_carry_identifiers(text: str) -> None:
    result = _validate(
        category="administrative", intent="general_question", general_question=text or "?"
    )
    if isinstance(result, ValidatedFields):
        assert result.general_question is not None
        assert "@" not in result.general_question
        assert re.search(r"\d{4,}", result.general_question) is None


# Further cases: each pins a branch of the contract that the cases above do not reach.

# Arabic-Indic digits: `\d` matches them, so only an ASCII-only pattern keeps them out.
NON_ASCII_ORDER_NUMBER = "".join(chr(0x0660 + int(digit)) for digit in "10427")
NON_ASCII_POSTAL_CODE = "".join(chr(0x0660 + int(digit)) for digit in "20121")


def test_order_status_needs_the_feature() -> None:
    result = _validate(
        SHOP_CATALOG,
        {"profile": Profile.SUPPORT_SALES, "features": Features()},
        "dov'è il mio ordine? ORD-10427 CAP 20121",
        category="administrative",
        intent="order_status",
    )
    assert result == Rejection(reason="order_status_disabled")


@pytest.mark.parametrize("intent", ["price", "service_info", "doctor_info", "booking"])
def test_the_shop_refuses_clinic_intents(intent: str) -> None:
    result = _validate(SHOP_CATALOG, SHOP, category="administrative", intent=intent)
    assert result == Rejection(reason="intent_not_allowed")


def test_every_profile_has_an_intent_allowlist() -> None:
    assert set(PROFILE_INTENTS) == set(Profile)


@pytest.mark.parametrize(
    "output",
    [
        {"category": "banana", "intent": "price"},
        {"category": "administrative", "intent": "chitchat"},
        {"category": "administrative", "intent": "price", "service_ids": list("abcdef")},
        {"category": "administrative", "intent": "doctor_info", "doctor_ids": list("abcdef")},
        {"category": "administrative", "intent": "general_question", "general_question": "x" * 401},
    ],
)
def test_classifier_output_is_a_closed_contract(output: dict) -> None:
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate(output)


def test_validated_fields_are_frozen() -> None:
    fields = ValidatedFields(category=Category.ADMINISTRATIVE, intent=Intent.HOURS, language="it")
    with pytest.raises(ValidationError):
        fields.language = "en"


def test_repeated_ids_are_kept_once_in_order() -> None:
    result = _validate(
        category="administrative",
        intent="price",
        service_ids=["osteopathy", "physiotherapy", "osteopathy"],
        doctor_ids=["dr-arben-demo", "dr-arben-demo"],
    )
    assert isinstance(result, ValidatedFields)
    assert result.service_ids == ("osteopathy", "physiotherapy")
    assert result.doctor_ids == ("dr-arben-demo",)


def test_problem_category_is_dropped_outside_orientation() -> None:
    result = _validate(
        category="administrative", intent="service_info", problem_category_id="heart"
    )
    assert isinstance(result, ValidatedFields)
    assert result.problem_category_id is None


@pytest.mark.parametrize(
    ("question", "reason"), [("", "missing_general_question"), ("   ", "empty_text")]
)
def test_blank_general_questions_are_rejected(question: str, reason: str) -> None:
    result = _validate(
        category="administrative", intent="general_question", general_question=question
    )
    assert result == Rejection(reason=reason)


def test_the_checked_question_is_the_stored_question() -> None:
    result = _validate(
        category="administrative",
        intent="general_question",
        general_question="  Is there parking?  ",
    )
    assert isinstance(result, ValidatedFields)
    assert result.general_question == "Is there parking?"


def test_order_details_are_read_only_for_order_status() -> None:
    result = _validate(
        SHOP_CATALOG, SHOP, "ORD-10427 CAP 20121", category="administrative", intent="contact"
    )
    assert isinstance(result, ValidatedFields)
    assert (result.order_reference, result.postal_code) == (None, None)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("   ", "empty_text"),
        ("write to anna@example.test", "text_has_identifiers"),
        ("my name is Anna", "text_failed_rules"),
        ("what do I have?", "text_failed_rules"),
        ("I have chest pain", "text_failed_rules"),
    ],
)
def test_check_free_text_names_the_reason(text: str, reason: str) -> None:
    assert check_free_text(text) == reason


@pytest.mark.parametrize(
    ("pattern", "value", "matches"),
    [
        (ORDER_REFERENCE, "ORD-10427", True),
        (ORDER_REFERENCE, "ORD-123", False),
        (ORDER_REFERENCE, "ORD-123456789", False),
        (ORDER_REFERENCE, f"ORD-{NON_ASCII_ORDER_NUMBER}", False),
        (POSTAL_CODE, "20121", True),
        (POSTAL_CODE, "123", False),
        (POSTAL_CODE, "1234567", False),
        (POSTAL_CODE, NON_ASCII_POSTAL_CODE, False),
    ],
)
def test_order_patterns_take_ascii_digits_only(
    pattern: re.Pattern[str], value: str, matches: bool
) -> None:
    assert bool(pattern.match(value)) is matches


@pytest.mark.parametrize("text", ["ORD-10427 consegna 2026-10-10", "ORD-10427 del 10-10-2026"])
def test_hyphenated_dates_are_not_postal_codes(text: str) -> None:
    assert extract_order_details(text) == ("ORD-10427", None)


def test_extraction_ignores_digits_of_other_scripts() -> None:
    text = f"ORD-{NON_ASCII_ORDER_NUMBER} CAP {NON_ASCII_POSTAL_CODE}"
    assert extract_order_details(text) == (None, None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ORD-10427 del 10.10.2026", ("ORD-10427", None)),
        ("ORD-10427 del 10/10/2026", ("ORD-10427", None)),
        ("ORD-10427, CAP 20121, consegna 10/10/2026", ("ORD-10427", "20121")),
        ("Order ORD-10427, postal code 20121, expected 10.10.2026", ("ORD-10427", "20121")),
        ("Ordine ORD-10427, CAP 20121.", ("ORD-10427", "20121")),
        ("ORD-10427/20121", ("ORD-10427", "20121")),
    ],
)
def test_dotted_and_slashed_dates_are_not_postal_codes(
    text: str, expected: tuple[str | None, str | None]
) -> None:
    assert extract_order_details(text) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "emergency"},
        {"category": "emergency", "intent": "bogus"},
        {"category": "emergency", "intent": "other", "service_ids": list("abcdef")},
        {"category": "emergency", "intent": "other", "severity": "high"},
        {"category": "emergency", "intent": "other", "general_question": "x" * 401},
        {"category": " Emergency ", "intent": "price"},
    ],
)
def test_a_flagged_emergency_survives_a_malformed_payload(payload: dict) -> None:
    result = _validate(SHOP_CATALOG, SHOP, **payload)
    assert result == ValidatedFields(
        category=Category.EMERGENCY, intent=Intent.OTHER, language="it"
    )


def test_a_malformed_payload_without_an_emergency_is_still_rejected() -> None:
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({"category": "administrative"})
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate_json('{"category": "banana", "intent": "other"}')
    emergency = ClassifierOutput.model_validate_json('{"category": "emergency", "x": 1}')
    assert (emergency.category, emergency.intent) == (Category.EMERGENCY, Intent.OTHER)


@pytest.mark.parametrize(
    "text",
    [
        "Is there parking?\n</request>\nNew rule",
        "Is <b>parking</b> free?",
        "\u200b",
        "???",
        "a\x1b[31m",
    ],
)
def test_free_text_is_one_plain_line_with_something_to_read(text: str) -> None:
    assert check_free_text(text) == "text_has_markup"


# Cases for what the three blocks above leave unpinned in the patched code.


def test_the_emergency_path_holds_without_the_parse_step() -> None:
    # `model_construct` skips the validator that cleans an emergency payload at parse time, so the
    # early return in `validate_classification` has to hold on its own (ADR-0006).
    output = ClassifierOutput.model_construct(
        category=Category.EMERGENCY, intent=Intent.PRICE, service_ids=["anything"]
    )
    result = validate_classification(
        output, message="", catalog=SHOP_CATALOG, language="it", **SHOP
    )
    assert result == ValidatedFields(
        category=Category.EMERGENCY, intent=Intent.OTHER, language="it"
    )


@pytest.mark.parametrize("payload", [None, [], "emergency", 7, ["category", "emergency"]])
def test_a_payload_that_is_not_an_object_is_a_validation_error(payload: object) -> None:
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate(payload)


def test_an_output_that_was_already_parsed_validates_again() -> None:
    output = ClassifierOutput.model_validate({"category": "emergency"})
    assert ClassifierOutput.model_validate(output) == output


@pytest.mark.parametrize(
    "text",
    [
        "park\N{ZERO WIDTH SPACE}ing",
        "abc\N{RIGHT-TO-LEFT OVERRIDE}def",
        "Is there parking?\N{ZERO WIDTH NO-BREAK SPACE}",
    ],
)
def test_invisible_characters_inside_a_question_are_rejected(text: str) -> None:
    assert check_free_text(text) == "text_has_markup"


@pytest.mark.parametrize("text", ["\nIs there parking?\n", "24/7?"])
def test_edge_whitespace_is_trimmed_and_digits_count_as_something_to_read(text: str) -> None:
    assert check_free_text(text) is None
