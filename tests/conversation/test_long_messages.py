"""A message over the length limit meets no model and only the emergency rules.

The other rules are not linear in the length of a message (the personal-data patterns take time
that grows with the square of a long word), so a long message is checked for an emergency alone
and otherwise goes to reception. None of these tests needs a database.
"""

import time

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.vocab import Category
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import MAX_MESSAGE_CHARS, TurnOutcome, TurnRuntime, handle_message
from frontdesk.safety.normalize import normalize
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey, Language
from tests.conversation.helpers import classifier_model, document_responder, make_runtime
from tests.timing import LINEAR_TIME_LIMIT_SECONDS

# A session factory bound to no database: a long message must not reach the database either.
NO_DATABASE: sessionmaker[Session] = sessionmaker(expire_on_commit=False)
# Calm text in each language, and an emergency to end it with.
CALM: dict[Language, str] = {
    "it": "Vorrei sapere gli orari di apertura della clinica. ",
    "sq": "Dua të di orarin e hapjes së klinikës. ",
    "en": "I would like to know the opening hours of the clinic. ",
}
EMERGENCY: dict[Language, str] = {
    "it": "Aiuto, mi fa male il petto, grazie",
    "sq": "Ndihmë, nuk marr frymë, faleminderit",
    "en": "Help, he isn't breathing, thanks",
}
LANGUAGES: list[Language] = list(CALM)


def _long(language: Language, tail: str = "", size: int = 2_500) -> str:
    """`size` characters of calm text in the language, ending with `tail`."""
    calm = CALM[language]
    room = size - len(tail)
    return (calm * (room // len(calm) + 1))[: room - 1] + " " + tail


def _runtime(pack: BusinessPack, prompts: list[str]) -> TurnRuntime:
    """A runtime whose models fail the test if they are called."""
    return make_runtime(
        pack, NO_DATABASE, classifier_model(None, prompts), document_responder(prompts)
    )


@pytest.mark.parametrize("language", LANGUAGES)
async def test_an_emergency_near_the_end_of_a_long_message_takes_the_emergency_path(
    clinic_pack: BusinessPack, language: Language
) -> None:
    prompts: list[str] = []
    runtime = _runtime(clinic_pack, prompts)
    state = ConversationState(turns=1, topic_service_ids=("physiotherapy",))
    long = _long(language, EMERGENCY[language])
    assert len(long) == 2_500 > MAX_MESSAGE_CHARS
    result, after = await handle_message(long, state, runtime)
    # The same reply, trace and memory as when the person writes the emergency alone.
    short, _ = await handle_message(EMERGENCY[language], state, runtime)
    assert result == short
    assert result.language == language
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rule == "emergency"
    assert result.trace.category is Category.EMERGENCY
    emergency = clinic_pack.fixed_texts[language][FixedTextKey.EMERGENCY]
    handoff = clinic_pack.fixed_texts[language][FixedTextKey.HANDOFF]
    assert result.text == f"{emergency}\n\n{handoff}"
    assert after.topic_service_ids == ()
    assert prompts == []


@pytest.mark.parametrize("language", LANGUAGES)
async def test_a_long_message_without_an_emergency_keeps_the_long_message_handoff(
    clinic_pack: BusinessPack, language: Language
) -> None:
    prompts: list[str] = []
    runtime = _runtime(clinic_pack, prompts)
    long = _long(language)
    assert len(long.strip()) > MAX_MESSAGE_CHARS
    result, _ = await handle_message(long, ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == clinic_pack.fixed_texts[language][FixedTextKey.HANDOFF]
    assert result.trace.rule == "too_long"
    assert result.trace.category is None
    assert prompts == []


@pytest.mark.parametrize(
    ("message", "rule"),
    [
        pytest.param("a" * 100_000, "too_long", id="one-long-word"),
        pytest.param("pain " * 20_000, "too_long", id="repeated-stem"),
        pytest.param("1 " * 50_000, "too_long", id="digits"),
        pytest.param("a.b@" * 25_000, "too_long", id="at-signs"),
        pytest.param(chr(0xFDFA) * 100_000, "too_long", id="characters-that-expand"),
        pytest.param(_long("it", size=100_000), "too_long", id="prose"),
        pytest.param(
            _long("it", EMERGENCY["it"], size=100_000), "emergency", id="prose-then-emergency"
        ),
    ],
)
async def test_a_very_long_message_is_checked_in_linear_time(
    clinic_pack: BusinessPack, message: str, rule: str
) -> None:
    prompts: list[str] = []
    runtime = _runtime(clinic_pack, prompts)
    started = time.perf_counter()
    result, _ = await handle_message(message, ConversationState(turns=1), runtime)
    assert time.perf_counter() - started < LINEAR_TIME_LIMIT_SECONDS
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rule == rule
    assert prompts == []


# U+33AF folds into six characters ("rad", a division slash, "s2"): a letter and 1,999 of them are
# at the limit as written and 11,995 characters once normalized, where the personal-data patterns
# take time that grows with the square of a long word (security review of the M3 plan). The letter
# matters: without one, the message is caught earlier as empty.
EXPANDING = "\N{SQUARE RAD OVER S SQUARED}"


async def test_a_message_that_normalizes_past_the_limit_meets_only_the_emergency_rules(
    clinic_pack: BusinessPack,
) -> None:
    prompts: list[str] = []
    runtime = _runtime(clinic_pack, prompts)
    message = "x" + EXPANDING * 1999
    assert len(message) == MAX_MESSAGE_CHARS < len(normalize(message))
    started = time.perf_counter()
    result, _ = await handle_message(message, ConversationState(turns=1), runtime)
    assert time.perf_counter() - started < LINEAR_TIME_LIMIT_SECONDS
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rule == "too_long"
    assert prompts == []


async def test_an_emergency_in_a_message_that_normalizes_past_the_limit_is_still_found(
    clinic_pack: BusinessPack,
) -> None:
    prompts: list[str] = []
    runtime = _runtime(clinic_pack, prompts)
    message = EXPANDING * 1900 + " " + EMERGENCY["en"]
    assert len(message) < MAX_MESSAGE_CHARS < len(normalize(message))
    result, _ = await handle_message(message, ConversationState(turns=1), runtime)
    assert result.trace.rule == "emergency"
    assert result.trace.category is Category.EMERGENCY
    assert prompts == []


async def test_a_message_that_folds_into_one_long_word_meets_only_the_emergency_rules(
    clinic_pack: BusinessPack,
) -> None:
    # U+3316 folds into six letters yet normalizes to nothing: the folded text is the long one.
    prompts: list[str] = []
    runtime = _runtime(clinic_pack, prompts)
    message = "x" + "\N{SQUARE KIROMEETORU}" * 1999
    assert len(message) == MAX_MESSAGE_CHARS > len(normalize(message))
    result, _ = await handle_message(message, ConversationState(turns=1), runtime)
    assert result.trace.rule == "too_long"
    assert prompts == []
