"""A claim may not carry a link, an address or markdown its quote does not contain (ADR-0009).

Contacts are written by code. A web address or an email address in a model's sentence would be a
contact worded by the model, so each one must be copied from the quote that backs the sentence.
"""

from collections.abc import Sequence

import pytest

from frontdesk.agent.answer_check import AnswerCheck, check_answer
from frontdesk.agent.responder import ResponderOutput
from frontdesk.knowledge.evidence import Evidence, EvidenceKind

SENTENCE = "Klinika Demo has six free parking spaces behind the building"
PASSAGE = (
    f"{SENTENCE}. Book a space at www.parking-demo.example or write to "
    "parking@parking-demo.example (see [the map](https://parking-demo.example/map))."
)


class _Similar:
    """Finds every text identical to every other, so that only the link check decides."""

    dimension = 2
    identity = "similar"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _check(text: str, quote: str, content: str = PASSAGE) -> AnswerCheck:
    passage = Evidence.create(
        kind=EvidenceKind.DOCUMENT, source_ref="doc#1", version="v1", title="Doc", content=content
    )
    output = ResponderOutput.model_validate(
        {
            "action": "answer",
            "claims": [{"text": text, "evidence_id": passage.evidence_id, "quote": quote}],
        }
    )
    return check_answer(output, [passage], embedder=_Similar(), min_similarity=0.5)


@pytest.mark.parametrize(
    "added",
    [
        pytest.param("Book them at www.phish.example.", id="www"),
        pytest.param("Book them at WWW.PHISH.EXAMPLE.", id="www-in-capitals"),
        pytest.param("Book them at https://phish.example/pay.", id="https"),
        pytest.param("Book them at http://phish.example.", id="http"),
        pytest.param("Download it from ftp://files.example/a.", id="another-scheme"),
        pytest.param("Write to parking@phish.example.", id="email"),
        pytest.param("Write to mailto:parking@phish.example.", id="mailto"),
        pytest.param("Write to Parking@Phish.Example.", id="email-in-capitals"),
        pytest.param("See [the map](https://phish.example/map).", id="markdown-link"),
        pytest.param("See [the map](https://parking-demo.example/other).", id="another-path"),
        pytest.param("Book them at www.ph**ish**.example.", id="markdown-hiding-the-address"),
        pytest.param("Book them at www.ph​ish.example.", id="zero-width-space-in-the-address"),
    ],
)
def test_a_link_or_address_the_quote_does_not_contain_is_refused(added: str) -> None:
    result = _check(f"{SENTENCE}. {added}", SENTENCE)
    assert not result.ok
    assert result.reasons == ("link_not_in_quote",)


def test_a_markdown_link_is_refused_even_when_only_its_brackets_are_new() -> None:
    quote = "Book a space at www.parking-demo.example"
    result = _check("Book a space at [www.parking-demo.example](www.parking-demo.example).", quote)
    assert result.reasons == ("link_not_in_quote",)


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        pytest.param(
            "Book a space at www.parking-demo.example/pay.",
            "Book a space at www.parking-demo.example or write to",
            id="a-longer-address-than-the-quote-gives",
        ),
        pytest.param(
            "Book a space at www.parking-demo.example.",
            "Book a space at www.parking-demo.example/book or write to",
            id="a-shorter-address-than-the-quote-gives",
        ),
        pytest.param(
            "Write to ing@parking-demo.example.",
            "write to parking@parking-demo.example (see",
            id="the-end-of-a-longer-email-address",
        ),
        pytest.param(
            "Write to reception@parking-demo.example.",
            "write to parking@parking-demo.example (see",
            id="another-person-at-the-same-site",
        ),
    ],
)
def test_the_link_must_be_the_one_in_the_quote_not_one_like_it(text: str, quote: str) -> None:
    assert _check(text, quote, content=f"{quote}.").reasons == ("link_not_in_quote",)


def test_a_link_in_the_passage_but_not_in_the_quote_is_still_refused() -> None:
    # The link is in the passage, so the quote check passes; the quote just does not hold it.
    result = _check(f"{SENTENCE}, and you can book at www.parking-demo.example.", SENTENCE)
    assert result.reasons == ("link_not_in_quote",)


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        pytest.param(
            "Book a space at www.parking-demo.example.",
            "Book a space at www.parking-demo.example or write to",
            id="www-with-a-full-stop-after-it",
        ),
        pytest.param(
            "You can write to parking@parking-demo.example for details.",
            "write to parking@parking-demo.example (see",
            id="email",
        ),
        pytest.param(
            "Write to Parking@Parking-Demo.example.",
            "write to parking@parking-demo.example (see",
            id="email-in-another-case",
        ),
        pytest.param(
            "Book a space (www.parking-demo.example), or write to us.",
            "Book a space at www.parking-demo.example or write to",
            id="link-in-brackets-and-a-comma-after-it",
        ),
        pytest.param(
            "See [the map](https://parking-demo.example/map).",
            "(see [the map](https://parking-demo.example/map))",
            id="markdown-link-copied-with-its-address",
        ),
        pytest.param(
            "Book a space at www.parking-demo.example or write to parking@parking-demo.example.",
            "Book a space at www.parking-demo.example or write to parking@parking-demo.example",
            id="two-links-both-quoted",
        ),
    ],
)
def test_a_link_copied_from_the_quote_passes(text: str, quote: str) -> None:
    result = _check(text, quote)
    assert result.ok, result.reasons
    assert result.reasons == ()


def test_a_sentence_without_a_link_is_not_affected() -> None:
    result = _check(f"{SENTENCE}.", SENTENCE)
    assert result.ok, result.reasons


@pytest.mark.parametrize(
    "text", ["Opening hours are 10.30 to 18.00, e.g. on weekdays.", "Ask @ the desk."]
)
def test_dots_and_at_signs_that_do_not_make_a_link_are_left_alone(text: str) -> None:
    quote = "Opening hours are 10.30 to 18.00, e.g. on weekdays. Ask @ the desk."
    assert _check(text, quote, quote).ok


def test_the_link_rule_adds_to_the_others_it_does_not_replace_them() -> None:
    result = _check(f"{SENTENCE}, for 12 cars. Book at www.phish.example.", SENTENCE)
    assert result.reasons == ("number_not_in_quote", "link_not_in_quote")
