"""Metadata-only spans (spec §13; blueprint OBS-TRC-001, §26.11).

A turn span carries IDs, categories, outcomes, prompt versions and evidence IDs: the verified
ones, plus the number of invalid ones. An ID the model wrote that matched no provided passage is
never recorded, and neither are source references, which can include order numbers. Live evidence
(an order's status) is only counted: its ID is derived from the order reference. Message and
reply text are added only when the deployment enables content capture, and then always redacted.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

TRACER_NAME: Final = "frontdesk"
ALLOWED_ATTRIBUTES: Final = frozenset(
    {
        "business_id",
        "category",
        "check_reasons",
        "classifier_prompt_version",
        "evidence_ids",
        "intent",
        "invalid_evidence_ids",
        "language",
        "live_evidence",
        "outcome",
        "rejection",
        "responder_prompt_version",
        "rule",
    }
)
_ADDRESS_RUN: Final = re.compile(r"[\w.+-]+")
_DOMAIN: Final = re.compile(r"@[\w-]+\.[\w.-]+")
_PHONE: Final = re.compile(r"(?:\+|00)?\d(?:[\s().-]?\d){7,}")


@contextmanager
def turn_span(business_id: str) -> Iterator[Span]:
    tracer = trace.get_tracer(TRACER_NAME)
    # OpenTelemetry would copy exception messages into the span; they can carry content.
    with tracer.start_as_current_span(
        "frontdesk.turn", record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("frontdesk.business_id", business_id)
        try:
            yield span
        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            raise


def record_metadata(span: Span, **attributes: str | int | list[str] | None) -> None:
    for key, value in attributes.items():
        if key in ALLOWED_ATTRIBUTES and value is not None:
            kept = value if isinstance(value, list | int) else str(value)
            span.set_attribute(f"frontdesk.{key}", kept)


def redact(text: str) -> str:
    """Emails first, so digits inside an address are not mistaken for a phone number."""
    return _PHONE.sub("[phone]", _mask_emails(text))


def _mask_emails(text: str) -> str:
    """Every address (`name@domain.tld`) becomes [email], in time linear in the length of the text.

    One pattern for the whole address finds the same ones, but it takes time that grows with the
    square of the length of a word with no "@", and a message can be 100,000 characters long. Here
    each run of address characters is read once and only a run that ends at an "@" is looked at.
    """
    masked: list[str] = []
    done = 0  # the text before this index is already written
    for run in _ADDRESS_RUN.finditer(text):
        # The previous address can end inside this run: its domain may be followed by "+" and a name.
        start = max(run.start(), done)
        domain = _DOMAIN.match(text, run.end())
        if start < run.end() and domain is not None:
            masked.extend((text[done:start], "[email]"))
            done = domain.end()
    masked.append(text[done:])
    return "".join(masked)


def record_content(span: Span, *, enabled: bool, message: str, reply: str) -> None:
    if enabled:
        span.add_event("frontdesk.content", {"message": redact(message), "reply": redact(reply)})
