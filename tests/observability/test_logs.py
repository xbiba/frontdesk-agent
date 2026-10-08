import logging
import sys

import pytest

from frontdesk.observability.logs import MetadataOnlyFormatter, log_event

SENTINEL = "SENTINEL-7f3a"


def _record(
    msg: str, fields: dict[str, object] | None = None, exc: bool = False
) -> logging.LogRecord:
    logger = logging.getLogger("test")
    exc_info = sys.exc_info() if exc else None
    extra = {"fields": fields} if fields is not None else None
    return logger.makeRecord("test", logging.INFO, __file__, 1, msg, (), exc_info, extra=extra)


def test_content_fields_are_dropped() -> None:
    out = MetadataOnlyFormatter().format(
        _record("turn.completed", {"category": "administrative", "message_text": SENTINEL})
    )
    assert SENTINEL not in out
    assert '"dropped_fields": ["message_text"]' in out
    assert '"category": "administrative"' in out


def test_free_text_event_names_are_replaced() -> None:
    out = MetadataOnlyFormatter().format(_record(f"patient wrote {SENTINEL}"))
    assert SENTINEL not in out
    assert '"event": "invalid_event_name"' in out


def test_exception_messages_are_not_logged() -> None:
    try:
        raise ValueError(SENTINEL)
    except ValueError:
        record = _record("turn.failed", exc=True)
    out = MetadataOnlyFormatter().format(record)
    assert SENTINEL not in out
    assert '"exception_type": "ValueError"' in out


def test_log_event_attaches_fields(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("frontdesk.test")
    with caplog.at_level(logging.INFO):
        log_event(logger, "turn.completed", outcome="replied")
    assert caplog.records[0].fields == {"outcome": "replied"}  # type: ignore[attr-defined]
