"""Metadata-only structured logs (spec §13; blueprint OBS-TRC-001).

Call sites pass an event name and keyword fields. Each record becomes one JSON line with
its level and logger name, and:

- "event": the event name when it is a dotted lower-case identifier, else "invalid_event_name";
- the allowlisted fields whose value is plain metadata: None, a bool, a number, an enum
  member (written as its value), a UUID, one word (no whitespace, at most 200 characters)
  or a flat list or tuple of those;
- "dropped_fields": the names of the other fields, those not allowlisted and those holding
  anything else (free text, exceptions, mappings, arbitrary objects). Field names are
  chosen in code and written as given;
- "exception_type": the class name of an exception, never its message or traceback.

So message text, tool results and exception messages are never written, provided call
sites pass codes and identifiers rather than pieces of content: one word is accepted
whatever it says. `format` never raises. A record it cannot render becomes a line with
only level, logger, event and "format_error", so the error never reaches
`logging.Handler.handleError`, which prints the raw message and arguments to stderr.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from enum import Enum
from typing import Any, Final

ALLOWED_FIELDS: Final = frozenset(
    {
        "action",
        "agent_version",
        "business_id",
        "category",
        "check_reasons",
        "conversation_id",
        "correlation_id",
        "count",
        "duration_ms",
        "evidence_ids",
        "input_tokens",
        "invalid_evidence_ids",
        "intent",
        "knowledge_version",
        "language",
        "outcome",
        "output_tokens",
        "reason",
        "route",
        "rule",
        "status",
        "tool",
    }
)
_EVENT_NAME: Final = re.compile(r"^[a-z][a-z0-9_.]{0,79}$")
_MAX_WORD_CHARS: Final = 200
_NOT_METADATA: Final = object()


def _event_name(message: object) -> str:
    if isinstance(message, str) and _EVENT_NAME.match(message):
        return message
    return "invalid_event_name"


def _is_word(text: str) -> bool:
    return len(text) <= _MAX_WORD_CHARS and not any(char.isspace() for char in text)


def _scalar(value: object) -> object:
    """The form of a scalar that may be written, or _NOT_METADATA."""
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str) and _is_word(value):
        return value
    return _NOT_METADATA


def _metadata(value: object) -> object:
    """The form of an allowlisted field's value that may be written, or _NOT_METADATA."""
    if isinstance(value, list | tuple):
        items = [_scalar(item) for item in value]
        return _NOT_METADATA if any(item is _NOT_METADATA for item in items) else items
    return _scalar(value)


def _render(record: logging.LogRecord) -> str:
    payload: dict[str, Any] = {
        "level": record.levelname,
        "logger": record.name,
        "event": _event_name(record.msg),
    }
    dropped: list[str] = []
    fields: dict[str, Any] = getattr(record, "fields", None) or {}
    for key, value in fields.items():
        written = _metadata(value) if key in ALLOWED_FIELDS else _NOT_METADATA
        if written is _NOT_METADATA:
            dropped.append(key)
        else:
            payload[key] = written
    if dropped:
        payload["dropped_fields"] = sorted(dropped)
    if record.exc_info and record.exc_info[0] is not None:
        payload["exception_type"] = record.exc_info[0].__name__
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _fallback(record: logging.LogRecord) -> str:
    """A line without field values, for a record that could not be rendered."""
    payload = {
        "level": record.levelname,
        "logger": record.name,
        "event": _event_name(record.msg),
        "format_error": True,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class MetadataOnlyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            return _render(record)
        except Exception:
            return _fallback(record)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(MetadataOnlyFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"fields": fields})
