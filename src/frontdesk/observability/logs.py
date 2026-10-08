"""Metadata-only structured logs (spec §13; blueprint OBS-TRC-001).

Call sites pass an event name and keyword fields. Only allowlisted metadata is
written: message text, tool results and exception messages never are.
"""

from __future__ import annotations

import json
import logging
import re
import sys
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


class MetadataOnlyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.msg
        event = (
            message
            if isinstance(message, str) and _EVENT_NAME.match(message)
            else "invalid_event_name"
        )
        payload: dict[str, Any] = {"level": record.levelname, "logger": record.name, "event": event}
        fields: dict[str, Any] = getattr(record, "fields", None) or {}
        payload.update({key: value for key, value in fields.items() if key in ALLOWED_FIELDS})
        dropped = sorted(key for key in fields if key not in ALLOWED_FIELDS)
        if dropped:
            payload["dropped_fields"] = dropped
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(MetadataOnlyFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"fields": fields})
