import json
import logging
import sys
import uuid
from collections.abc import Iterator
from enum import Enum, IntEnum, StrEnum

import pytest

from frontdesk.observability.logs import MetadataOnlyFormatter, configure_logging, log_event

SENTINEL = "SENTINEL-7f3a"


class _Outcome(StrEnum):
    REPLIED = "replied"


class _Route(Enum):
    DEMO = "offline-demo"
    PROSE = f"free text {SENTINEL}"


class _Attempt(IntEnum):
    FIRST = 1


class _Printable:
    """An object whose string forms hold text, as an exception or a model result would."""

    def __str__(self) -> str:
        return f"text {SENTINEL}"

    def __repr__(self) -> str:
        return f"text {SENTINEL}"


def _circular() -> list[object]:
    items: list[object] = []
    items.append(items)
    return items


def _record(msg: str, fields: object | None = None, exc: bool = False) -> logging.LogRecord:
    logger = logging.getLogger("test")
    exc_info = sys.exc_info() if exc else None
    extra = {"fields": fields} if fields is not None else None
    return logger.makeRecord("test", logging.INFO, __file__, 1, msg, (), exc_info, extra=extra)


def _format(fields: object) -> str:
    return MetadataOnlyFormatter().format(_record("turn.completed", fields))


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


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(ValueError(SENTINEL), id="exception"),
        pytest.param(f"free text {SENTINEL}", id="string_with_spaces"),
        pytest.param(f"{SENTINEL}\tmore", id="string_with_tab"),
        pytest.param(f"{SENTINEL}\nmore", id="string_with_newline"),
        pytest.param(f"{SENTINEL}{chr(0xA0)}more", id="string_with_no_break_space"),
        pytest.param({"note": SENTINEL}, id="nested_mapping"),
        pytest.param(_Printable(), id="object_with_text_in_str"),
        pytest.param(["ok", {"note": SENTINEL}], id="list_holding_a_mapping"),
        pytest.param([[SENTINEL]], id="nested_list"),
        pytest.param(["ok", f"free text {SENTINEL}"], id="list_holding_free_text"),
        pytest.param(SENTINEL.encode(), id="bytes"),
        pytest.param({SENTINEL}, id="set"),
        pytest.param(_Route.PROSE, id="enum_whose_value_is_free_text"),
        pytest.param(_circular(), id="circular_list"),
    ],
)
def test_values_that_are_not_metadata_are_dropped(value: object) -> None:
    out = _format({"reason": value})
    written = json.loads(out)
    assert SENTINEL not in out
    assert "reason" not in written
    assert written["dropped_fields"] == ["reason"]


def test_a_word_may_have_200_characters_but_not_201() -> None:
    kept = json.loads(_format({"reason": "a" * 200}))
    dropped = json.loads(_format({"reason": "a" * 201}))
    assert kept["reason"] == "a" * 200
    assert "dropped_fields" not in kept
    assert "reason" not in dropped
    assert dropped["dropped_fields"] == ["reason"]


def test_dropped_fields_combine_unknown_keys_and_rejected_values() -> None:
    written = json.loads(
        _format({"zeta": "x", "reason": "two words", "category": "ok", "alpha": 1})
    )
    assert written["dropped_fields"] == ["alpha", "reason", "zeta"]
    assert written["category"] == "ok"


def test_values_from_the_planned_call_sites_are_kept() -> None:
    fields = {
        "business_id": "demo-clinic",
        "rule": "personal_data:phone",
        "reason": "ModelHTTPError",
        "language": "sq",
        "category": None,
        "intent": None,
        "invalid_evidence_ids": 0,
        "duration_ms": 12.5,
        "evidence_ids": ["D-1a2b3c4d", "R-5e6f7a8b"],
        "check_reasons": ["quote_not_in_source", "price_in_answer"],
    }
    written = json.loads(_format(fields))
    assert written == {
        "event": "turn.completed",
        "level": "INFO",
        "logger": "test",
        **fields,
    }


def test_enum_members_and_uuids_are_written_as_plain_values() -> None:
    conversation_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    written = json.loads(
        _format(
            {
                "outcome": _Outcome.REPLIED,
                "route": _Route.DEMO,
                "count": _Attempt.FIRST,
                "conversation_id": conversation_id,
                "evidence_ids": [_Outcome.REPLIED, conversation_id, 3, None, True],
            }
        )
    )
    assert written["outcome"] == "replied"
    assert written["route"] == "offline-demo"
    assert written["count"] == 1
    assert written["conversation_id"] == "12345678-1234-5678-1234-567812345678"
    assert written["evidence_ids"] == [
        "replied",
        "12345678-1234-5678-1234-567812345678",
        3,
        None,
        True,
    ]
    assert "dropped_fields" not in written


@pytest.mark.parametrize(
    ("msg", "fields", "event"),
    [
        pytest.param(
            "turn.completed",
            {"duration_ms": float("nan")},
            "turn.completed",
            id="number_json_cannot_hold",
        ),
        pytest.param(
            "turn.completed",
            f"not a mapping {SENTINEL}",
            "turn.completed",
            id="fields_not_a_mapping",
        ),
        pytest.param(
            f"patient wrote {SENTINEL}",
            f"not a mapping {SENTINEL}",
            "invalid_event_name",
            id="free_text_event_name",
        ),
    ],
)
def test_format_never_raises_and_falls_back_to_a_bare_line(
    msg: str, fields: object, event: str
) -> None:
    out = MetadataOnlyFormatter().format(_record(msg, fields))
    assert SENTINEL not in out
    assert json.loads(out) == {
        "event": event,
        "format_error": True,
        "level": "INFO",
        "logger": "test",
    }


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    """configure_logging replaces the root logger's handlers: put them back afterwards."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_logging_writes_one_metadata_only_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    logger = logging.getLogger("frontdesk.test")
    try:
        raise ValueError(SENTINEL)
    except ValueError:
        logger.exception("turn.failed")
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["event"] == "turn.failed"
    assert written["exception_type"] == "ValueError"
    assert SENTINEL not in captured.out
    assert captured.err == ""
