"""The per-address limit in memory (the site's chat, design §3): a window that slides, addresses
never kept as they are, a bounded memory, and the address Cloud Run's front end saw."""

from fastapi import Request

from frontdesk.api.ratelimit import SlidingWindow, client_address


class Ticks:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_an_address_gets_its_hits_back_as_the_window_slides() -> None:
    ticks = Ticks()
    window = SlidingWindow(2, 600, clock=ticks)
    assert window.allow("203.0.113.7") is True
    ticks.now += 100
    assert window.allow("203.0.113.7") is True
    assert window.allow("203.0.113.7") is False
    assert window.allow("198.51.100.1") is True  # another address counts on its own
    ticks.now += 501  # the first hit is out of the window
    assert window.allow("203.0.113.7") is True
    assert window.allow("203.0.113.7") is False


def test_no_address_is_kept_as_it_is() -> None:
    window = SlidingWindow(2, 600)
    window.allow("203.0.113.7")
    assert "203.0.113.7" not in repr(vars(window))


def test_memory_stays_bounded_and_old_addresses_go_first() -> None:
    ticks = Ticks()
    window = SlidingWindow(5, 600, clock=ticks, max_keys=2)
    assert window.allow("a")
    assert window.allow("b")
    assert window.allow("c") is False  # full of addresses still in their window
    ticks.now += 601
    assert window.allow("c") is True  # the old ones were forgotten


def _request(headers: dict[str, str], peer: str | None = "10.0.0.9") -> Request:
    scope = {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        "client": (peer, 1234) if peer else None,
    }
    return Request(scope)


def test_the_address_is_the_last_forwarded_one() -> None:
    forged = {"x-forwarded-for": "1.2.3.4, 203.0.113.7"}
    assert client_address(_request(forged)) == "203.0.113.7"
    assert client_address(_request({})) == "10.0.0.9"
    assert client_address(_request({}, peer=None)) == "unknown"
