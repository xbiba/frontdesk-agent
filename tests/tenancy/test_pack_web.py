"""The `web:` section of a pack (the chat on the business's site, 2026-11-07): the sites allowed to
embed the widget, its name and words in each language, the privacy link, an optional style; bookings and
orders always off."""

from pathlib import Path
from typing import Any

import pytest

from frontdesk.tenancy.loader import PackError, load_pack
from frontdesk.tenancy.pack import ChannelPolicy, WebChannel
from tests.tenancy.helpers import CLINIC, copy_pack, edit_yaml, web_section, web_texts


def _load(businesses_dir: Path, tmp_path: Path, section: dict[str, Any]) -> WebChannel | None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda d: d.update(channels={"web": section}))
    return load_pack(pack_dir).channels.web


def test_a_pack_has_no_web_channel_by_default(businesses_dir: Path) -> None:
    assert load_pack(businesses_dir / CLINIC).channels.web is None


def test_the_web_section_is_read(businesses_dir: Path, tmp_path: Path) -> None:
    web = _load(businesses_dir, tmp_path, web_section())
    assert web is not None
    assert web.origins == ["https://clinic.example", "http://localhost:8000"]
    assert web.style is None  # the engine's neutral look until the client styles it
    assert web.texts["it"].send == "Send it"
    assert web.allows("https://clinic.example") is True
    assert web.allows("https://other.example") is False
    assert web.allows(None) is False


def test_the_web_channel_never_books_or_looks_up_orders(
    businesses_dir: Path, tmp_path: Path
) -> None:
    web = _load(businesses_dir, tmp_path, web_section())
    assert web is not None
    assert web.policy() == ChannelPolicy(chat_booking=False, order_status=False, orientation=True)


def test_the_privacy_link_may_wait(businesses_dir: Path, tmp_path: Path) -> None:
    web = _load(businesses_dir, tmp_path, web_section(privacy_url=None))
    assert web is not None
    assert web.privacy_url is None


@pytest.mark.parametrize(
    "changes",
    [
        {"origins": []},
        {"origins": ["clinic.example"]},
        {"origins": ["https://clinic.example/"]},
        {"origins": ["http://clinic.example"]},  # plain http only for this computer
        {"colour": "#0a7d5a"},  # styling lives in the style block only
        {"style": {"accent": "green"}},
        {"style": {"font": "x; } body { display: none"}},
        {"style": {"position": "top"}},
        {"style": {"logo_url": "http://clinic.example/logo.png"}},
        {"privacy_url": "javascript:alert(1)"},
        {"privacy_url": "http://clinic.example/privacy"},
        {"texts": {"sq": web_texts("sq"), "it": web_texts("it")}},  # no English
        {"texts": {lang: {**web_texts(lang), "send": " "} for lang in ("sq", "it", "en")}},
        {
            "texts": {
                lang: {**web_texts(lang), "notice": "No number"} for lang in ("sq", "it", "en")
            }
        },
        {"unknown": 1},
    ],
)
def test_a_bad_web_section_is_refused(
    businesses_dir: Path, tmp_path: Path, changes: dict[str, Any]
) -> None:
    with pytest.raises(PackError, match=r"web|origin|style|privacy|notice"):
        _load(businesses_dir, tmp_path, web_section(**changes))


def test_a_client_styles_its_widget_in_its_pack(businesses_dir: Path, tmp_path: Path) -> None:
    style = {
        "accent": "#0a7d5a",
        "on_accent": "#ffffff",
        "font": "'Source Sans 3', Arial, sans-serif",
        "position": "left",
        "logo_url": "https://clinic.example/logo.png",
    }
    web = _load(businesses_dir, tmp_path, web_section(style=style))
    assert web is not None
    assert web.style is not None
    assert web.style.model_dump() == style
