"""Helpers for tests that edit a copy of a demo business pack."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

CLINIC = "demo-clinic"
SHOP = "demo-shop"


PHONE = "+355 00 000 0000"  # the demo clinic's reception


def web_texts(language: str) -> dict[str, str]:
    return {
        "title": f"Chat {language}",
        "open": f"Open {language}",
        "close": f"Close {language}",
        "placeholder": f"Write {language}",
        "send": f"Send {language}",
        "notice": f"Automatic assistant {language}. Reception: {PHONE}.",
        "person": f"Talk to a person {language}",
        "privacy": f"Privacy {language}",
    }


def web_section(**changes: Any) -> dict[str, Any]:
    section: dict[str, Any] = {
        "origins": ["https://clinic.example", "http://localhost:8000"],
        "name": "Klinika Demo",
        "colour": "#0a7d5a",
        "privacy_url": "https://clinic.example/privacy",
        "texts": {language: web_texts(language) for language in ("sq", "it", "en")},
    }
    section.update(changes)
    return section


Change = Callable[[Any], None]


def copy_pack(businesses_dir: Path, name: str, tmp_path: Path) -> Path:
    target = tmp_path / name
    shutil.copytree(businesses_dir / name, target)
    return target


def edit_yaml(path: Path, change: Change) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    write_yaml(path, data)


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def edit_document(path: Path, change: Change) -> None:
    """Edit a knowledge document's front matter and keep its body."""
    _, front, body = path.read_text(encoding="utf-8").split("---\n", 2)
    meta = yaml.safe_load(front)
    change(meta)
    dumped = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
    path.write_text(f"---\n{dumped}---\n{body}", encoding="utf-8")
