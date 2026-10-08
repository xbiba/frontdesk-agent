"""Helpers for tests that edit a copy of a demo business pack."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

CLINIC = "demo-clinic"
SHOP = "demo-shop"

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
