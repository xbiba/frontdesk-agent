"""Load a business pack directory into a validated BusinessPack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from frontdesk.tenancy.pack import BusinessPack

_DATA_FILES = ("location", "services", "doctors", "prices", "orientation", "orders")


class PackError(ValueError):
    """A business pack is missing, malformed or inconsistent."""


def load_pack(directory: Path) -> BusinessPack:
    if not directory.is_dir():
        raise PackError(f"business pack directory not found: {directory}")
    business = _read_yaml(directory / "business.yaml")
    if not isinstance(business, dict):
        raise PackError(f"{directory / 'business.yaml'}: expected a mapping")
    raw: dict[str, Any] = dict(business)
    for name in _DATA_FILES:
        path = directory / "data" / f"{name}.yaml"
        if path.exists():
            raw[name] = _read_yaml(path)
    raw["documents"] = [
        _read_document(path) for path in sorted((directory / "knowledge").glob("*.md"))
    ]
    try:
        return BusinessPack.model_validate(raw)
    except ValidationError as error:
        raise PackError(str(error)) from error


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise PackError(f"missing file: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise PackError(f"{path}: invalid YAML") from error


def _read_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or text.count("---\n") < 2:
        raise PackError(f"{path}: knowledge documents start with YAML front matter")
    _, front, body = text.split("---\n", 2)
    meta = yaml.safe_load(front)
    if not isinstance(meta, dict):
        raise PackError(f"{path}: front matter must be a mapping")
    return {**meta, "key": path.stem, "body": body.strip()}
