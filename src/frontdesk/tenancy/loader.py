"""Load a business pack directory into a validated BusinessPack."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from frontdesk.tenancy.pack import BusinessPack

_DATA_FILES = ("location", "services", "doctors", "prices", "orientation", "orders")


class PackError(ValueError):
    """A business pack is missing, malformed or inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that refuses a mapping key written twice.

    PyYAML keeps the last of two equal keys without a word, so a second `handoff:` line would
    replace a fixed text, and a pack that looks right on screen would load with another meaning.
    Keys that a merge (`<<`) brings in are not counted: a key written beside a merge overrides it.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        first: dict[Hashable, yaml.Mark] = {}
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                continue  # the standard error for it follows below
            if key in first:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"found duplicate key {key!r} (first at line {first[key].line + 1})",
                    key_node.start_mark,
                )
            first[key] = key_node.start_mark
        return super().construct_mapping(node, deep=deep)


def _safe_load(text: str) -> Any:
    """`yaml.safe_load` with the one change above."""
    loader = _UniqueKeyLoader(text)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def load_pack(directory: Path) -> BusinessPack:
    if not directory.is_dir():
        raise PackError(f"business pack directory not found: {directory}")
    business_file = directory / "business.yaml"
    business = _read_yaml(business_file, directory)
    if not isinstance(business, dict):
        raise PackError(f"{_name(business_file, directory)}: expected a mapping")
    raw: dict[str, Any] = dict(business)
    for name in _DATA_FILES:
        path = directory / "data" / f"{name}.yaml"
        if path.exists():
            raw[name] = _read_yaml(path, directory)
    raw["documents"] = [
        _read_document(path, directory) for path in sorted((directory / "knowledge").glob("*.md"))
    ]
    try:
        return BusinessPack.model_validate(raw)
    except ValidationError as error:
        raise PackError(str(error)) from error


def _name(path: Path, root: Path) -> str:
    """A pack file as error messages show it: its path inside the pack."""
    return path.relative_to(root).as_posix()


def _load[T](path: Path, root: Path, parse: Callable[[str], T]) -> T:
    """Read a pack file as UTF-8 text and parse it.

    Text that does not decode, YAML that does not parse, a value that cannot exist (the 30th of
    February) and a file that cannot be read all become PackErrors that name the file; a YAML error
    also gives the line and column, counted in the file. A leading byte-order mark (some Windows
    editors add one) is ignored.
    """
    try:
        return parse(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as error:  # a ValueError, so it comes first
        raise PackError(
            f"{_name(path, root)}: not valid UTF-8 text (save the file as UTF-8)"
        ) from error
    except yaml.YAMLError as error:
        raise PackError(f"{_name(path, root)}: invalid YAML{_where(error)}") from error
    except ValueError as error:
        raise PackError(f"{_name(path, root)}: invalid value: {error}") from error
    except OSError as error:
        reason = error.strerror or type(error).__name__  # never the path: it is the server's
        raise PackError(f"{_name(path, root)}: cannot be read ({reason})") from error


def _where(error: yaml.YAMLError) -> str:
    """` at line L, column C: reason` for an error PyYAML placed; `: reason` for one it did not."""
    if isinstance(error, yaml.MarkedYAMLError):
        mark = error.problem_mark or error.context_mark
        if mark is not None:
            reason = ", ".join(part for part in (error.context, error.problem) if part)
            return f" at line {mark.line + 1}, column {mark.column + 1}: {reason}"
    return ": " + " ".join(str(error).split())


def _read_yaml(path: Path, root: Path) -> Any:
    if not path.exists():
        raise PackError(f"missing file: {_name(path, root)}")
    return _load(path, root, _safe_load)


def _read_document(path: Path, root: Path) -> dict[str, Any]:
    name = _name(path, root)
    split = _load(path, root, _split_document)
    if split is None:
        raise PackError(f"{name}: knowledge documents start with YAML front matter")
    meta, body = split
    if not isinstance(meta, dict):
        raise PackError(f"{name}: front matter must be a mapping")
    return {**meta, "key": path.stem, "body": body.strip()}


def _split_document(text: str) -> tuple[Any, str] | None:
    """The parsed front matter and the body of a Markdown document; None without front matter."""
    if not text.startswith("---\n") or text.count("---\n") < 2:
        return None
    _, front, body = text.split("---\n", 2)
    # The front matter starts on line 2. A blank line stands in for the opening delimiter, so the
    # line numbers in a YAML error are the file's.
    return _safe_load("\n" + front), body
