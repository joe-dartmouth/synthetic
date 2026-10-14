"""The citation registry's shape: the file, the documents citing into it, and its parsers.

`docs/literature.md` holds every cited work. `tests/test_docs_citations.py` reads it through this
module to check that every reference-style citation resolves to a registry entry.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

from tests._corpus import REPO_ROOT, numbered_prose_lines


REGISTRY = Path("docs/literature.md")
"""The registry every citation resolves to."""

CITING_DOCS = (Path("docs/RESEARCH.md"), Path("docs/IDEAS.md"))
"""The documents that cite into it reference-style. Every other doc links by URL or by path."""

DEFINITION = re.compile(r'^\[([a-z][a-z0-9-]*)\]:\s*(\S+)(?:\s+"[^"\n]*")?\s*$')
"""A markdown link definition, ``[key]: target``, with an optional ``"title"`` after the target."""

ENTRY_KEY = re.compile(r"`([a-z][a-z0-9-]*-(?:(?:19|20)[0-9]{2}|nd)[a-z]?)`\s*·")
"""A registry entry: a key followed by a middot and its identifiers.

A key ends in its four-digit year or in ``nd`` when no source gives a year, either with an optional
letter to disambiguate (`docs/literature.md`, *How this registry works*).
"""

FIRST_LINK = re.compile(r"\]\(([^)\s]+)\)")
"""The target of the first inline link on a registry line: the entry's canonical URL."""


@functools.cache
def read(path: Path) -> str:
    """The document's text, cached because callers re-read the same registry per parameter."""
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def definitions(text: str) -> list[tuple[str, str]]:
    """Link definitions in document order, as ``(key, target)`` pairs."""
    return [(key, target) for _, key, target in numbered_definitions(text)]


def numbered_definitions(text: str) -> list[tuple[int, str, str]]:
    """Link definitions in document order, as ``(line, key, target)`` triples."""
    return [
        (number, m.group(1), m.group(2))
        for number, line in numbered_prose_lines(text)
        if (m := DEFINITION.match(line))
    ]


def registry_entries(text: str) -> list[tuple[int, str, str | None]]:
    """Registry entries in document order, as ``(line, key, canonical URL)`` triples.

    The URL is the entry's first inline link, or ``None`` for an entry that gives no identifier.
    A key listed twice appears twice, so a caller can refuse the duplicate.
    """
    entries: list[tuple[int, str, str | None]] = []
    for number, line in numbered_prose_lines(text):
        if (m := ENTRY_KEY.search(line)) is None:
            continue
        link = FIRST_LINK.search(line, m.end())
        entries.append((number, m.group(1), link.group(1) if link else None))
    return entries
