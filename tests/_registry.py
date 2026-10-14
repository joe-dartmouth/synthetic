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

DEFINITION = re.compile(r"^\[([a-z][a-z0-9-]*)\]:\s*(\S+)\s*$")
"""A markdown link definition, ``[key]: target``."""

ENTRY_KEY = re.compile(r"`([a-z][a-z0-9-]*-(?:(?:19|20)[0-9]{2}[a-z]?|nd))`\s*·")
"""A registry entry: a key followed by a middot and its identifiers.

A key ends in its four-digit year, with an optional letter to disambiguate, or in ``nd`` when no
source gives a year (`docs/literature.md`, *How this registry works*).
"""


@functools.cache
def read(path: Path) -> str:
    """The document's text, cached because callers re-read the same registry per parameter."""
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def definitions(text: str) -> list[tuple[str, str]]:
    """Link definitions in document order, as ``(key, target)`` pairs."""
    return [
        (m.group(1), m.group(2))
        for _, line in numbered_prose_lines(text)
        if (m := DEFINITION.match(line))
    ]
