"""The document corpus the repository's guard tests walk, and the primitives for reading it.

The roots live here as named walkers a caller composes, so every guard answers "which files does
this check scan" the same way. `REPO_ROOT` lives here too.

Not named `conftest.py` on purpose, and not collected by pytest, being `_`-prefixed. The dividing
line with `tests/conftest.py` is the subject: that module holds what a test *drives the system
with*; this one holds what a guard *reads the repository with*.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
"""The repository root, one level above ``tests/``."""

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
"""A fenced code block's delimiter: a run of three or more backticks or tildes, as CommonMark
spells it. A fence closes only on a run of the same character at least as long as its opener, so a
four-backtick fence can show a three-backtick example."""


def markdown_under(*roots: str) -> list[Path]:
    """Every ``.md`` file under each named root, recursively, plus the root-level ones.

    Args:
        roots: Directory names relative to :data:`REPO_ROOT`. ``"."`` means the root's own
            top-level ``*.md`` and nothing below it.

    Returns:
        Absolute paths, sorted within each root, in the order the roots were given.
    """
    found: list[Path] = []
    for root in roots:
        if root == ".":
            found.extend(sorted(REPO_ROOT.glob("*.md")))
        else:
            found.extend(sorted((REPO_ROOT / root).rglob("*.md")))
    return [p for p in found if p.is_file()]


def python_under(root: str) -> list[Path]:
    """Every ``.py`` file under one root, recursively.

    A raw glob rather than a tracked-set filter: a filter would blind every consumer to a new
    module that is not staged yet, which is the state a developer's tree is in when they run the
    gate before committing.

    Args:
        root: A directory name relative to :data:`REPO_ROOT`.

    Returns:
        Absolute paths, sorted.
    """
    return [p for p in sorted((REPO_ROOT / root).rglob("*.py")) if p.is_file()]


def relative(paths: list[Path], *, without: Path | None = None) -> list[Path]:
    """Repo-relative forms of the paths that exist, dropping one by absolute path.

    The existence filter is load-bearing: a caller that appends a path by name, such as a
    generated file that may be absent, relies on it rather than on the walkers, so a missing file
    does not raise ``FileNotFoundError`` for every parametrized case reading it.

    Args:
        paths: Absolute paths under :data:`REPO_ROOT`.
        without: A path to exclude, or ``None``.

    Returns:
        The repo-relative paths that are files, in the order given.
    """
    return [
        p.relative_to(REPO_ROOT) for p in paths if p.is_file() and (without is None or p != without)
    ]


def prose_lines(text: str) -> list[str]:
    """Lines outside fenced code blocks.

    A document that shows markdown or YAML in a fence would otherwise appear to declare extra
    sections and a second frontmatter block, and a mermaid or shell snippet can hold bracket pairs
    that parse as links but are not.

    Args:
        text: A document's full text.

    Returns:
        The lines outside any fence, fence delimiters dropped.
    """
    return [line for _, line in numbered_prose_lines(text)]


def numbered_prose_lines(text: str) -> list[tuple[int, str]]:
    """:func:`prose_lines`, with each line's 1-indexed number.

    The numbering is of the *original* text, so a caller can report a line a reader can find.

    Args:
        text: A document's full text.

    Returns:
        ``(line number, line)`` for each line outside any fence.
    """
    return _scan_fences(text)[0]


def fences_balanced(text: str) -> bool:
    """Whether every fence in ``text`` closes, so no fence silently swallows the rest of it.

    Args:
        text: A document's full text.

    Returns:
        ``True`` when the scan ends outside a fence.
    """
    return _scan_fences(text)[1] is None


def _scan_fences(text: str) -> tuple[list[tuple[int, str]], str | None]:
    """Split ``text`` into numbered prose lines and the fence left open at the end, if any.

    Args:
        text: A document's full text.

    Returns:
        The ``(line number, line)`` pairs outside any fence, and the opening run of a fence that
        never closed, or ``None``.
    """
    out: list[tuple[int, str]] = []
    opener: str | None = None
    for number, line in enumerate(text.split("\n"), 1):
        match = FENCE.match(line)
        if opener is None and match:
            opener = match.group(1)
            continue
        if opener is not None:
            run = line.strip()
            if match and set(run) == {opener[0]} and len(run) >= len(opener):
                opener = None
            continue
        out.append((number, line))
    return out, opener


DUPLICATE_KEY = "<duplicated>"
"""Substituted for a frontmatter value whose key appears twice, so the duplicate is not silently
unioned or last-wins. Not a legal value of any field, so a caller that forwards it fails its own
check rather than needing to know about this one.
"""

HEADING = re.compile(r"^#{1,6}\s+(.*)$")
"""An ATX heading line; group 1 is the heading text as written."""


def slug(heading: str) -> str:
    """GitHub's heading anchor: strip formatting, drop punctuation, space to hyphen.

    Each space maps to its own hyphen, so an em dash between two spaces yields a double hyphen.

    Args:
        heading: The heading text, with any inline markdown.

    Returns:
        The anchor a reader can target.
    """
    text = heading.strip()
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text.lower(), flags=re.UNICODE)
    return text.replace(" ", "-")


@functools.cache
def anchors_of(path: Path) -> frozenset[str]:
    """Every anchor a reader can target in one document.

    A heading whose slug repeats an earlier one also answers to GitHub's de-duplicated form,
    ``<slug>-1``, ``<slug>-2`` and so on in document order. Cached because several guard modules
    ask for the same document in one session, and each is parametrized over a corpus.

    Args:
        path: The document, relative to the repository root or absolute.

    Returns:
        The slug of every heading outside a fenced code block, plus each repeat's suffixed form.
    """
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    seen: Counter[str] = Counter()
    found: set[str] = set()
    for _, line in numbered_prose_lines(text):
        if (m := HEADING.match(line)) is None:
            continue
        base = slug(m.group(1))
        found.add(base if seen[base] == 0 else f"{base}-{seen[base]}")
        seen[base] += 1
    return frozenset(found)
