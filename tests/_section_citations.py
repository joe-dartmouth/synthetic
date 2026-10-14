"""Resolve a `§N` section-number citation against the headings of the document it names.

`tests/test_docs_citations.py` drives this over the documentation. Three citation shapes are
matched, because a document can name its target three ways: a code span with the number beside it,
a link with the number after it, and a link with the number inside its text. A bare `§N` naming no
document is a self-reference and is out of scope. A site may match two shapes, so
:func:`unresolved_section_citations` deduplicates its findings.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._corpus import REPO_ROOT


# `(?![\dA-Za-z])` and not `(?![A-Za-z])`, which was the first revision and is worse than nothing.
# `docs/MATH.md` numbers subsections `§11d` and `§11e`, a different space from the `## N.` headings
# these resolve against. With only the letter excluded, `\d+` *backtracks*: `11` fails the lookahead
# on `e`, the engine retries `1`, whose next character is `1` rather than a letter, and the citation
# matches as `§1` -- resolving silently against `## 1. Notation`. Measured:
# `re.compile(r"§(\d+)(?![A-Za-z])").findall("§11e")` returns `['1']`, while excluding the digit too
# returns `[]`. Refusing the whole run is the behaviour the docstrings claimed and did not have.
_NUMBER = r"§(\d+)(?![\dA-Za-z])"
# An italic title beside the number, optional: `§13 *Deliberately deferred*`.
_TITLE = r"(?:\s+\*([^*\n]+)\*)?"

SECTION_CITATION = re.compile(
    rf"`((?:docs/|tasks/)?[A-Za-z0-9_.-]+\.md)`(?:'s)?\s+(?:\*\*)?{_NUMBER}{_TITLE}"
)

"""A citation naming its document in a code span: `` `RESEARCH.md` §4 *Success criteria* ``.

Captures the document, the number and the optional italic title, so both are checked against the
target's `## N. Title` headings; cite with the title or only the number is checked. Only `docs/`
and `tasks/` prefixes are admitted."""

LINKED_SECTION_CITATION = re.compile(
    rf"\[`?([A-Za-z0-9_.-]+\.md)`?[^\[\]\n]*\]\(([^)\s]*)\)\s+(?:\*\*)?{_NUMBER}{_TITLE}"
)

"""A citation as a link with the number after it: ``[`RESEARCH.md`](RESEARCH.md) §4``.

Both the link text and the target are captured, and the target wins; the text is the fallback for a
target that is not a markdown path."""

BRACKETED_SECTION_CITATION = re.compile(
    rf"\[`?([A-Za-z0-9_.-]+\.md)`?\s+(?:\*\*)?{_NUMBER}{_TITLE}[^\[\]\n]*\]\(([^)\s]*)\)"
)

"""A citation with the number inside the link text.

For example ``[RESEARCH §4](RESEARCH.md#4-success-criteria)``.
"""

SECTION_SHAPES = (SECTION_CITATION, LINKED_SECTION_CITATION, BRACKETED_SECTION_CITATION)

"""Every pattern here, for a caller deriving a population rather than resolving one."""


def resolve_document(name: str, *, cited_from: Path | None = None) -> Path | None:
    """A cited document path, relative to the citing document or under a known root.

    Args:
        name: The path as it appears in the citation. A ``#fragment`` is stripped.
        cited_from: The document the citation was read from, whose directory is tried first.
            Without it a relative path cannot resolve at all: five live citations spell
            ``../RESEARCH.md`` or ``../../ROADMAP.md`` from under ``docs/adr/``, and the
            fixed roots below find none of them.

    Returns:
        The first existing candidate, or ``None``.
    """
    path = name.partition("#")[0]
    if not path:
        return None
    candidates = []
    if cited_from is not None:
        candidates.append((REPO_ROOT / cited_from).parent / path)
    candidates += [REPO_ROOT / path, REPO_ROOT / "docs" / path, REPO_ROOT / "tasks" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def numbered_headings(target: Path) -> dict[str, str]:
    """``## N. Title`` headings of ``target``, keyed by the number as written.

    Args:
        target: The document being cited into.

    Returns:
        Number to title. Empty when the document numbers no heading, which is itself a
        finding: a `§N` into such a document can never resolve.
    """
    return {
        match.group(1): match.group(2).strip()
        for match in re.finditer(
            r"^## (\d+)\.\s+(.*)$", target.read_text(encoding="utf-8"), re.MULTILINE
        )
    }


def section_citations(body: str) -> list[tuple[str, str, str]]:
    """Every section citation in ``body`` that names its document, in any of the three shapes.

    Args:
        body: The citing document's text.

    Returns:
        ``(document, number, title)`` triples; the title is ``""`` when none was given. The
        document is the link *target* where the shape has one, falling back to the link text
        when that target is not a markdown path.
    """
    found = [(name, number, title) for name, number, title in SECTION_CITATION.findall(body)]
    for name, target, number, title in LINKED_SECTION_CITATION.findall(body):
        found.append((target if target.partition("#")[0].endswith(".md") else name, number, title))
    for name, number, title, target in BRACKETED_SECTION_CITATION.findall(body):
        found.append((target if target.partition("#")[0].endswith(".md") else name, number, title))
    return found


def unresolved_section_citations(
    citations: list[tuple[str, str, str]], *, cited_from: Path | None = None
) -> list[str]:
    """The citations that do not resolve, each described once.

    Args:
        citations: ``(document, number, title)`` triples, from :func:`section_citations` and
            whatever corpus-specific shapes the caller adds.
        cited_from: Passed through to :func:`resolve_document` for a relative path.

    Returns:
        One string per distinct failure, in the order first seen; empty when every citation
        resolves. Deduplicated because a site can match two patterns, so a single defect
        would otherwise be reported twice.
    """
    problems: list[str] = []
    for name, number, title in citations:
        target = resolve_document(name, cited_from=cited_from)
        if target is None:
            problem = f"`{name}` §{number}: no such document"
        else:
            headings = numbered_headings(target)
            if not headings:
                problem = f"`{name}` §{number}: {name} has no numbered `## N.` headings"
            elif number not in headings:
                problem = f"`{name}` §{number}: {name} has sections {sorted(headings, key=int)}"
            elif title and title.strip().lower() not in headings[number].lower():
                problem = f"`{name}` §{number} *{title}*: that section is `{headings[number]}`"
            else:
                continue
        if problem not in problems:
            problems.append(problem)
    return problems
