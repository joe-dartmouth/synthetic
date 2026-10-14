"""Resolve a document's cross-references to the numbered sections of another document.

`tests/test_docs_citations.py` drives this over the documentation. Two notations are checked.

**`§N`.** Three citation shapes are matched, because a document can name its target three ways: a
code span with the number beside it, a link with the number after it, and a link with the number
inside its text. A bare `§N` naming no document is a self-reference and is out of scope. A site may
match two shapes, so :func:`unresolved_section_citations` deduplicates its findings.

**A link whose text says "section N".** The corpus writes most cross-references this way, as
``[RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria)``. The anchor check proves the
fragment exists; :func:`section_link_mismatches` proves the number in the text is the number in
the fragment, which the anchor check cannot see.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._corpus import REPO_ROOT, numbered_prose_lines


# `(?![\dA-Za-z])` and not `(?![A-Za-z])`, which is worse than nothing. A lettered subsection
# such as `§11e` lives in a different numbering space from the `## N.` headings these resolve
# against. With only the letter excluded, `\d+` *backtracks*: `11` fails the lookahead on `e`, the
# engine retries `1`, whose next character is `1` rather than a letter, and the citation matches
# as `§1`, resolving silently against section 1.
# `re.compile(r"§(\d+)(?![A-Za-z])").findall("§11e")` returns `['1']`; excluding the digit too
# returns `[]`. The positive controls in `tests/test_docs_citations.py` plant `§11e` in every
# shape, so a regression to the letter-only lookahead fails.
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
            Without it a relative path cannot resolve at all: a citation written from a
            subdirectory, such as ``../RESEARCH.md`` from under ``docs/adr/`` or
            ``../docs/RESEARCH.md`` from ``tasks/``, matches none of the fixed roots below.

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
    return [citation for _, citation in numbered_section_citations(body)]


def numbered_section_citations(body: str) -> list[tuple[int, tuple[str, str, str]]]:
    """:func:`section_citations`, each with the 1-indexed line its match starts on."""

    def line_of(match: re.Match[str]) -> int:
        return body.count("\n", 0, match.start()) + 1

    def document(name: str, target: str) -> str:
        return target if target.partition("#")[0].endswith(".md") else name

    found = [
        (line_of(m), (m.group(1), m.group(2), m.group(3) or ""))
        for m in SECTION_CITATION.finditer(body)
    ]
    for m in LINKED_SECTION_CITATION.finditer(body):
        name, target, number, title = m.groups()
        found.append((line_of(m), (document(name, target), number, title or "")))
    for m in BRACKETED_SECTION_CITATION.finditer(body):
        name, number, title, target = m.groups()
        found.append((line_of(m), (document(name, target), number, title or "")))
    return found


INLINE_LINK = re.compile(r"\[([^\[\]\n]*)\]\(([^)\s]*)\)")
"""An inline link: its text (no nested brackets) and its target."""

SECTION_WORD = re.compile(r"\bsection\s+(\d+)\b", re.IGNORECASE)
"""The section number a link's text names: ``RESEARCH section 4``, ``Section 9``."""

NUMBERED_FRAGMENT = re.compile(r"(\d+)-")
"""A fragment naming a numbered heading's anchor, ``#4-success-criteria``."""


def section_link_mismatches(body: str) -> list[str]:
    """Links whose text names ``section N`` while the fragment targets a different number.

    Only a link whose fragment is a numbered heading's anchor is compared; a link saying "section
    4" into an unnumbered anchor has no second number to disagree with. Lines inside fences are
    skipped, since a fenced example is not a live link.

    Args:
        body: The citing document's text.

    Returns:
        ``line N: [text](target) ...`` per disagreeing link, in document order.
    """
    problems: list[str] = []
    for number, line in numbered_prose_lines(body):
        for text, target in INLINE_LINK.findall(line):
            named = SECTION_WORD.search(text)
            fragment = NUMBERED_FRAGMENT.match(target.partition("#")[2])
            if named and fragment and named.group(1) != fragment.group(1):
                problems.append(
                    f"line {number}: [{text}]({target}) names section {named.group(1)} "
                    f"but targets section {fragment.group(1)}"
                )
    return problems


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
