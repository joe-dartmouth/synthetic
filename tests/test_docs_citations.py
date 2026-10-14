"""Consistency tests for the hand-authored documentation set.

Citations resolve to `docs/literature.md`, link-definition blocks are sorted and complete, DOIs are
not truncated at a parenthesis, relative links and intra-repo anchors resolve, and every `§N`
citation of another document names a section that document has.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._corpus import (
    REPO_ROOT,
    anchors_of,
    markdown_under,
    numbered_prose_lines,
    relative,
)
from tests._registry import (
    CITING_DOCS,
    ENTRY_KEY,
    REGISTRY,
    definitions,
    read,
)
from tests._section_citations import section_citations, unresolved_section_citations


# ``[`key`][key]`` -- the label is itself a code span, so this matches the target half.
CITATION = re.compile(r"\]\[([a-z][a-z0-9-]*)\]")
ANCHOR_LINK = re.compile(r"\]\(([^)\s]*#[^)\s]+)\)")
PATH_LINK = re.compile(r"\]\(([^)\s]+)\)")
TRUNCATED_DOI = re.compile(r"\]\(https://doi\.org/[^)]*\([^)]*\)")
INLINE_CODE = re.compile(r"`[^`]*`")
EXTERNAL = ("http://", "https://", "mailto:")


def doc_files() -> list[Path]:
    """Every hand-authored markdown file, repo-relative.

    Only the named roots are walked, which keeps ``.venv`` and ``scratchpad`` out.
    ``docs/adr/template.md`` is excluded because every link in it is a placeholder.
    """
    found = markdown_under(".", "docs", "src", "tasks", "tools")
    return relative(found, without=REPO_ROOT / "docs" / "adr" / "template.md")


def linkable_lines(text: str) -> list[tuple[int, str]]:
    """Numbered prose lines with inline code spans removed.

    A code span is a rendered example rather than live markup, and a plain-text gloss such as
    ``cost[i][j]`` parses as a citation. Code spans here sit inside link labels, never targets, so
    stripping them keeps every live target intact.
    """
    return [(number, INLINE_CODE.sub("", line)) for number, line in numbered_prose_lines(text)]


def citations(text: str) -> set[str]:
    """Keys cited reference-style."""
    return {key for _, line in linkable_lines(text) for key in CITATION.findall(line)}


ALL_DOCS = doc_files()
ANCHORS: dict[Path, frozenset[str]] = {p: anchors_of(p) for p in ALL_DOCS}
REGISTRY_KEYS = set(ENTRY_KEY.findall(read(REGISTRY)))


# --- citations resolve ----------------------------------------------------


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_every_cited_key_has_a_link_definition(doc: Path):
    """A citation with no definition renders as literal ``[`key`][key]`` text."""
    defined = {key for key, _ in definitions(read(doc))}
    undefined = sorted(citations(read(doc)) - defined)
    assert not undefined, f"{doc} cites keys it never defines: {undefined}"


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_no_link_definition_is_unused(doc: Path):
    """A definition for a key the document no longer cites is stale weight."""
    cited = citations(read(doc))
    unused = sorted({key for key, _ in definitions(read(doc))} - cited)
    assert not unused, f"{doc} defines keys it never cites: {unused}"


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_every_defined_key_has_a_registry_entry(doc: Path):
    """The registry is the single home for a citation, so a key invented at the point of use fails."""
    unregistered = sorted({key for key, _ in definitions(read(doc))} - REGISTRY_KEYS)
    assert not unregistered, f"{doc} defines keys absent from {REGISTRY}: {unregistered}"


# --- definition blocks are tidy -------------------------------------------


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_link_definitions_are_alphabetically_sorted(doc: Path):
    keys = [key for key, _ in definitions(read(doc))]
    assert keys == sorted(keys), f"{doc}'s link block is not sorted"


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_no_key_is_defined_twice(doc: Path):
    keys = [key for key, _ in definitions(read(doc))]
    duplicated = sorted({key for key in keys if keys.count(key) > 1})
    assert not duplicated, f"{doc} defines these keys more than once: {duplicated}"


# --- links resolve --------------------------------------------------------


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_no_doi_link_is_truncated_by_a_parenthesis(doc: Path):
    """Markdown link parsing stops at the first unescaped ``)``, so a DOI with parentheses must be
    percent-encoded (``%28``/``%29``).
    """
    offenders = [n for n, line in numbered_prose_lines(read(doc)) if TRUNCATED_DOI.search(line)]
    assert not offenders, f"{doc} has DOI links truncated at a parenthesis on lines {offenders}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_every_relative_link_target_exists(doc: Path):
    """A relative link to a moved or renamed file is a dead link on GitHub."""
    broken: list[str] = []
    for number, line in linkable_lines(read(doc)):
        for target in PATH_LINK.findall(line):
            if target.startswith(EXTERNAL) or target.startswith("#"):
                continue
            path = target.partition("#")[0]
            if not path:
                continue
            if not (REPO_ROOT / doc).parent.joinpath(path).exists():
                broken.append(f"line {number}: {target}")
    assert not broken, f"{doc} links to missing files: {broken}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_every_section_number_citation_resolves(doc: Path):
    """A `§N` into another document must name a section that document has.

    Reads raw text rather than fence-stripped lines: a `§N` citation has no plain-text lookalike,
    and one inside a fence is still a citation that should resolve. A bare `§N` naming no document
    is a self-reference and is out of scope.
    """
    problems = unresolved_section_citations(section_citations(read(doc)), cited_from=doc)
    assert not problems, f"{doc} cites sections that do not resolve: {problems}"


def test_the_section_citation_resolver_refuses_a_number_that_is_not_there():
    """Positive control, because every case above passes on a corpus that resolves.

    One synthetic citation per shape against `docs/RESEARCH.md`, which numbers its headings, plus
    the branches the live corpus cannot reach: a wrong italic title, a lettered subsection, a link
    whose text and target disagree, a nested link, and a relative target.
    """
    for shape in (
        "`RESEARCH.md` §99",
        "[`RESEARCH.md`](RESEARCH.md) §99",
        "[`RESEARCH.md` §99](RESEARCH.md)",
    ):
        citations = section_citations(f"Tracked in {shape}, which is where it lives.")
        assert citations, f"the resolver no longer matches the shape {shape!r}"
        assert unresolved_section_citations(citations), f"{shape!r} must not resolve"

    refused = unresolved_section_citations(
        section_citations("See `RESEARCH.md` §4 *Nothing at all* for it.")
    )
    assert refused, "a title that is not the cited section's title must be refused"
    assert "that section is" in refused[0], (
        f"the refusal came from the wrong branch: {refused[0]!r}"
    )

    assert section_citations("See `docs/RESEARCH.md` §4e, a subsection.") == []

    disagreeing = section_citations("See [`RESEARCH.md`](ROADMAP.md) §4 for the rule.")
    assert disagreeing == [("ROADMAP.md", "4", "")], f"the link target must win, got {disagreeing}"
    assert unresolved_section_citations(disagreeing), "ROADMAP.md has no numbered section 4"

    assert section_citations("See [`RESEARCH.md` and [a note](ROADMAP.md) §4] for it.") == []

    rel = section_citations("See [`RESEARCH.md` §4](../RESEARCH.md#4-success-criteria).")
    assert unresolved_section_citations(rel, cited_from=Path("docs/adr/README.md")) == []
    assert unresolved_section_citations(rel), "a relative target needs the citing document"

    resolving = section_citations("Tracked in [`RESEARCH.md`](RESEARCH.md) §4.")
    assert resolving, "precondition: the live shape still matches"
    assert unresolved_section_citations(resolving) == [], "an existing section must resolve"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_every_intra_repo_anchor_resolves(doc: Path):
    """An anchor survives a heading rename silently, so only a check catches it.

    Links into a file outside the hand-authored set are skipped here -- their existence is
    covered by :func:`test_every_relative_link_target_exists` -- as are anchors into
    non-markdown targets such as ``#page=3`` on a PDF.
    """
    broken: list[str] = []
    for number, line in linkable_lines(read(doc)):
        for target in ANCHOR_LINK.findall(line):
            if target.startswith(EXTERNAL):
                continue
            path, _, fragment = target.partition("#")
            if not path:
                destination = doc
            else:
                resolved = (REPO_ROOT / doc).parent.joinpath(path).resolve()
                if not resolved.is_relative_to(REPO_ROOT):
                    continue
                destination = resolved.relative_to(REPO_ROOT)
            if destination not in ANCHORS:
                continue
            if fragment not in ANCHORS[destination]:
                broken.append(f"line {number}: {target} (no such heading in {destination})")
    assert not broken, f"{doc} has unresolved anchors: {broken}"


# --- the corpus itself ----------------------------------------------------


def test_the_document_set_is_not_silently_empty():
    """Every check above is parametrized, so an empty corpus would pass them all.

    A glob that stops matching -- a renamed directory, a move out of ``docs/`` -- would
    turn this module into a no-op that still reports green. These floors are deliberately
    below the current counts, so they pin the discovery rather than the content.
    """
    assert len(ALL_DOCS) >= 15, f"only found {len(ALL_DOCS)} docs; discovery is likely broken"
    assert REGISTRY in ALL_DOCS
    assert Path("tools/README.md") in ALL_DOCS, "the tools/ root is no longer walked"
    for doc in CITING_DOCS:
        assert doc in ALL_DOCS
    assert Path("tasks/proposal-index.md") in ALL_DOCS, "the tasks/ root is no longer walked"
    assert len(REGISTRY_KEYS) >= 10, f"only parsed {len(REGISTRY_KEYS)} registry keys"
    assert sum(len(a) for a in ANCHORS.values()) >= 50
