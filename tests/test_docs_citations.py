"""Consistency tests for the hand-authored documentation set.

Citations resolve to `docs/literature.md` and carry its URLs, link-definition blocks are sorted and
complete, DOIs are not truncated at a parenthesis, relative links, link definitions and intra-repo
anchors resolve, every `§N` citation of another document names a section that document has, and a
link whose text says "section N" targets section N. Every failure names its line.
"""

from __future__ import annotations

import re
from itertools import pairwise
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
    numbered_definitions,
    read,
    registry_entries,
)
from tests._section_citations import (
    numbered_section_citations,
    section_citations,
    section_link_mismatches,
    unresolved_section_citations,
)


# ``[`key`][key]`` -- the label is itself a code span, so this matches the target half.
CITATION = re.compile(r"\]\[([a-z][a-z0-9-]*)\]")
ANCHOR_LINK = re.compile(r"\]\(([^)\s]*#[^)\s]+)\)")
PATH_LINK = re.compile(r"\]\(([^)\s]+)\)")
TRUNCATED_DOI = re.compile(r"\]\(https://doi\.org/[^)]*\([^)]*\)")
ANY_DEFINITION = re.compile(r"^ {0,3}\[[^\]\n]+\]:\s*<?([^>\s]+)>?")
"""Any reference-style link definition, whatever its label, capturing the target."""
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


def citations(text: str) -> dict[str, int]:
    """Keys cited reference-style, each with the line of its first citation."""
    found: dict[str, int] = {}
    for number, line in linkable_lines(text):
        for key in CITATION.findall(line):
            found.setdefault(key, number)
    return found


ALL_DOCS = doc_files()
ANCHORS: dict[Path, frozenset[str]] = {p: anchors_of(p) for p in ALL_DOCS}
REGISTRY_ENTRIES = registry_entries(read(REGISTRY))
REGISTRY_KEYS = {key for _, key, _ in REGISTRY_ENTRIES}
REGISTRY_URLS = {key: url for _, key, url in REGISTRY_ENTRIES}

CORPUS_WITNESSES = (
    Path("AGENTS.md"),
    Path("docs/RESEARCH.md"),
    Path("src/bko/bkb/README.md"),
    Path("tasks/proposal-index.md"),
    Path("tools/README.md"),
)
"""One file per root :func:`doc_files` walks, so a root that stops matching fails by name."""


# --- citations resolve ----------------------------------------------------


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_every_cited_key_has_a_link_definition(doc: Path):
    """A citation with no definition renders as literal ``[`key`][key]`` text."""
    defined = {key for key, _ in definitions(read(doc))}
    undefined = [
        f"line {line}: {key}" for key, line in citations(read(doc)).items() if key not in defined
    ]
    assert not undefined, f"{doc} cites keys it never defines: {undefined}"


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_no_link_definition_is_unused(doc: Path):
    """A definition for a key the document no longer cites is stale weight."""
    cited = citations(read(doc))
    unused = [
        f"line {n}: {key}" for n, key, _ in numbered_definitions(read(doc)) if key not in cited
    ]
    assert not unused, f"{doc} defines keys it never cites: {unused}"


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_every_defined_key_has_a_registry_entry(doc: Path):
    """The registry is a citation's single home, so a key invented at the point of use fails."""
    unregistered = [
        f"line {n}: {key}"
        for n, key, _ in numbered_definitions(read(doc))
        if key not in REGISTRY_KEYS
    ]
    assert not unregistered, f"{doc} defines keys absent from {REGISTRY}: {unregistered}"


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_every_definition_points_where_its_registry_entry_does(doc: Path):
    """A definition is a copy of the entry's canonical URL, so the two must not drift apart."""
    drifted = [
        f"line {n}: {key} -> {target}, registry has {REGISTRY_URLS[key]}"
        for n, key, target in numbered_definitions(read(doc))
        if key in REGISTRY_URLS and REGISTRY_URLS[key] != target
    ]
    assert not drifted, f"{doc} defines keys whose URL disagrees with {REGISTRY}: {drifted}"


def test_no_registry_key_is_listed_twice():
    """Two entries under one key make every citation of it ambiguous."""
    lines: dict[str, list[int]] = {}
    for number, key, _ in REGISTRY_ENTRIES:
        lines.setdefault(key, []).append(number)
    duplicated = {key: found for key, found in lines.items() if len(found) > 1}
    assert not duplicated, f"{REGISTRY} lists these keys more than once (lines): {duplicated}"


def test_registry_entries_reports_a_duplicate_key():
    """Positive control: the parser keeps both copies, so the check above can see them."""
    text = "- `smith-2020` · [a](https://a) · x\n- `smith-2020` · [b](https://b) · y\n"
    assert registry_entries(text) == [
        (1, "smith-2020", "https://a"),
        (2, "smith-2020", "https://b"),
    ]


@pytest.mark.parametrize(
    ("key", "matches"),
    [
        ("smith-2020", True),
        ("smith-2020b", True),
        ("abecto-nd", True),
        ("abecto-ndb", True),
        ("smith-20", False),
        ("smith-ndx2", False),
    ],
)
def test_the_entry_key_pattern_keeps_every_verdict(key: str, matches: bool):
    """A year or ``nd``, each with an optional disambiguating letter, and the near-misses."""
    assert bool(ENTRY_KEY.search(f"- `{key}` · id")) is matches


def test_a_definition_may_carry_a_title():
    assert definitions('[smith-2020]: https://a "A title"\n') == [("smith-2020", "https://a")]


# --- definition blocks are tidy -------------------------------------------


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_link_definitions_are_alphabetically_sorted(doc: Path):
    found = numbered_definitions(read(doc))
    for (_, before, _), (line, key, _) in pairwise(found):
        assert before <= key, f"{doc}'s link block is not sorted: line {line}: {key} after {before}"


@pytest.mark.parametrize("doc", CITING_DOCS, ids=str)
def test_no_key_is_defined_twice(doc: Path):
    lines: dict[str, list[int]] = {}
    for number, key, _ in numbered_definitions(read(doc)):
        lines.setdefault(key, []).append(number)
    duplicated = {key: found for key, found in lines.items() if len(found) > 1}
    assert not duplicated, f"{doc} defines these keys more than once (lines): {duplicated}"


# --- links resolve --------------------------------------------------------


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_no_doi_link_is_truncated_by_a_parenthesis(doc: Path):
    """Markdown link parsing stops at the first unescaped ``)``, so a DOI with parentheses must be
    percent-encoded (``%28``/``%29``).
    """
    offenders = [n for n, line in numbered_prose_lines(read(doc)) if TRUNCATED_DOI.search(line)]
    assert not offenders, f"{doc} has DOI links truncated at a parenthesis on lines {offenders}"


@pytest.mark.parametrize(
    ("line", "truncated"),
    [
        ("[x](https://doi.org/10.1016/S0140-6736(20)30183-5)", True),
        ("[x](https://doi.org/10.1016/S0140-6736%2820%2930183-5)", False),
    ],
    ids=["raw-parenthesis", "percent-encoded"],
)
def test_the_truncated_doi_pattern_keeps_every_verdict(line: str, truncated: bool):
    """Positive control: the live corpus is clean, so only a planted DOI shows the check bites."""
    assert bool(TRUNCATED_DOI.search(line)) is truncated


def broken_targets(doc: Path, targets: list[tuple[int, str]]) -> list[str]:
    """The relative targets, resolved from ``doc``'s directory, that name no file."""
    broken: list[str] = []
    for number, target in targets:
        if target.startswith(EXTERNAL) or target.startswith("#"):
            continue
        path = target.partition("#")[0]
        if path and not (REPO_ROOT / doc).parent.joinpath(path).exists():
            broken.append(f"line {number}: {target}")
    return broken


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_every_relative_link_target_exists(doc: Path):
    """A relative link to a moved or renamed file is a dead link on GitHub."""
    targets = [(n, t) for n, line in linkable_lines(read(doc)) for t in PATH_LINK.findall(line)]
    broken = broken_targets(doc, targets)
    assert not broken, f"{doc} links to missing files: {broken}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_every_reference_definition_target_exists(doc: Path):
    """``[label]: target`` is a link too, and inline-link parsing never sees it."""
    targets = [
        (n, m.group(1))
        for n, line in numbered_prose_lines(read(doc))
        if (m := ANY_DEFINITION.match(line))
    ]
    broken = broken_targets(doc, targets)
    assert not broken, f"{doc} defines links to missing files: {broken}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_every_section_number_citation_resolves(doc: Path):
    """A `§N` into another document must name a section that document has.

    Reads raw text rather than fence-stripped lines: a `§N` citation has no plain-text lookalike,
    and one inside a fence is still a citation that should resolve. A bare `§N` naming no document
    is a self-reference and is out of scope.
    """
    problems: list[str] = []
    for number, citation in numbered_section_citations(read(doc)):
        for problem in unresolved_section_citations([citation], cited_from=doc):
            if (found := f"line {number}: {problem}") not in problems:
                problems.append(found)
    assert not problems, f"{doc} cites sections that do not resolve: {problems}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=str)
def test_every_section_link_names_the_section_it_targets(doc: Path):
    """``[RESEARCH section 4](RESEARCH.md#5-...)`` resolves as an anchor and still misleads."""
    problems = section_link_mismatches(read(doc))
    assert not problems, f"{doc} has links naming one section and targeting another: {problems}"


# --- positive controls for the section resolvers ---------------------------
#
# Every repository case above passes on a corpus that resolves, so each branch gets a synthetic
# citation of its own, against `docs/RESEARCH.md`, which numbers its headings. One test per branch,
# so one regression does not mask another, and each refusal pins its branch by message.

SHAPES = (
    "`RESEARCH.md` {n}",
    "[`RESEARCH.md`](RESEARCH.md) {n}",
    "[`RESEARCH.md` {n}](RESEARCH.md)",
)
"""The three `§N` shapes, with the section number to plant."""


@pytest.mark.parametrize("shape", SHAPES)
def test_a_missing_section_number_is_refused(shape: str):
    found = section_citations(f"Tracked in {shape.format(n='§99')}, which is where it lives.")
    # A set: the bracketed shape also matches the code-span shape, and the resolver dedupes.
    assert set(found) == {("RESEARCH.md", "99", "")}, f"the shape {shape!r} no longer matches"
    refused = unresolved_section_citations(found)
    assert len(refused) == 1, refused
    assert "has sections" in refused[0], f"the refusal came from the wrong branch: {refused[0]!r}"


@pytest.mark.parametrize("shape", SHAPES)
def test_a_lettered_multi_digit_subsection_is_not_a_section(shape: str):
    """`§11e` must not backtrack to `§1`; the letter-only lookahead failed exactly this."""
    assert section_citations(f"See {shape.format(n='§11e')}, a subsection.") == []


@pytest.mark.parametrize("shape", SHAPES)
def test_an_existing_section_resolves(shape: str):
    found = section_citations(f"Tracked in {shape.format(n='§4')}.")
    # A set: the bracketed shape also matches the code-span shape, and the resolver dedupes.
    assert set(found) == {("RESEARCH.md", "4", "")}, f"the shape {shape!r} no longer matches"
    assert unresolved_section_citations(found) == []


def test_a_wrong_title_is_refused():
    refused = unresolved_section_citations(
        section_citations("See `RESEARCH.md` §4 *Nothing at all* for it.")
    )
    assert len(refused) == 1, refused
    assert "that section is" in refused[0], (
        f"the refusal came from the wrong branch: {refused[0]!r}"
    )


def test_the_link_target_wins_over_the_link_text():
    disagreeing = section_citations("See [`RESEARCH.md`](ROADMAP.md) §4 for the rule.")
    assert disagreeing == [("ROADMAP.md", "4", "")], f"the link target must win, got {disagreeing}"
    refused = unresolved_section_citations(disagreeing)
    assert len(refused) == 1, refused
    assert "has no numbered" in refused[0], (
        f"the refusal came from the wrong branch: {refused[0]!r}"
    )


def test_a_nested_link_is_not_a_citation():
    assert section_citations("See [`RESEARCH.md` and [a note](ROADMAP.md) §4] for it.") == []


def test_a_relative_target_resolves_only_from_its_citing_document():
    rel = section_citations("See [`RESEARCH.md` §4](../RESEARCH.md#4-success-criteria).")
    assert unresolved_section_citations(rel, cited_from=Path("docs/adr/README.md")) == []
    refused = unresolved_section_citations(rel)
    assert len(refused) == 1, refused
    assert "no such document" in refused[0], (
        f"the refusal came from the wrong branch: {refused[0]!r}"
    )


def test_a_citation_reports_the_line_it_starts_on():
    body = "First line.\nSee `RESEARCH.md` §4 here.\n"
    assert numbered_section_citations(body) == [(2, ("RESEARCH.md", "4", ""))]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("[RESEARCH section 4](../docs/RESEARCH.md#5-evaluation-contract)", ["line 1:"]),
        ("[Section 9](#2-intended-users)", ["line 1:"]),
        ("[RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria)", []),
        ("[section 4](../docs/RESEARCH.md#approach)", []),
        ("[RESEARCH](../docs/RESEARCH.md#5-evaluation-contract)", []),
        ("```\n[section 4](RESEARCH.md#5-x)\n```", []),
    ],
    ids=["disagree", "disagree-in-document", "agree", "unnumbered-fragment", "no-number", "fenced"],
)
def test_a_section_link_must_target_the_number_it_names(body: str, expected: list[str]):
    problems = section_link_mismatches(body)
    assert [p[: len("line 1:")] for p in problems] == expected, problems
    assert all("names section" in p for p in problems), problems


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
    turn this module into a no-op that still reports green. These floors sit near half the
    current counts, so they pin the discovery rather than the content, and
    :data:`CORPUS_WITNESSES` pins which roots it reaches.
    """
    assert len(ALL_DOCS) >= 23, f"only found {len(ALL_DOCS)} docs; discovery is likely broken"
    assert REGISTRY in ALL_DOCS
    for doc in CITING_DOCS:
        assert doc in ALL_DOCS
    assert len(REGISTRY_KEYS) >= 23, f"only parsed {len(REGISTRY_KEYS)} registry keys"
    anchors = sum(len(a) for a in ANCHORS.values())
    assert anchors >= 125, f"only found {anchors} anchors"


@pytest.mark.parametrize("witness", CORPUS_WITNESSES, ids=str)
def test_every_walked_root_still_reaches_the_corpus(witness: Path):
    """A root that stops matching takes its whole contribution silently, floor or no floor."""
    assert witness in ALL_DOCS, (
        f"{witness} is absent from the corpus, so the root it represents is not being walked"
    )
