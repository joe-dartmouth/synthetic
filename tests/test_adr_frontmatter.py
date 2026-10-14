"""Structural tests for the decision log under ``docs/adr/``.

The topic/slug scheme deliberately has **no index and no sequence counter**: records are
discovered by globbing ``docs/adr/**/*.md`` and their status and dates are read from YAML
frontmatter rather than parsed out of prose. That is what keeps two concurrent ADR writes
from colliding. The invariants an index would make checkable by eye live here instead.

No YAML parser is a dependency of this project and none was added for this. The frontmatter
block is flat ``key: value`` pairs, so it is hand-parsed between the first two ``---``
lines; a nested block would fail loudly rather than parse wrong.

Two of these tests scan ``src/`` and ``tests/`` rather than the decision log. Those are the
anti-regression pair: a docstring citation is prose, so a stale one is invisible to a
markdown link checker and lies silently forever.
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

import pytest

from tests._corpus import (
    DUPLICATE_KEY,
    REPO_ROOT,
    markdown_under,
    prose_lines,
    python_under,
    relative,
)


ADR_DIR = Path("docs") / "adr"
ADR_ROOT = REPO_ROOT / ADR_DIR

# Both match ``docs/adr/**/*.md`` and neither is a decision, so every enumeration drops them.
SCAFFOLD = frozenset({"README.md", "template.md"})

STATUSES = frozenset({"proposed", "accepted"})

# No ``supersedes``/``superseded-by``: a record whose decision no longer exists is deleted, so
# there is no file at either end of such a link. Asserting the exact key set -- rather than a
# subset -- is what stops a retired field drifting back in from an old record used as a model.
FRONTMATTER_KEYS = frozenset({"status", "topic", "created", "updated"})

REQUIRED_SECTIONS = (
    "Context",
    "Decision Drivers",
    "Decision",
    "Rationale",
    "Consequences",
    "Ruled Out",
    "Enforcement",
    "Reversal Conditions",
    "Related",
)

SECTION = re.compile(r"^##\s+(.+?)\s*$")
TOPIC_BULLET = re.compile(r"^[-*]\s+`([a-z][a-z0-9-]*)`")
NUMBERED_FILENAME = re.compile(r"^\d+[-_]")

# A four-digit numbered citation. Three shapes are caught: spaced or hyphenated, singular or
# plural (a range); as markdown link text, where a bracket sits between the word and the number;
# and split across a line break, which :func:`citation_windows` handles. Written as character
# classes and described rather than exemplified, so this module does not match itself. The run is
# possessive so a failing match does not retry every split of a whitespace run, which would be
# quadratic in the run's length.
NUMBERED_CITATION = re.compile(r"\bADRs?\b[\s-]*+\[?\s*\d{4}\b")
# The docstring citation form: the bare id in backticks, since a docstring cannot link.
ID_CITATION = re.compile(r"ADR\s+`([a-z][a-z0-9-]*/[a-z][a-z0-9-]*)`")


def read(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def adr_files() -> list[Path]:
    """Every decision record, repo-relative and sorted.

    The two scaffold files are dropped, but **only at the top level**. Matching those names at
    any depth would let ``docs/adr/<topic>/README.md`` escape every invariant below, since each
    one is parametrized over exactly this list.
    """
    found = sorted(ADR_ROOT.rglob("*.md"))
    return [
        p.relative_to(REPO_ROOT)
        for p in found
        if p.is_file() and not (p.parent == ADR_ROOT and p.name in SCAFFOLD)
    ]


def source_files() -> list[Path]:
    """Every hand-authored file that can carry an ADR citation in prose or a docstring."""
    found = markdown_under(".", "docs", "src", "tools")
    found += python_under("src")
    found += python_under("tests")
    found += python_under("tools")
    return relative(found)


def citation_windows(text: str) -> list[tuple[int, str, int]]:
    """Each line joined to the one after it, as ``(line number, window, head length)``.

    Prose wrapping puts ``ADR`` and its number or id on separate lines, and a regex applied to
    one physical line cannot see that. A wrap inserts exactly one break, so a two-line window is
    enough.

    ``head length`` is where the first line ends inside the window. A caller keeps only matches
    beginning before it, so a citation sitting wholly on one line is reported once -- at its own
    line -- rather than again as the tail of the previous window.
    """
    lines = text.split("\n")
    return [
        (number, " ".join(lines[number - 1 : number + 1]), len(lines[number - 1]))
        for number in range(1, len(lines) + 1)
    ]


def cited(text: str, pattern: re.Pattern[str]) -> list[tuple[int, re.Match[str]]]:
    """Every ``pattern`` match in ``text``, wrap-aware, as ``(line number, match)``."""
    return [
        (number, match)
        for number, window, head in citation_windows(text)
        for match in pattern.finditer(window)
        if match.start() < head
    ]


def frontmatter_block(text: str) -> list[str] | None:
    """The lines between the opening and closing ``---``, or ``None`` if there is no block.

    ``None`` covers both "does not open with a fence" and "never closes it", which are the
    two ways a hand-written block goes wrong.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return lines[1:index]
    return None


def frontmatter(text: str) -> dict[str, str]:
    """The flat ``key: value`` block, comment-stripped and unquoted.

    Total by construction: a malformed block yields ``{}`` rather than raising, so the
    failure surfaces as one readable assertion instead of a collection error that takes the
    whole module down with it.

    Surrounding quotes come off here rather than in each caller, because ``created: "2026-06-12"``
    is a natural thing to write and left quoted it reaches :func:`datetime.date.fromisoformat`
    as an unparseable string. A flow sequence is unaffected -- ``["a/b"]`` opens with a bracket,
    so there is nothing at either end to strip.

    A repeated key is recorded under :data:`DUPLICATE_KEY` rather than silently overwritten,
    since two contradictory lines are exactly the hand-edit this module exists to catch.
    """
    parsed: dict[str, str] = {}
    for line in frontmatter_block(text) or []:
        key, separator, value = line.partition(":")
        if not separator:
            continue
        name = key.strip()
        cleaned = value.partition(" #")[0].strip().strip("\"'")
        parsed[name] = DUPLICATE_KEY if name in parsed else cleaned
    return parsed


def adr_id(doc: Path) -> str:
    """The record's id: its path under ``docs/adr/`` minus the extension."""
    return doc.relative_to(ADR_DIR).with_suffix("").as_posix()


def id_exists(identifier: str) -> bool:
    return (ADR_ROOT / f"{identifier}.md").is_file()


def sections(text: str) -> list[str]:
    """Level-two headings in document order."""
    return [match.group(1) for line in prose_lines(text) if (match := SECTION.match(line))]


TITLE = "(title)"
"""The key for text above the first level-two heading, which no real heading can be named."""


def bodies_by_section(text: str) -> dict[str, list[str]]:
    """Level-two heading -> the prose lines under it, in document order, fences dropped.

    A heading that appears twice collects both bodies under one key. Text above the first heading
    sits under :data:`TITLE`.
    """
    bodies: dict[str, list[str]] = {}
    current = TITLE
    for line in prose_lines(text):
        if match := SECTION.match(line):
            current = match.group(1)
            bodies.setdefault(current, [])
            continue
        bodies.setdefault(current, []).append(line)
    return bodies


def section_body(text: str, heading: str) -> str:
    """The lines under one level-two heading, excluding the heading itself."""
    return "\n".join(bodies_by_section(text).get(heading, []))


def rows_under(text: str, heading: str, pattern: re.Pattern[str]) -> list[re.Match[str]]:
    """Table rows matching ``pattern`` inside one level-two section."""
    return [
        row for line in bodies_by_section(text).get(heading, []) if (row := pattern.match(line))
    ]


def topic_directories() -> set[str]:
    return {child.name for child in ADR_ROOT.iterdir() if child.is_dir()}


ALL_ADRS = adr_files()
CITING_SOURCES = source_files()
FRONTMATTER: dict[Path, dict[str, str]] = {doc: frontmatter(read(doc)) for doc in ALL_ADRS}


def field(doc: Path, key: str) -> str:
    """One frontmatter value, failing readably when it is absent or declared twice.

    Every accessor goes through here rather than subscripting :data:`FRONTMATTER`, so a
    malformed record produces a sentence naming the file and the key instead of a bare
    ``KeyError`` from inside a test body.
    """
    values = FRONTMATTER[doc]
    assert key in values, f"{doc} frontmatter has no {key!r} key"
    assert values[key] != DUPLICATE_KEY, f"{doc} frontmatter declares {key!r} more than once"
    return values[key]


def iso_date(doc: Path, label: str, value: str) -> date:
    """An ISO calendar date, failing readably rather than raising ``ValueError``.

    The row pattern matches on digit *shape*, so ``2026-13-01`` reaches this function looking
    well formed. Calendar validity is checked here, where the file can be named.
    """
    try:
        return date.fromisoformat(value)
    except ValueError:
        pytest.fail(f"{doc} has {label} = {value!r}, which is not an ISO calendar date")


# --- frontmatter is well formed -------------------------------------------


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_every_adr_opens_with_frontmatter(doc: Path):
    """Status and dates are read by globbing, so they have to be machine-readable."""
    assert frontmatter_block(read(doc)) is not None, f"{doc} has no closed frontmatter block"
    keys = set(FRONTMATTER[doc])
    assert keys == set(FRONTMATTER_KEYS), (
        f"{doc} frontmatter keys are {sorted(keys)}, expected {sorted(FRONTMATTER_KEYS)}"
    )


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_status_is_a_known_value(doc: Path):
    status = field(doc, "status")
    assert status in STATUSES, f"{doc} has status {status!r}, expected one of {sorted(STATUSES)}"


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_topic_matches_the_directory(doc: Path):
    """The path is authoritative; the field mirrors it so frontmatter alone is readable."""
    assert field(doc, "topic") == doc.parent.name, (
        f"{doc} declares topic {field(doc, 'topic')!r} but sits in {doc.parent.name!r}"
    )


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_created_is_an_iso_date_not_after_updated(doc: Path):
    """``created`` records when the *record* was first written, never when the file was touched."""
    created = iso_date(doc, "created", field(doc, "created"))
    updated = iso_date(doc, "updated", field(doc, "updated"))
    assert created <= updated, f"{doc} was created {created} but claims it was updated {updated}"


# --- the record has the expected shape ------------------------------------


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_required_sections_are_present_and_ordered(doc: Path):
    """The Standard MADR section set, in order, with nothing extra at level two."""
    found = sections(read(doc))
    assert tuple(found) == REQUIRED_SECTIONS, (
        f"{doc} has sections {found}, expected {list(REQUIRED_SECTIONS)}"
    )


def test_no_filename_carries_a_number():
    """Ordering comes from ``created``. A counter is what concurrent writers collide on."""
    offenders = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in ADR_ROOT.rglob("*.md")
        if NUMBERED_FILENAME.match(p.name)
    )
    assert not offenders, f"these ADR filenames still carry a sequence number: {offenders}"


# --- citations resolve ----------------------------------------------------


@pytest.mark.parametrize("source", CITING_SOURCES, ids=str)
def test_no_prose_cites_a_numbered_adr(source: Path):
    """There are no ADR numbers, so a numbered citation names nothing at all."""
    offenders = [
        f"line {number}: {match.group(0)!r}"
        for number, match in cited(read(source), NUMBERED_CITATION)
    ]
    assert not offenders, f"{source} cites ADRs by retired number: {offenders}"


def unresolved_adr_ids(text: str) -> list[str]:
    """The ADR ids ``text`` cites that name no record, in citation order.

    Args:
        text: Document or docstring text to scan.

    Returns:
        The unresolved ids.
    """
    return [
        match.group(1) for _, match in cited(text, ID_CITATION) if not id_exists(match.group(1))
    ]


@pytest.mark.parametrize("source", CITING_SOURCES, ids=str)
def test_every_cited_adr_id_resolves(source: Path):
    """A docstring citation is prose: nothing but this test notices when it goes stale."""
    broken = unresolved_adr_ids(read(source))
    assert not broken, f"{source} cites ADR ids that do not resolve: {broken}"


def test_an_unresolvable_adr_id_is_reported():
    """The anti-vacuity control: a scan that matched nothing, or resolved everything, passes above.

    The citation is assembled at run time and never spelled whole, because this file is in its own
    corpus -- a literal one would make the module fail its own resolve check.
    """
    invented = "tooling/" + "no-such-record-was-ever-written"
    real = min(adr_id(doc) for doc in ALL_ADRS)
    citation = " and ".join(f"ADR `{one}`" for one in (real, invented))

    assert unresolved_adr_ids(citation) == [invented], (
        "the check must report an id that names no record, and only that one"
    )


# --- the scaffold ---------------------------------------------------------


PINNED = re.compile(r"^- Pinned by: (.+)$", re.MULTILINE)

# Directories git tracks. A path under a gitignored root such as `data/` or `logs/` names a local
# artifact a fresh clone does not have, so its absence is not a dead citation.
TRACKED_ROOTS = frozenset({"docs", "src", "tests", "tools"})
CODE_SPAN = re.compile(r"`([^`\s]+)`")
# A pytest node id, then a line or line range, both trailing the path they qualify.
PATH_SUFFIX = re.compile(r"(?:::\S*)?(?::\d+(?:-\d+)?)?$")
# A placeholder or a glob names a shape rather than a file.
NOT_A_PATH = re.compile(r"[<>*{}]")


def enforcement_paths(text: str) -> list[str]:
    """Every repository path a record's `## Enforcement` section names in a code span.

    Every line of the section, not only `- Pinned by:`: an Owner, an `- Also pinned by:`, a wrapped
    continuation and a prose bullet each name a path a reader will follow.
    """
    found: list[str] = []
    for span in CODE_SPAN.findall(section_body(text, "Enforcement")):
        path = PATH_SUFFIX.sub("", span)
        if "/" in path and path.split("/")[0] in TRACKED_ROOTS and not NOT_A_PATH.search(path):
            found.append(path)
    return found


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_enforcement_names_an_owner_and_what_pins_it(doc: Path):
    """`## Enforcement` is only worth having if a reader can follow it."""
    body = section_body(read(doc), "Enforcement")
    assert body.lstrip().startswith("- Owner:"), (
        f"{doc}'s Enforcement does not open with `- Owner:`"
    )
    assert PINNED.search(body), f"{doc}'s Enforcement names no `- Pinned by:`"


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_every_enforcement_path_resolves(doc: Path):
    """A plausible-looking path that does not exist is worse than naming none.

    Every line of the section is read, not only `- Pinned by:`: an Owner, an `- Also pinned by:`,
    a wrapped continuation and a prose bullet each name a path a reader will follow.
    `nothing -- convention only` is the honest answer for a convention and passes here.
    """
    missing = [path for path in enforcement_paths(read(doc)) if not (REPO_ROOT / path).exists()]
    assert not missing, f"{doc}'s Enforcement names paths that do not exist: {missing}"


ENFORCEMENT_DOCUMENT = """\
# A record

## Consequences
The layout is described in `tests/absent_in_consequences.py`.

## Enforcement
- Owner: `src/absent_owner.py`
- Pinned by: `tests/test_x.py::test_y`
- Also pinned by: `tests/absent_also.py`
- The order is pinned by
  `tests/absent_continuation.py::TestThing::test_case`, and `tools/x.py:47-52` holds the rule.
- Local artifacts such as `data/heart/` and shapes such as `src/<package>/` are not paths.
"""


def test_every_enforcement_line_is_read_for_paths():
    assert enforcement_paths(ENFORCEMENT_DOCUMENT) == [
        "src/absent_owner.py",
        "tests/test_x.py",
        "tests/absent_also.py",
        "tests/absent_continuation.py",
        "tools/x.py",
    ]


# The passive forms "pinned by", "pinned in" and "pinned against", a colon allowed after them and
# the two words split across a line break. The colon is possessive: an optional colon that could be
# given back would let the lookahead below see it and miss the exemption. The one exempt object is
# a file's content digest, or a SHA or hash *of its bytes*: that fixes the identity of an input and
# names no guard. A guard named for a digest or a hash ("the digest test") is still a claim, which
# is why the exemption names the whole phrase rather than the noun.
PIN_CLAIM = re.compile(
    r"\bpinned\s+(?:by|in|against)\b:?+"
    r"(?!\s+(?:its\s+|the\s+|a\s+)?"
    r"(?:content\s+digest\b|(?:sha-?\d*|hash)\s+of\s+(?:its|the)\s+bytes\b))",
    re.IGNORECASE,
)
# The field's bare label in a code span names the field rather than using it: a record describing
# the Enforcement format has to be able to say `- Pinned by:` without making a claim.
FIELD_LABEL = re.compile(r"`-?\s*pinned\s+by:?`", re.IGNORECASE)


def pin_claims_outside_enforcement(text: str) -> list[str]:
    """Every pin claim in a record's prose outside `## Enforcement`, as `section: excerpt`.

    Read per section with line breaks folded to spaces, so a claim wrapped across two lines is
    one claim. Fenced code is skipped, as every section reader here skips it, and so is the field's
    bare label in a code span. Text above the first level-two heading is reported under
    :data:`TITLE`.
    """
    claims: list[str] = []
    for heading, lines in bodies_by_section(text).items():
        if heading == "Enforcement":
            continue
        body = FIELD_LABEL.sub("`field`", " ".join(lines))
        claims.extend(
            f"{heading}: {body[max(0, hit.start() - 40) : hit.end() + 40]}"
            for hit in PIN_CLAIM.finditer(body)
        )
    return claims


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_no_pin_claim_sits_outside_enforcement(doc: Path):
    """A "pinned by" claim lives in `## Enforcement`, where its path is at least checked to exist.

    ADR `tooling/prose-describes-guards-never-licenses-changes` owns the rule.
    """
    claims = pin_claims_outside_enforcement(read(doc))
    assert not claims, (
        f"{doc} makes a pin claim outside `## Enforcement`; move it there or state the argument "
        f"instead: {claims}"
    )


PIN_DOCUMENT = """\
# A record

## Consequences
{consequences}

## Enforcement
- Owner: `src/x.py`
- Pinned by: `tests/test_x.py`
"""


@pytest.mark.parametrize(
    "consequences",
    [
        "The layout is pinned by `tests/test_x.py`.",
        "The layout is pinned\nby a regeneration test.",
        "This record ships `Pinned by: nothing -- convention only`.",
        "The order is Pinned By the golden report.",
        "Both are pinned in `tests/test_belief.py`.",
        "The output is pinned against a committed golden capture.",
        "The order is pinned by the digest test `tests/test_cross_platform_digest.py`.",
        "The layout is pinned by the hash-equality test in `tests/test_x.py`.",
        "The layout is pinned by: `tests/test_x.py`.",
    ],
    ids=[
        "code-span",
        "wrapped",
        "quoted-field",
        "capitalized",
        "pinned-in",
        "pinned-against",
        "digest-named-guard",
        "hash-named-guard",
        "colon",
    ],
)
def test_a_pin_claim_outside_enforcement_is_refused(consequences: str):
    assert pin_claims_outside_enforcement(PIN_DOCUMENT.format(consequences=consequences))


@pytest.mark.parametrize(
    "consequences",
    [
        "The paper was fetched, pinned by content digest, and read.",
        "The paper is pinned by: its content digest.",
        "The file is pinned by the sha256 of its bytes.",
        "The file is pinned by the hash of its bytes.",
        "```\npinned by `tests/test_x.py`\n```",
        "Require `- Pinned by:` to name a test function.",
        # Two documented limits rather than wanted behaviour: a word between the verb and its
        # preposition, and the active voice, both pass. ADR
        # `tooling/prose-describes-guards-never-licenses-changes` states why neither is widened to.
        "The decision is pinned byte-for-byte by a golden report.",
        "`tests/test_x.py` pins the layout.",
    ],
    ids=[
        "content-digest",
        "colon-content-digest",
        "sha-of-bytes",
        "hash-of-bytes",
        "fenced",
        "field-label",
        "limit-word-split",
        "limit-active-voice",
    ],
)
def test_what_is_not_a_pin_claim_passes(consequences: str):
    assert pin_claims_outside_enforcement(PIN_DOCUMENT.format(consequences=consequences)) == []


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_ruled_out_is_a_dated_table(doc: Path):
    """Named ideas with a reason and a date, never a prose list and never numbered."""
    rows = [
        line
        for line in section_body(read(doc), "Ruled Out").splitlines()
        if line.startswith("|") and not set(line) <= set("|- ")
    ]
    assert len(rows) >= 2, f"{doc}'s Ruled Out is not a table with at least one row"
    for row in rows[1:]:
        assert re.search(r"\|\s*\d{4}-\d{2}-\d{2}\s*\|?\s*$", row), (
            f"{doc} has a Ruled Out row with no ISO date: {row[:80]}"
        )


UNCHECKABLE = (
    "if requirements change",
    "if this turns out to be wrong",
    "if circumstances change",
    "if it stops working",
    "tbd",
    "to be determined",
)
"""Phrases that make a reversal condition unfalsifiable.

Not exhaustive and not meant to be. It refuses the specific wordings the reference format calls
out as useless, so the retired-deferral pattern cannot come back verbatim; a reader determined to
phrase a non-condition around it can, and review is what catches that.
"""


@pytest.mark.parametrize("doc", ALL_ADRS, ids=str)
def test_reversal_conditions_say_something_checkable(doc: Path):
    """A section a reader cannot evaluate is the same as no section.

    The whole point of the field is that a future reader can tell whether the condition has
    happened. "If requirements change" cannot be told either way, and an empty section asserts
    that the decision is irreversible without saying so.
    """
    body = section_body(read(doc), "Reversal Conditions").strip()
    assert body, f"{doc} has an empty Reversal Conditions section"
    lowered = body.lower()
    vague = [phrase for phrase in UNCHECKABLE if phrase in lowered]
    assert not vague, f"{doc}'s reversal condition is not checkable: {vague}"


def test_the_template_frontmatter_matches_the_enforced_key_set():
    """`template.md` is excluded from `ALL_ADRS`, so nothing else ever parses it.

    It is the file every new record is copied from, which makes it the one scaffold that can hand
    a whole generation of records a schema the guards reject.
    """
    keys = set(frontmatter(read(ADR_DIR / "template.md")))
    assert keys == set(FRONTMATTER_KEYS), (
        f"docs/adr/template.md declares {sorted(keys)}, expected {sorted(FRONTMATTER_KEYS)}"
    )


def test_the_readme_example_frontmatter_matches_the_enforced_key_set():
    """The README shows a filled-in frontmatter block; a reader copies what it shows."""
    body = read(ADR_DIR / "README.md")
    fence = re.search(r"```yaml\n(.*?)```", body, re.DOTALL)
    assert fence is not None, "docs/adr/README.md no longer shows a yaml frontmatter example"
    shown = {
        line.partition(":")[0].strip()
        for line in fence.group(1).splitlines()
        if line.strip() and not line.startswith("---")
    }
    assert shown == set(FRONTMATTER_KEYS), (
        f"docs/adr/README.md's example declares {sorted(shown)}, "
        f"expected {sorted(FRONTMATTER_KEYS)}"
    )


def test_the_readme_names_every_status_and_no_more():
    """The example's status comment enumerates the vocabulary.

    A stale one invites a value the guard rejects, from the file that documents the guard.
    """
    body = read(ADR_DIR / "README.md")
    line = next(row for row in body.splitlines() if row.startswith("status:") and "#" in row)
    shown = {value.strip() for value in line.partition("#")[2].split("|")}
    assert shown == set(STATUSES), (
        f"docs/adr/README.md's status comment names {sorted(shown)}, expected {sorted(STATUSES)}"
    )


def test_readme_lists_every_topic_and_no_more():
    """A new topic is the one case that touches shared state, so it has to be recorded."""
    readme = Path("docs/adr/README.md")
    assert (REPO_ROOT / readme).is_file(), f"{readme} is missing"
    listed = {bullet.group(1) for bullet in rows_under(read(readme), "Topics", TOPIC_BULLET)}
    assert listed == topic_directories(), (
        f"{readme} lists topics {sorted(listed)} but the directories are "
        f"{sorted(topic_directories())}"
    )


# One file per root-and-suffix pair `source_files` walks, named rather than counted: a count
# cannot show which root stopped being walked, so the floor below holds that discovery happens and
# these names hold which discovery.
CORPUS_WITNESSES = (
    Path("AGENTS.md"),
    Path("docs/DEVELOPMENT.md"),
    Path("src/bko/bkb/README.md"),
    Path("tools/README.md"),
    Path("src/bko/__init__.py"),
    Path("tests/test_adr_frontmatter.py"),
    Path("tools/check_rebased.py"),
)


def test_the_adr_set_is_not_silently_empty():
    """Every check above is parametrized over a glob, so an empty log would pass them all.

    The floors are deliberately below the current counts: they hold that discovery happens, and
    :data:`CORPUS_WITNESSES` holds which roots it reaches.
    """
    assert len(ALL_ADRS) >= 5, f"only found {len(ALL_ADRS)} ADRs; discovery is likely broken"
    assert (ADR_ROOT / "README.md").is_file()
    assert (ADR_ROOT / "template.md").is_file()
    assert len(CITING_SOURCES) >= 20, f"only found {len(CITING_SOURCES)} citing sources"


@pytest.mark.parametrize("witness", CORPUS_WITNESSES, ids=str)
def test_every_walked_root_still_reaches_the_corpus(witness: Path):
    """A root that stops matching takes its whole contribution silently, floor or no floor."""
    assert witness in set(CITING_SOURCES), (
        f"{witness} is absent from the corpus, so the root it represents is not being walked"
    )


# The word the pattern keys on, bound rather than written into the cases below. This module is in
# its own corpus -- `test_no_prose_cites_a_numbered_adr` scans it -- so a case spelled literally
# would be a citation of a numbered ADR and the guard would refuse this file. The pattern's own
# comment states the same rule for the same reason.
WORD = "ADR"


@pytest.mark.parametrize(
    ("text", "matches"),
    [
        (f"{WORD} 0001", True),
        (f"{WORD}s 0002", True),
        (f"{WORD}-0003", True),
        (f"{WORD} [0004", True),
        (f"{WORD}  [ 0005", True),
        (f"{WORD}\n0006", True),
        (f"{WORD} - [ 0007", True),
        (f"see {WORD} 0008 here", True),
        (f"{WORD} 12", False),
        (f"{WORD} 00012", False),
        (f"{WORD}x 0009", False),
    ],
    ids=str,
)
def test_the_numbered_citation_pattern_keeps_every_verdict(text: str, matches: bool):
    """Each shape the pattern's comment reasons about, and the two near-misses.

    The plural, the hyphen, the optional bracket, a line break from prose wrapping, and too few or
    too many digits, which keep it from matching an ordinary number.
    """
    assert bool(NUMBERED_CITATION.search(text)) is matches


def test_the_numbered_citation_pattern_does_not_backtrack_on_a_long_run():
    """A guard over operator-supplied text must not have a length at which it stops returning.

    A non-possessive run either side of the optional bracket retries every split of a whitespace
    run between them, which is quadratic. The threshold is loose by two orders of magnitude, so it
    fails on a reintroduced quadratic and not on a busy machine.
    """
    subject = WORD + " " * 10_000 + "x"
    started = time.perf_counter()
    assert NUMBERED_CITATION.search(subject) is None
    assert time.perf_counter() - started < 0.1, "the pattern is backtracking again"
