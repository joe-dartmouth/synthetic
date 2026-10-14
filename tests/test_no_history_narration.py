"""The working documents state current facts, not the history of their own corrections.

`AGENTS.md`'s *Documentation* section sets the rule: a document says what is true now, with its
reason and provenance, and git holds the history. These files are the ones a session reads to work
in the repository, so a dated correction narrative there costs every reader and goes stale. The ADRs
are out of scope, because superseded decisions are part of their content.

A dated measurement (`measured 2026-08-29: ...`) is provenance and passes, and so is an imperative
rule (`do not read X until Y`). Text inside a code span or a fence is an example rather than a claim
and is removed before matching. Matching runs over each document as one string, so a phrase split
by a line wrap is still caught and a code span that wraps is still removed whole.
"""

from __future__ import annotations

import re
from operator import itemgetter
from pathlib import Path

import pytest

from tests._corpus import REPO_ROOT, fences_balanced, numbered_prose_lines


GUARDED = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("GEMINI.md"),
    Path("README.md"),
    Path("docs/DEVELOPMENT.md"),
    Path("tools/README.md"),
    *sorted(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "src").rglob("README.md")),
)
"""The working documents whose prose must carry no correction narrative, stage READMEs included."""

NARRATION = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\buntil \d{4}-\d{2}-\d{2}",
        r"\bas of \d{4}-\d{2}-\d{2}",
        r"\(set \d{4}-\d{2}-\d{2}",
        r"\b(?:said|claimed)\b.{0,40}\buntil\b",
        r"\bread\b.{0,40}\buntil \d{4}-",
        r"\bwas wrong\b(?! by)",
        r"\bthe first version\b",
        r"\bused to (?:say|read|be)\b",
        r"\bwent stale\b",
        r"\bthis (?:row|line|paragraph|sentence|section|file) (?:said|read|claimed)\b",
        r"\bis what this \w+ (?:said|read)\b",
        r"\b(?:module|package|file|function|directory) (?:which|that) is now\b",
    )
)
"""Shapes of a correction narrative, matched case-insensitively.

`read` is ambiguous between past and present tense, so it counts only beside a dated "until"; `was
wrong by` is a measured error rather than a correction; a module "that is now" somewhere else is a
rename told as history.
"""

CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)", re.DOTALL)
"""A CommonMark code span: a backtick run closed by a run of the same length, across line breaks."""


def _blank_keeping_lines(match: re.Match[str]) -> str:
    return "\n" * match.group(0).count("\n")


def narration_lines(text: str) -> list[tuple[int, str]]:
    """Return the numbered lines where a narration shape starts.

    Args:
        text: A markdown document.

    Returns:
        `(line number, line)` pairs, numbered in the original text, for each match of a `NARRATION`
        pattern in the document's prose once fences and code spans are removed.
    """
    lines = text.split("\n")
    prose_numbers = {number for number, _ in numbered_prose_lines(text)}
    prose = "\n".join(
        line if number in prose_numbers else "" for number, line in enumerate(lines, 1)
    )
    prose = CODE_SPAN.sub(_blank_keeping_lines, prose)
    flat = prose.replace("\n", " ")
    hits = {
        prose.count("\n", 0, match.start()) + 1
        for pattern in NARRATION
        for match in pattern.finditer(flat)
    }
    return [(number, lines[number - 1]) for number in sorted(hits)]


@pytest.mark.parametrize("doc", GUARDED, ids=str)
def test_the_document_carries_no_correction_narrative(doc: Path) -> None:
    path = REPO_ROOT / doc
    assert path.is_file(), f"{doc} is guarded but missing, so the guard would check nothing"
    text = path.read_text(encoding="utf-8")
    assert fences_balanced(text), f"{doc} has an unclosed fence, hiding the rest from the guard"
    assert numbered_prose_lines(text), f"{doc} has no prose lines, so the guard read nothing"
    hits = narration_lines(text)
    assert not hits, (
        f"{doc} narrates its own history; state the current fact and let git hold the rest: "
        + "; ".join(f"line {number}: {line.strip()}" for number, line in hits)
    )


@pytest.mark.parametrize(
    "text",
    [
        "It resolved fresh until 2026-09-07.",
        "Until 2026-09-07 the flag resolved fresh.",
        "It became a package as of 2026-09-05.",
        "As of 2026-09-05 it is a package.",
        "Entries are keyed by surface (set 2026-08-28).",
        "This row said three until the next change landed.",
        "It read seven until 2026-09-08.",
        "That claim was wrong because the probe never ran.",
        "The first version compared names, not uids.",
        "The cap used to be four.",
        "The figure went stale the next day.",
        "This line said four.",
        '"Four" is what this line said.',
        "The module that is now `bko.gaps.counts` reported half.",
        "The flag resolved fresh until\n2026-09-07.",
        "Start `code\nend` and it said three until 2026-01-01 `z`.",
    ],
    ids=itemgetter(slice(40)),
)
def test_the_shapes_catch_a_narrative(text: str) -> None:
    assert narration_lines(text), f"the guard let a narrative through: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Measured 2026-08-29: two overlapping runs each under-reported.",
        "Docs avoid `this said X until <date>` and `was wrong because`.",
        "Avoid ``said X until `y` 2026-01-01`` in prose.",
        "```\nit said four until 2026-09-07\n```",
        "~~~\nit said four until 2026-09-07\n~~~",
        "````\n```\nit said four until 2026-09-07\n```\n````",
        "The sweep waits until the batch closes.",
        "Do not read the summary until `make certify` passes.",
        "Read the stream until EOF.",
        "The estimate was wrong by 2x, measured on one host.",
    ],
    ids=itemgetter(slice(40)),
)
def test_the_shapes_pass_provenance_rules_and_examples(text: str) -> None:
    assert not narration_lines(text), f"the guard refused a current fact: {text!r}"


def test_an_unclosed_fence_is_reported() -> None:
    assert not fences_balanced("prose\n```\nnever closed\n")
    assert fences_balanced("prose\n```python\ncode\n```\nmore prose\n")
