"""Generate the `diabetes-codes` frame's ICD-9 code rows from a MONDO release.

`python -m bko.ontology codes` runs `generate` over every diagnosis token of the pinned Diabetes
130 archive and writes `render`'s module text, committed as `bko.ontology.icd9_codes`. A code maps
to the MONDO concept that carries its exact `ICD9:` xref, its whole part zero-padded to three
digits: the curated `ICD9_MAPPINGS` row's concept where that row's xref is the code's, else the one
live holder every other live holder descends from by `is_a`, else nothing. Obsolete holders are
ignored, and V and E codes are not looked up. ADR `research/diabetes-diagnosis-codes` records the
rule and the unmapped counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from bko.datasets.diagnoses import UNKNOWN, canonical, code_number
from bko.ontology.icd9 import ICD9_MAPPINGS, Icd9Code, Icd9Mapping
from bko.ontology.terms import Ontology, ancestors


REASONS: Final = (
    "no live xref",
    "obsolete holder only",
    "holders not nested",
    "V code",
    "E code",
)
"""Why a code stays unmapped, in the order `Generated.unmapped` lists them."""

_XREF: Final = "ICD9:"

_HEADER: Final = '''"""The `diabetes-codes` frame's ICD-9 code rows, generated from MONDO.

`python -m bko.ontology codes` writes this module; regenerate it rather than editing it. Each row
maps one exact diagnosis code of the Diabetes 130 archive to the MONDO concept that carries its
`ICD9:` xref (ADR `research/diabetes-diagnosis-codes`). `_ROWS` holds one (code, concept, label)
tuple per line, and `_ALSO` the other live holders of a code that has more than one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from bko.ontology.icd9 import Icd9Code


'''

_FOOTER: Final = '''
ICD9_CODES: Final[tuple[Icd9Code, ...]] = tuple(
    Icd9Code(code, concept, label, frozenset({concept, *_ALSO.get(code, ())}))
    for code, concept, label in _ROWS
)
"""Each mapped code, in encoding order, with its concept and every live holder of its xref."""
'''


@dataclass(frozen=True, slots=True)
class Generated:
    """What `generate` found.

    Attributes:
        rows: One row per mapped code, in `bko.datasets.diagnoses.code_number` order.
        unmapped: Each of `REASONS` and how many distinct codes it leaves unmapped.
    """

    rows: tuple[Icd9Code, ...]
    unmapped: Mapping[str, int]


def generate(
    mondo: Ontology, tokens: Iterable[str], curated: Sequence[Icd9Mapping] = ICD9_MAPPINGS
) -> Generated:
    """Map each distinct code among `tokens` to one MONDO concept, or count why it stays unmapped.

    Args:
        mondo: The MONDO release.
        tokens: Diagnosis tokens as the archive writes them; `?` is skipped, and a code written
            two ways (`8`, `008`) is one code.
        curated: The curated rows whose concept wins for the code their xref names.

    Returns:
        The rows and the unmapped tallies.

    Raises:
        ValueError: If a token is not a diagnosis token, or a chosen concept has no name.
    """
    codes = sorted(
        {canonical(token) for token in tokens if code_number(token) != UNKNOWN}, key=code_number
    )
    live, dead = _holders(mondo)
    chosen = {row.xref: row.concept for row in curated}
    rows: list[Icd9Code] = []
    unmapped = dict.fromkeys(REASONS, 0)
    for code in codes:
        holders = live.get(code, ())
        if code[0] in "VE":
            unmapped[f"{code[0]} code"] += 1
            continue
        if not holders:
            unmapped["obsolete holder only" if code in dead else "no live xref"] += 1
            continue
        concept = chosen.get(f"{_XREF}{code}") or _spanning(mondo, holders)
        if concept is None:
            unmapped["holders not nested"] += 1
            continue
        rows.append(Icd9Code(code, concept, _name(mondo, concept), frozenset(holders)))
    return Generated(tuple(rows), MappingProxyType(unmapped))


def render(rows: Sequence[Icd9Code]) -> str:
    """The text of `bko.ontology.icd9_codes` holding `rows`, as `ruff format` leaves it.

    `_ROWS` sits between `# fmt: off` and `# fmt: on`, so a long label keeps its tuple on one line.
    """
    lines = [_HEADER, "# fmt: off\n", "_ROWS: Final[tuple[tuple[str, str, str], ...]] = (\n"]
    lines.extend(
        f"    ({_quoted(row.code)}, {_quoted(row.concept)}, {_quoted(row.label)}),\n"
        for row in rows
    )
    lines.append(")\n# fmt: on\n\n")
    also = [row for row in rows if len(row.holders) > 1]
    lines.append(f"_ALSO: Final[Mapping[str, tuple[str, ...]]] = {{{'\n' if also else ''}")
    for row in also:
        others = [_quoted(curie) for curie in sorted(row.holders - {row.concept})]
        tail = "," if len(others) == 1 else ""
        lines.append(f"    {_quoted(row.code)}: ({', '.join(others)}{tail}),\n")
    lines.extend(("}\n", _FOOTER))
    return "".join(lines)


def _holders(mondo: Ontology) -> tuple[dict[str, tuple[str, ...]], frozenset[str]]:
    """Each ICD-9 code to its live holders in term order, and the codes only obsolete terms hold."""
    live: dict[str, list[str]] = {}
    dead: set[str] = set()
    for curie, term in mondo.terms.items():
        for xref in term.xrefs:
            if xref.startswith(_XREF):
                code = xref.removeprefix(_XREF)
                if term.obsolete:
                    dead.add(code)
                elif curie not in live.setdefault(code, []):
                    live[code].append(curie)
    return {code: tuple(curies) for code, curies in live.items()}, frozenset(dead - set(live))


def _spanning(mondo: Ontology, holders: Sequence[str]) -> str | None:
    """The one holder every other holder descends from by `is_a`; None if there is none."""
    spanning = [
        curie
        for curie in holders
        if all(
            other == curie or curie in ancestors(mondo, other, prefix="MONDO") for other in holders
        )
    ]
    return spanning[0] if len(spanning) == 1 else None


def _name(mondo: Ontology, concept: str) -> str:
    term = mondo.terms.get(concept)
    if term is None or term.name is None:
        msg = f"{concept} has no name in {mondo.name} {mondo.version}"
        raise ValueError(msg)
    return term.name


def _quoted(text: str) -> str:
    """`text` as a double-quoted Python string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
