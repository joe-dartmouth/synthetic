"""Generate the `diabetes-codes` frame's ICD-9 code rows from a MONDO release.

`python -m bko.ontology codes` runs `generate` over every diagnosis token of the `diabetes-codes`
cohort and writes `render`'s module text, committed as `bko.ontology.icd9_codes`. A code maps to
the MONDO concept that carries its exact `ICD9:` xref, its whole part zero-padded to three digits,
under the qualifier `source="MONDO:equivalentTo"` (`EXACT`): the curated `ICD9_MAPPINGS` row's
concept where that row's xref is the code's, else the one live exact holder every other live exact
holder descends from by `is_a`, else nothing. Obsolete holders are ignored, an xref under any
other qualifier maps nothing, and V and E codes are not looked up. ADR
`research/diabetes-diagnosis-codes` records the rule and the unmapped counts.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from bko.datasets.diagnoses import UNKNOWN, canonical, code_number
from bko.ontology.icd9 import EXACT, ICD9_MAPPINGS, Icd9Code, Icd9Mapping
from bko.ontology.terms import Ontology, ancestors


if TYPE_CHECKING:
    from bko.ontology.icd10 import Icd10Code


REASONS: Final = (
    "no live xref",
    "obsolete holder only",
    "relatedTo only",
    "unreviewed source only",
    "holders not nested",
    "V code",
    "E code",
)
"""Why a code stays unmapped, in the order `Generated.unmapped` lists them. A code whose live
xrefs carry no `EXACT` qualifier is `relatedTo only` when one carries `RELATED`, else
`unreviewed source only`."""

RELATED: Final = "MONDO:relatedTo"
"""The `source` qualifier of a curator-reviewed link weaker than equivalence; matched ignoring
case, since the release also writes `MONDO:RelatedTo`."""

_XREF: Final = "ICD9:"

_HEADER: Final = '''"""The `diabetes-codes` frame's ICD-9 code rows, generated from MONDO.

`python -m bko.ontology codes` writes this module; regenerate it rather than editing it. Each row
maps one exact diagnosis code of the `diabetes-codes` cohort to the MONDO concept that carries its
`ICD9:` xref under `source="MONDO:equivalentTo"` (ADR `research/diabetes-diagnosis-codes`).
`_ROWS` holds one (code, concept, label) tuple per line, and `_ALSO` the other live holders of
such an xref for a code that has more than one.
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
"""Each mapped code, in encoding order, with its concept and every live exact holder of its xref."""
'''


@dataclass(frozen=True, slots=True, eq=False)
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
        ValueError: If a token is not a diagnosis token, a curated row's concept is not among the
            live exact holders of a code that has some, or a chosen concept has no name.
    """
    codes = sorted(
        {canonical(token) for token in tokens if code_number(token) != UNKNOWN}, key=code_number
    )
    exact, weaker, dead = _holders(mondo, _XREF)
    chosen = {row.xref: row.concept for row in curated}
    rows: list[Icd9Code] = []
    unmapped = dict.fromkeys(REASONS, 0)
    for code in codes:
        holders = exact.get(code, ())
        if code[0] in "VE":
            unmapped[f"{code[0]} code"] += 1
            continue
        if not holders:
            unmapped[_unmapped(code, weaker, dead)] += 1
            continue
        concept = chosen.get(f"{_XREF}{code}")
        if concept is not None and concept not in holders:
            msg = (
                f"the curated row for {_XREF}{code} maps {concept}, which is not among its live "
                f"{EXACT} holders {', '.join(holders)}"
            )
            raise ValueError(msg)
        concept = concept or _spanning(mondo, holders)
        if concept is None:
            unmapped["holders not nested"] += 1
            continue
        rows.append(Icd9Code(code, concept, _name(mondo, concept), frozenset(holders)))
    return Generated(tuple(rows), MappingProxyType(unmapped))


def render(rows: Sequence[Icd9Code]) -> str:
    """The text of `bko.ontology.icd9_codes` holding `rows`, as `ruff format` leaves it.

    `_ROWS` sits between `# fmt: off` and `# fmt: on`, so a long label keeps its tuple on one line.

    Raises:
        ValueError: If a row's code, concept or label holds a control character, which would
            break its line, naming the row's code.
    """
    return render_rows(rows, header=_HEADER, footer=_FOOTER, system="ICD-9")


def render_rows(
    rows: Sequence[Icd9Code | Icd10Code], *, header: str, footer: str, system: str
) -> str:
    """A generated code module: `header`, `_ROWS` and `_ALSO` holding `rows`, then `footer`.

    Raises:
        ValueError: If a row's code, concept or label holds a control character, naming the row's
            code under `system`.
    """
    for row in rows:
        if any(_controlled(text) for text in (row.code, row.concept, row.label)):
            msg = f"{system} {row.code!r}: a control character in {row!r} would break its line"
            raise ValueError(msg)
    lines = [header, "# fmt: off\n", "_ROWS: Final[tuple[tuple[str, str, str], ...]] = (\n"]
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
    lines.extend(("}\n", footer))
    return "".join(lines)


def _holders(
    mondo: Ontology, prefix: str = _XREF
) -> tuple[dict[str, tuple[str, ...]], dict[str, bool], frozenset[str]]:
    """Each code's live `EXACT` holders, and whether its other live xrefs include `RELATED`.

    Args:
        mondo: The MONDO release.
        prefix: The cross-reference prefix whose codes are read, `ICD9:` or `ICD10CM:`.

    Returns:
        Each code's live `EXACT` holders in term order; each code some other live xref carries,
        and whether one of those carries `RELATED`; and the codes only obsolete terms hold.
    """
    exact: dict[str, list[str]] = {}
    weaker: dict[str, bool] = {}
    dead: set[str] = set()
    for curie, term in mondo.terms.items():
        for xref in term.xrefs:
            if not xref.startswith(prefix):
                continue
            code = xref.removeprefix(prefix)
            sources = term.xref_sources(xref)
            if term.obsolete:
                dead.add(code)
            elif EXACT in sources:
                if curie not in exact.setdefault(code, []):
                    exact[code].append(curie)
            else:
                related = any(source.casefold() == RELATED.casefold() for source in sources)
                weaker[code] = weaker.get(code, False) or related
    live = set(exact) | set(weaker)
    return {code: tuple(c) for code, c in exact.items()}, weaker, frozenset(dead - live)


def _unmapped(code: str, weaker: Mapping[str, bool], dead: frozenset[str]) -> str:
    """Why `code`, which no live term carries under `EXACT`, stays unmapped."""
    if code in weaker:
        return "relatedTo only" if weaker[code] else "unreviewed source only"
    return "obsolete holder only" if code in dead else "no live xref"


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


def _controlled(text: str) -> bool:
    """Whether `text` holds a control character (Unicode category Cc), such as a newline."""
    return any(unicodedata.category(char) == "Cc" for char in text)


def _quoted(text: str) -> str:
    """`text` as a double-quoted Python string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
