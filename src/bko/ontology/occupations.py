"""Generate ACS's occupation rows from an Occupation Ontology release and the file's own SOC codes.

`python -m bko.ontology occupations --part low` and `--part high` run `generate` over each kept
`OCCP` code's `SOCP` code (`bko.datasets.acs.occupation_pairs`) and write `render`'s module text,
committed as `bko.ontology.occupation_codes_low` (SOC major groups below 31) and
`bko.ontology.occupation_codes_high` (31 and up), since one module would pass the size cap. A
`SOCP` code maps by its SOC level to the one live class whose `OCCO:00000008` literal is its SOC
id (`soc_ids`); a code no live class holds, two hold, or whose one holder carries several distinct
SOC ids stays unmapped and is counted by reason.
ADR `research/acs-frame` records the rule and the counts.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from bko.ontology.schema import OccupationCode
from bko.ontology.terms import SOC_ID, Ontology


REASONS: Final = ("no class", "several classes", "several SOC ids")
"""Why a code stays unmapped: no live class carries any of its SOC ids, two carry the first that
is carried, or the one class carrying it also carries another SOC id, which the check refuses."""

type Part = Literal["low", "high"]

PARTS: Final[tuple[Part, ...]] = ("low", "high")
"""The two generated modules: SOC major groups below `SPLIT_MAJOR`, then from it up."""

SPLIT_MAJOR: Final = 31
"""The first SOC major group of the `high` part."""

_SOCP: Final = re.compile(r"(?P<digits>[0-9]{2,6})(?P<open>[XY]*)")


def soc_ids(socp: str) -> tuple[str, ...]:
    """The SOC ids a `SOCP` code may map to, in the order they are tried.

    A detailed code, ending in a digit other than 0, tries its O*NET id `XX-XXXX.00`, then
    `XX-XXXX`; a code ending `000` is a minor group, `XX-X000`, and one ending `0` a broad
    occupation, `XX-XXX0`. An aggregate with one trailing `X` or `Y` maps to its broad class,
    `XX-XXX0`, and one with two or three to its minor class, `XX-X000`.

    Raises:
        ValueError: If `socp` is not six characters of digits then trailing `X` or `Y`, or leaves
            four or more digits open.
    """
    match = _SOCP.fullmatch(socp)
    if len(socp) != 6 or match is None or len(match["open"]) > 3:
        msg = f"SOCP {socp!r} is not a SOC code or aggregate the rule reads"
        raise ValueError(msg)
    digits, open_ = match["digits"], len(match["open"])
    major = digits[:2]
    if open_ == 1:
        return (f"{major}-{digits[2:5]}0",)
    if open_ > 1:
        return (f"{major}-{digits[2]}000",)
    rest = digits[2:]
    if rest.endswith("0"):
        return (f"{major}-{rest}",)
    return (f"{major}-{rest}.00", f"{major}-{rest}")


def soc_classes(occo: Ontology) -> dict[str, frozenset[str]]:
    """Each live class carrying an `OCCO:00000008` literal and its distinct SOC ids."""
    return {
        curie: frozenset(socs)
        for curie, term in occo.terms.items()
        if not term.obsolete and (socs := term.annotation(SOC_ID))
    }


def generate(pairs: Mapping[str, str], occo: Ontology) -> tuple[list[OccupationCode], Counter[str]]:
    """Map each `OCCP` code to one OccO class through its `SOCP` code, or count why it is not.

    Args:
        pairs: Each `OCCP` code and the `SOCP` code its rows carry.
        occo: The Occupation Ontology release.

    Returns:
        One row per mapped code, in code order, and each of `REASONS` with how many codes it
        leaves unmapped.

    Raises:
        ValueError: As `soc_ids`, or if a chosen class has no name.
    """
    classes = soc_classes(occo)
    holders: dict[str, list[str]] = {}
    for curie, socs in classes.items():
        for soc in socs:
            holders.setdefault(soc, []).append(curie)
    rows: list[OccupationCode] = []
    unmapped: Counter[str] = Counter(dict.fromkeys(REASONS, 0))
    for code, socp in sorted(pairs.items()):
        found = next(((soc, holders[soc]) for soc in soc_ids(socp) if soc in holders), None)
        if found is None:
            unmapped["no class"] += 1
        elif len(found[1]) > 1:
            unmapped["several classes"] += 1
        elif len(classes[found[1][0]]) > 1:
            unmapped["several SOC ids"] += 1
        else:
            soc, [concept] = found
            rows.append(OccupationCode(code, soc, concept, _name(occo, concept)))
    return rows, unmapped


def part(row: OccupationCode) -> Part:
    """Which generated module holds `row`: `low` below SOC major group `SPLIT_MAJOR`."""
    return "low" if int(row.soc[:2]) < SPLIT_MAJOR else "high"


_HEADER: Final = '''"""ACS's occupation rows for SOC major groups {span}, generated from OccO.

`python -m bko.ontology occupations --part {part}` writes this module; regenerate it rather than
editing it. Each row maps one `OCCP` code of the ACS cohort to the Occupation Ontology class whose
`OCCO:00000008` literal is the SOC id its `SOCP` code reaches (ADR `research/acs-frame`). `_ROWS`
holds one (code, SOC id, concept, label) tuple per line.
"""

from __future__ import annotations

from itertools import starmap
from typing import Final

from bko.ontology.schema import OccupationCode


'''

_FOOTER: Final = '''
OCCUPATION_CODES: Final[tuple[OccupationCode, ...]] = tuple(starmap(OccupationCode, _ROWS))
"""Each mapped `OCCP` code of this part, in code order, with its SOC id, class and label."""
'''


def render(rows: Sequence[OccupationCode], which: Part) -> str:
    """The text of `which` part's generated module holding `rows`, as `ruff format` leaves it.

    `_ROWS` sits between `# fmt: off` and `# fmt: on`, so a long label keeps its tuple on one line.

    Raises:
        ValueError: If a row belongs to the other part, or its fields hold a control character or
            a line or paragraph separator, which would break its line, naming the row's code.
    """
    span = f"below {SPLIT_MAJOR}" if which == "low" else f"from {SPLIT_MAJOR} up"
    lines = [
        _HEADER.format(span=span, part=which),
        "# fmt: off\n",
        "_ROWS: Final[tuple[tuple[str, str, str, str], ...]] = (\n",
    ]
    for row in rows:
        if part(row) != which:
            msg = f"OCCP {row.code!r}: SOC id {row.soc} belongs to the {part(row)} part"
            raise ValueError(msg)
        fields = (row.code, row.soc, row.concept, row.label)
        if any(unicodedata.category(char) in _BREAKING for text in fields for char in text):
            msg = f"OCCP {row.code!r}: a line-breaking character in {row!r} would break its line"
            raise ValueError(msg)
        lines.append(f"    ({', '.join(_quoted(text) for text in fields)}),\n")
    lines.extend((")\n# fmt: on\n", _FOOTER))
    return "".join(lines)


_BREAKING: Final = frozenset({"Cc", "Zl", "Zp"})
"""The Unicode categories holding every character `str.splitlines` splits at."""


def _name(occo: Ontology, concept: str) -> str:
    name = occo.terms[concept].name
    if name is None:
        msg = f"{concept} has no name in {occo.name} {occo.version}"
        raise ValueError(msg)
    return name


def _quoted(text: str) -> str:
    """`text` as a double-quoted Python string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
