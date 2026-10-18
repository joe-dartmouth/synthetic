"""The prompt: one system text every call shares, and a user turn built up the guidance ladder.

The system text names every column with its meaning, its codes or its unit and plausible range,
and asks for new patients. The user turn states the cell and the record count, then adds one
component per level reached. No prompt holds a record. ADR `research/record-generation-design`
records what each level sends.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from bko.datasets.cells import LABEL, LABEL_MEANINGS, SEXES
from bko.datasets.uci import CODEBOOK, DESCRIPTIONS, FRACTIONAL, FRAME_COLUMNS, NUMERIC
from bko.generate.guidance import Basis, Finding, Guidance, Profile, Relationship, Spread
from bko.generate.messages import Level, reaches
from bko.ontology.mapping import RANGES
from bko.ontology.schema import Clause, Codes, FieldMapping


UNITS: Final = {
    "a": "years",
    "mm[Hg]": "mm Hg",
    "mg/dL": "mg/dL",
    "/min": "beats per minute",
    "mm": "mm",
}
"""Each `RANGES` UCUM unit as the prompt writes it."""

COHERENCE: Final = (
    "Keep each record clinically coherent: its values should fit the patient's disease status, "
    "age and sex together."
)
"""The sentence the `coherent` level adds."""

BASES: Final = {
    Basis.ABSENT: (
        "No record of this group in the training data shows the findings below. At least one "
        "record should show each; otherwise keep each as frequent as is clinically plausible for "
        "this group."
    ),
    Basis.RARE: (
        "The findings below are rarer in this group than among comparable patients. Show each "
        "about as often as among comparable patients, as far as is clinically plausible for this "
        "group."
    ),
    Basis.SHOWN: (
        "Records of this group show the findings below; not every record shows each. Keep each "
        "as frequent as is clinically plausible for this group."
    ),
}
"""The sentence opening the `findings` block, by why the findings were chosen."""

_FIELD_UNITS: Final = {bound.field: UNITS[bound.unit] for bound in RANGES}


def _codes(name: str) -> dict[int, str]:
    if name == "sex":
        return dict(SEXES)
    if name == LABEL:
        return dict(LABEL_MEANINGS)
    return dict(CODEBOOK.get(name, {}))


def _column(name: str) -> str:
    codes = _codes(name)
    if codes:
        details = "".join(f"; {code} = {meaning}" for code, meaning in codes.items())
    else:
        bound = next(bound for bound in RANGES if bound.field == name)
        unit = UNITS[bound.unit]
        shown = "" if DESCRIPTIONS[name].endswith(unit) else f"{unit}, "
        details = f"; {shown}plausible from {bound.low:g} to {bound.high:g}"
    return f"- {name}: {DESCRIPTIONS[name]}{details}"


def _alternatives(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"


def _clause(clause: Clause) -> str:
    if isinstance(clause, Codes):
        codes = sorted(clause.codes)
        meanings = _codes(clause.field)
        shown = ", ".join(meanings[code] for code in codes)
        return f"{clause.field} is {_alternatives([str(code) for code in codes])} ({shown})"
    value = f"{clause.value:.1f}" if clause.field in FRACTIONAL else f"{clause.value:g}"
    return f"{clause.field} is at least {value} {_FIELD_UNITS[clause.field]}"


def _state(row: FieldMapping) -> str:
    return " and ".join(_clause(clause) for clause in row.when)


def _bottom_up(relationships: tuple[Relationship, ...]) -> list[Relationship]:
    """Order `relationships` lowest class first, each level in its stored order.

    A relationship is placed once no remaining one concludes its antecedent. `is_a` is acyclic,
    but if what remains forms a cycle it is placed whole, in stored order, so the loop ends.
    """
    pending = list(relationships)
    ordered: list[Relationship] = []
    while pending:
        above = {one.statement.consequent for one in pending}
        lowest = [one for one in pending if one.statement.antecedent not in above] or pending
        ordered += lowest
        pending = [one for one in pending if one not in lowest]
    return ordered


def _finding(finding: Finding, *, chains: bool) -> str:
    states = " or ".join(_state(row) for row in finding.states)
    labels = ", ".join(dict.fromkeys(row.label for row in finding.states))
    line = f"- {states} ({labels})"
    if chains:
        line += ": " + "; ".join(
            f"{one.antecedent} is a kind of {one.consequent}"
            for one in _bottom_up(finding.relationships)
        )
    return line


def _number(name: str, value: float) -> str:
    return f"{value:.1f}" if name in FRACTIONAL else f"{value:.0f}"


def _profile(profile: Profile) -> str:
    lines = [
        (
            "Among comparable patients (the same target, other age and sex groups; "
            f"{profile.records} records), these findings occur at these rates:"
        ),
        *(
            f"- {share.label} ({' or '.join(_state(row) for row in share.states)}): "
            f"{round(100 * share.share)}%"
            for share in profile.shares
        ),
    ]
    return "\n".join(lines)


def _summary(summary: Mapping[str, Spread]) -> str:
    lines = [
        "Among the same patients, numeric fields run as follows (median; middle half):",
        *(
            f"- {name}: {_number(name, spread.median)}; {_number(name, spread.low)} to "
            f"{_number(name, spread.high)} {_FIELD_UNITS[name]}"
            for name, spread in summary.items()
        ),
    ]
    return "\n".join(lines)


def _system() -> str:
    lines = [
        (
            "You write synthetic patient records for a study of heart-disease prediction. Each "
            "record is one adult patient referred for coronary angiography. Invent new patients; "
            "do not reproduce records from any published dataset."
        ),
        "",
        "Each record has these fields:",
        *(_column(name) for name in FRAME_COLUMNS),
        "",
        (
            "Whole numbers except oldpeak, which has one decimal place. Every value must be "
            "plausible for a real patient."
        ),
    ]
    return "\n".join(lines)


SYSTEM: Final = _system()
"""The system prompt of every call."""


def render(guidance: Guidance, rows: int, level: Level) -> tuple[str, str]:
    """Render one call's prompt.

    Args:
        guidance: What the request says about the cell.
        rows: The records the call asks for.
        level: The guidance level; each component up to it is rendered, in ladder order.

    Returns:
        The system prompt and the user turn.
    """
    low, high = guidance.interval["age"]
    within = "; ".join(
        f"{name} {guidance.interval[name][0]:g} to {guidance.interval[name][1]:g} "
        f"{_FIELD_UNITS[name]}"
        for name in NUMERIC
    )
    parts = [
        (
            f"Write exactly {rows} records of patients in this group: sex {guidance.sex} "
            f"({SEXES[guidance.sex]}), target {guidance.label} "
            f"({LABEL_MEANINGS[guidance.label]}), age {low:g} to {high:g} years. The training "
            f"data holds {guidance.records} records of this group."
        ),
        f"Keep each numeric field within: {within}.",
    ]
    if reaches(level, Level.COHERENT):
        parts.append(COHERENCE)
    if reaches(level, Level.FINDINGS) and guidance.basis is not Basis.NONE and guidance.findings:
        chains = reaches(level, Level.ONTOLOGY)
        lines = [_finding(finding, chains=chains) for finding in guidance.findings]
        parts.append("\n".join([BASES[guidance.basis], *lines]))
    if reaches(level, Level.PROFILE) and guidance.profile is not None:
        parts.append(_profile(guidance.profile))
    if reaches(level, Level.SUMMARY) and guidance.summary is not None:
        parts.append(_summary(guidance.summary))
    return SYSTEM, "\n\n".join(parts)
