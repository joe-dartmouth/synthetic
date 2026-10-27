"""The prompt: one system text every call shares, and a user turn built up the guidance ladder.

The system text names every column with its meaning, its codes or its unit and plausible range,
and asks for new patients. The user turn states the cell and the record count, then adds one
component per level reached. Both read the frame given, UCI's by default. No prompt holds a
record. ADR `research/record-generation-design` records what each level sends.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import FRACTIONAL, UCI_FRAME
from bko.generate.guidance import Basis, Finding, Guidance, Profile, Relationship, Spread
from bko.generate.messages import Level, reaches
from bko.ontology.schema import Clause, Codes, FieldMapping


UNITS: Final = {
    "a": "years",
    "mm[Hg]": "mm Hg",
    "mg/dL": "mg/dL",
    "/min": "beats per minute",
    "mm": "mm",
}
"""Each range constraint's UCUM unit as the prompt writes it."""

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


@functools.cache
def field_units(frame: FrameSpec) -> Mapping[str, str]:
    """Return each of `frame`'s numeric fields' unit as the prompt writes it."""
    return MappingProxyType({bound.field: UNITS[bound.unit] for bound in frame.ranges})


def _codes(name: str, frame: FrameSpec) -> dict[int, str]:
    if name == frame.label:
        return dict(frame.label_meanings)
    return dict(frame.codebook.get(name, {}))


def _column(name: str, frame: FrameSpec) -> str:
    codes = _codes(name, frame)
    description = frame.descriptions[name]
    if codes:
        details = "".join(f"; {code} = {meaning}" for code, meaning in codes.items())
    else:
        bound = next(bound for bound in frame.ranges if bound.field == name)
        unit = UNITS[bound.unit]
        # A description that already names its unit, in any case, does not get it twice.
        shown = "" if unit.lower() in description.lower() else f"{unit}, "
        details = f"; {shown}plausible from {bound.low:g} to {bound.high:g}"
    return f"- {name}: {description}{details}"


def _alternatives(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"


def _clause(clause: Clause, frame: FrameSpec) -> str:
    if isinstance(clause, Codes):
        codes = sorted(clause.codes)
        meanings = _codes(clause.field, frame)
        shown = ", ".join(meanings[code] for code in codes)
        return f"{clause.field} is {_alternatives([str(code) for code in codes])} ({shown})"
    value = (
        f"{clause.value:.{frame.fraction_digits}f}"
        if clause.field in frame.fractional
        else f"{clause.value:g}"
    )
    return f"{clause.field} is at least {value} {field_units(frame)[clause.field]}"


def _state(row: FieldMapping, frame: FrameSpec) -> str:
    return " and ".join(_clause(clause, frame) for clause in row.when)


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


def _finding(finding: Finding, frame: FrameSpec, *, chains: bool) -> str:
    states = " or ".join(_state(row, frame) for row in finding.states)
    labels = ", ".join(dict.fromkeys(row.label for row in finding.states))
    line = f"- {states} ({labels})"
    if chains:
        line += ": " + "; ".join(
            f"{one.antecedent} is a kind of {one.consequent}"
            for one in _bottom_up(finding.relationships)
        )
    return line


def _number(name: str, value: float, frame: FrameSpec) -> str:
    return f"{value:.{frame.fraction_digits}f}" if name in frame.fractional else f"{value:.0f}"


def _profile(profile: Profile, frame: FrameSpec) -> str:
    lines = [
        (
            f"Among comparable patients (the same {frame.label}, other age and sex groups; "
            f"{profile.records} records), these findings occur at these rates:"
        ),
        *(
            f"- {share.label} ({' or '.join(_state(row, frame) for row in share.states)}): "
            f"{round(100 * share.share)}%"
            for share in profile.shares
        ),
    ]
    return "\n".join(lines)


def _summary(summary: Mapping[str, Spread], frame: FrameSpec) -> str:
    units = field_units(frame)
    lines = [
        "Among the same patients, numeric fields run as follows (median; middle half):",
        *(
            f"- {name}: {_number(name, spread.median, frame)}; "
            f"{_number(name, spread.low, frame)} to {_number(name, spread.high, frame)} "
            f"{units[name]}"
            for name, spread in summary.items()
        ),
    ]
    return "\n".join(lines)


def _decimals(digits: int) -> str:
    return "one decimal place" if digits == 1 else f"{digits} decimal places"


def _system_text(frame: FrameSpec) -> str:
    fractional = list(frame.fractional)
    lines = [
        (
            "You write synthetic patient records for a study of heart-disease prediction. Each "
            "record is one adult patient referred for coronary angiography. Invent new patients; "
            "do not reproduce records from any published dataset."
        ),
        "",
        "Each record has these fields:",
        *(_column(name, frame) for name in frame.columns),
        "",
        (
            f"Whole numbers except {_alternatives(fractional)}, which "
            f"{'has' if len(fractional) == 1 else 'have'} {_decimals(frame.fraction_digits)}. "
            "Every value must be plausible for a real patient."
        ),
    ]
    return "\n".join(lines)


@functools.cache
def system(frame: FrameSpec) -> str:
    """Return the system prompt of every call for `frame`'s records, built once per frame."""
    return _system_text(frame)


def _system() -> str:
    """UCI's system prompt with this module's `FRACTIONAL` as its fractional fields.

    A test replaces `FRACTIONAL` to show the closing sentence follows the fractional fields.
    """
    return _system_text(dataclasses.replace(UCI_FRAME, fractional=FRACTIONAL))


SYSTEM: Final = system(UCI_FRAME)
"""The system prompt of every call for UCI records."""


def render(
    guidance: Guidance, rows: int, level: Level, *, frame: FrameSpec = UCI_FRAME
) -> tuple[str, str]:
    """Render one call's prompt.

    Args:
        guidance: What the request says about the cell.
        rows: The records the call asks for.
        level: The guidance level; each component up to it is rendered, in ladder order.
        frame: The frame whose fields, codes, units and decimals the prompt states.

    Returns:
        The system prompt and the user turn.
    """
    units = field_units(frame)
    low, high = guidance.interval["age"]
    within = "; ".join(
        f"{name} {guidance.interval[name][0]:g} to {guidance.interval[name][1]:g} {units[name]}"
        for name in frame.numeric
    )
    parts = [
        (
            f"Write exactly {rows} records of patients in this group: sex {guidance.sex} "
            f"({_codes('sex', frame)[guidance.sex]}), {frame.label} {guidance.label} "
            f"({frame.label_meanings[guidance.label]}), age {low:g} to {high:g} "
            f"{units['age']}. The training data holds {guidance.records} records of this group."
        ),
        f"Keep each numeric field within: {within}.",
    ]
    if reaches(level, Level.COHERENT):
        parts.append(COHERENCE)
    if reaches(level, Level.FINDINGS) and guidance.basis is not Basis.NONE and guidance.findings:
        chains = reaches(level, Level.ONTOLOGY)
        lines = [_finding(finding, frame, chains=chains) for finding in guidance.findings]
        parts.append("\n".join([BASES[guidance.basis], *lines]))
    if reaches(level, Level.PROFILE) and guidance.profile is not None:
        parts.append(_profile(guidance.profile, frame))
    if reaches(level, Level.SUMMARY) and guidance.summary is not None:
        parts.append(_summary(guidance.summary, frame))
    return system(frame), "\n\n".join(parts)
