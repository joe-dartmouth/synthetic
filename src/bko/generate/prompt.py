"""The prompt: one system text every call shares, and a user turn built up the guidance ladder.

The system text opens with the frame's study and cohort, names every column with its meaning and its
codes or its unit, with no range, and asks for new patients. The user turn states the cell, the
record count and each numeric field's training-split bounds, then adds one component per level
reached, and for a prescribing request (`bko.generate.skeleton`) one line per asked record naming
the concepts it must show. Both read the frame given, a required keyword. The BKO prompts hold no
record; `render_shown` shows training records to the CLLM arms (ADR `research/cllm-arm-design`).
`sections` reads a user turn back into those components, as `bko.evaluate.trace` names them.
ADR `research/record-generation-design` records what each level sends.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.generate.guidance import Basis, Finding, Guidance, Profile, Relationship, Spread
from bko.generate.messages import Level, reaches
from bko.ontology.schema import Clause, Codes, FieldMapping


if TYPE_CHECKING:
    from bko.generate.skeleton import Skeleton


type Shown = tuple[Mapping[str, int | float], ...]
"""Training records a request shows, each a value per frame column in the frame's order."""

SHOWN: Final = "These records from the training data are examples of real patients, one per line:"
"""The sentence that opens a shown request's list of training records."""

UNITS: Final = {
    "a": "years",
    "mm[Hg]": "mm Hg",
    "mg/dL": "mg/dL",
    "/min": "beats per minute",
    "mm": "mm",
    "d": "days",
    "10.a": "decades",
    "{tests}": "lab tests",
    "{procedures}": "procedures",
    "{medications}": "medications",
    "{visits}": "visits",
    "{diagnoses}": "diagnoses",
}
"""Each range constraint's UCUM unit as the prompt writes it; a count's unit is a UCUM annotation
naming what it counts."""

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

PRESCRIBED: Final = (
    "Each record below must show exactly the listed findings and no other finding from this list; "
    "these lines take precedence over any finding named above."
)
"""The sentence opening a prescribing request's block of record lines."""

NO_FINDING: Final = "none of these findings"
"""A record line's text for a record that must show no named finding."""

SUMMARY_OPENS: Final = (
    "Among the same patients, numeric fields run as follows (median; middle half):"
)
"""The sentence that opens the `summary` level's block."""

_OPENING: Final = "Write exactly "
_BOUNDS: Final = "Keep each numeric field within: "
_PROFILE: Final = "Among comparable patients (the same "


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
        unit = field_units(frame)[name]
        # A description that already names its unit, in any case, does not get it twice.
        details = "" if unit.lower() in description.lower() else f"; {unit}"
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
            f"{_PROFILE}{frame.label}, other age and sex groups; "
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
        SUMMARY_OPENS,
        *(
            f"- {name}: {_number(name, spread.median, frame)}; "
            f"{_number(name, spread.low, frame)} to {_number(name, spread.high, frame)} "
            f"{units[name]}"
            for name, spread in summary.items()
        ),
    ]
    return "\n".join(lines)


def _prescribed(skeleton: Skeleton, frame: FrameSpec) -> str:
    labels = {one.concept: one.label for one in skeleton.named}
    lines = [
        PRESCRIBED,
        *(
            f"- {one.label} ({' or '.join(_state(row, frame) for row in one.states)})"
            for one in skeleton.named
        ),
        *(
            f"Record {number}: "
            + ("; ".join(labels[concept] for concept in sorted(line)) or NO_FINDING)
            for number, line in enumerate(skeleton.lines, start=1)
        ),
    ]
    return "\n".join(lines)


def _decimals(digits: int) -> str:
    return "one decimal place" if digits == 1 else f"{digits} decimal places"


def _precision(frame: FrameSpec) -> str:
    fractional = list(frame.fractional)
    if not fractional:
        return "Every field is a whole number."
    return (
        f"Whole numbers except {_alternatives(fractional)}, which "
        f"{'has' if len(fractional) == 1 else 'have'} {_decimals(frame.fraction_digits)}."
    )


def _system_text(frame: FrameSpec) -> str:
    lines = [
        (
            f"You write synthetic patient records for a study of {frame.study}. Each record is "
            f"one {frame.cohort}. Invent new patients; do not reproduce records from any "
            "published dataset."
        ),
        "",
        "Each record has these fields:",
        *(_column(name, frame) for name in frame.columns),
        "",
        f"{_precision(frame)} Every value must be plausible for a real patient.",
    ]
    return "\n".join(lines)


@functools.cache
def system(frame: FrameSpec) -> str:
    """Return the system prompt of every call for `frame`'s records, built once per frame."""
    return _system_text(frame)


SYSTEM: Final = system(UCI_FRAME)
"""The system prompt of every call for UCI records."""


def render(
    guidance: Guidance,
    rows: int,
    level: Level,
    *,
    frame: FrameSpec,
    skeleton: Skeleton | None = None,
) -> tuple[str, str]:
    """Render one call's prompt.

    Args:
        guidance: What the request says about the cell.
        rows: The records the call asks for.
        level: The guidance level; each component up to it is rendered, in ladder order.
        frame: The frame whose fields, codes, units and decimals the prompt states.
        skeleton: A prescribing request's lines, one per asked record, rendered after every
            level's component: the named concepts with their field states, then each record's
            labels sorted by curie; None renders none.

    Returns:
        The system prompt and the user turn. Each bound is written with `:g`, six significant
        digits, so a bound needing more would state a value the drop rule does not enforce.
    """
    parts = _opening(guidance, rows, frame)
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
    if skeleton is not None:
        parts.append(_prescribed(skeleton, frame))
    return system(frame), "\n\n".join(parts)


def render_shown(
    guidance: Guidance, rows: int, shown: Shown, *, frame: FrameSpec
) -> tuple[str, str]:
    """Render one call's prompt that shows training records, as the CLLM arms ask.

    Args:
        guidance: What the request says about the cell; only its cell, record count and bounds
            are read.
        rows: The records the call asks for.
        shown: The training records shown, each a JSON object on its own line in the frame's
            column order.
        frame: The frame whose fields, codes, units and decimals the prompt states.

    Returns:
        The system prompt and the user turn: `render`'s first sentence and bounds, then the shown
        records.
    """
    listed = "\n".join(
        json.dumps({name: record[name] for name in frame.columns}) for record in shown
    )
    parts = [*_opening(guidance, rows, frame), f"{SHOWN}\n{listed}"]
    return system(frame), "\n\n".join(parts)


def _opening(guidance: Guidance, rows: int, frame: FrameSpec) -> list[str]:
    """Every user turn's first two components: the cell and its record count, then the bounds."""
    units = field_units(frame)
    low, high = guidance.interval["age"]
    within = "; ".join(
        f"{name} {guidance.interval[name][0]:g} to {guidance.interval[name][1]:g} {units[name]}"
        for name in frame.numeric
    )
    return [
        (
            f"{_OPENING}{rows} records of patients in this group: sex {guidance.sex} "
            f"({_codes('sex', frame)[guidance.sex]}), {frame.label} {guidance.label} "
            f"({frame.label_meanings[guidance.label]}), age {low:g} to {high:g} "
            f"{units['age']}. The training data holds {guidance.records} records of this group."
        ),
        f"{_BOUNDS}{within}.",
    ]


@dataclass(frozen=True, slots=True)
class Sections:
    """A user turn's components, as `render` or `render_shown` wrote them.

    Attributes:
        opening: The cell, its record count and the training records it holds.
        bounds: Each numeric field's training-split bounds.
        coherence: Whether the `coherent` level's sentence is present.
        basis: Why the findings were chosen, by the block's opening sentence; None with no block.
        findings: Each finding line as rendered: its field states, its labels and, from the
            `ontology` level, its `is_a` chain.
        profile: The `profile` level's block, or None.
        summary: The `summary` level's block, or None.
        prescribed: A prescribing request's block of record lines, or None.
        shown: A CLLM request's block of training records, or None.
    """

    opening: str
    bounds: str
    coherence: bool = False
    basis: Basis | None = None
    findings: tuple[str, ...] = ()
    profile: str | None = None
    summary: str | None = None
    prescribed: str | None = None
    shown: str | None = None

    def text(self) -> str:
        """The user turn the sections were read from."""
        findings = None
        if self.basis is not None:
            findings = "\n".join([BASES[self.basis], *self.findings])
        parts = [
            self.opening,
            self.bounds,
            COHERENCE if self.coherence else None,
            findings,
            self.profile,
            self.summary,
            self.prescribed,
            self.shown,
        ]
        return "\n\n".join(part for part in parts if part is not None)


def sections(user: str) -> Sections:
    """Read a user turn back into the components `render` or `render_shown` joined.

    Raises:
        ValueError: If the turn does not open with the cell and its bounds, or a block opens
            with a sentence no component writes.
    """
    blocks = user.split("\n\n")
    if len(blocks) < 2 or not (blocks[0].startswith(_OPENING) and blocks[1].startswith(_BOUNDS)):
        msg = "the prompt does not open with the cell's opening and its bounds"
        raise ValueError(msg)
    read = Sections(blocks[0], blocks[1])
    bases = {sentence: basis for basis, sentence in BASES.items()}
    for block in blocks[2:]:
        first, *rest = block.split("\n")
        if block == COHERENCE:
            read = dataclasses.replace(read, coherence=True)
        elif first in bases:
            read = dataclasses.replace(read, basis=bases[first], findings=tuple(rest))
        else:
            key = next((k for k, opens in _BLOCKS if first.startswith(opens)), None)
            if key is None:
                msg = f"no section of the prompt opens with {first!r}"
                raise ValueError(msg)
            read = dataclasses.replace(read, **{key: block})
    return read


_BLOCKS: Final = (
    ("profile", _PROFILE),
    ("summary", SUMMARY_OPENS),
    ("prescribed", PRESCRIBED),
    ("shown", SHOWN),
)
"""Each block `sections` names by its opening sentence, beside the findings block."""
