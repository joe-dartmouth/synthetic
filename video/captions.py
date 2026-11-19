"""The demo video's scene order, titles and on-screen labels, and how long each label holds.

Captions are short labels for a presenter to talk over. A caption that states a number of the
running example formats it from `video.model`; a caption that states a result quotes a figure of
`docs/EVIDENCE.md`, listed in `RECALL_BARS` and `EVIDENCE_FIGURES`, which
`tests/test_video_captions.py` holds to that file. No module here imports ManimGL.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from video import model


WORDS_PER_SECOND: Final = 2.5
"""The reading speed a caption's hold allows for."""

FADE_SECONDS: Final = 0.4
"""How long a caption takes to fade in."""

PAUSE_SECONDS: Final = 1.0
"""What each hold adds after the reading time."""

MIN_HOLD_SECONDS: Final = 2.5
"""The shortest hold."""

SCENE_END_SECONDS: Final = 1.5
"""How long each scene's finished diagram holds after its last caption."""


@dataclass(frozen=True, slots=True)
class Section:
    """One scene of the video.

    Attributes:
        name: The Scene class's name.
        title: The title shown at the top throughout.
        captions: The labels, one per beat, in order.
    """

    name: str
    title: str
    captions: tuple[str, ...]


_N = model.CELL_RECORDS[model.CELL]
_TOTAL = sum(model.CELL_RECORDS.values())
_ST_PRIOR = model.HOLDERS["HP:0012250"] / _N
_EKG_PRIOR = model.HOLDERS["HP:0003115"] / _N
_CALCIFICATION_GROUP = model.GROUP_HOLDERS["HP:0001717"] / model.GROUP_RECORDS
_W = model.GROUP_WEIGHT

SECTIONS: Final[tuple[Section, ...]] = (
    Section(
        "ThinCells",
        "The problem: thin cells",
        (
            f"UCI Heart Disease: {_TOTAL} training records",
            "Cells: age band, sex, diagnosis",
            f"Women under 55 with heart disease: {_N} records",
            f"Under {model.COUNT_GAP_BELOW} records: a count-based gap",
        ),
    ),
    Section(
        "Ontology",
        "Medical knowledge: an ontology",
        ("HPO links each finding to broader ones", "is_a: narrow to broad"),
    ),
    Section(
        "INodesAndSNodes",
        "BKB: I-nodes and S-nodes",
        (
            "I-node: a component and one of its states",
            f'Each class: one component, state "{model.STATE}"',
            "S-node: P(head | tail)",
            "is_a becomes P(parent | child) = 1",
            "An empty tail is a prior",
        ),
    ),
    Section(
        "DatasetFragment",
        "The cell's data as a fragment",
        (
            "Prior = share of the cell's records showing the concept",
            f"ST segment depression: {model.HOLDERS['HP:0012250']} of {_N}, prior {_ST_PRIOR}",
            f"Abnormal EKG: {model.HOLDERS['HP:0003115']} of {_N}, prior {_EKG_PRIOR}",
            "Never shown: no prior",
        ),
    ),
    Section(
        "FusingFragments",
        "Fusion 1: one BKB from many sources",
        (
            "Every S-node is kept",
            "Each gains a source I-node",
            "Two sources at weight 1: prior 1/2 each",
            "Sources may disagree: both claims stay",
            "UCI pipeline: MONDO, HPO and data, 1/3 each",
            "Ontology + this cell's data = the cell's BKO",
        ),
    ),
    Section(
        "IncompletenessKept",
        "A BKO keeps what is missing",
        (
            "BKO = ontology logic + BKB",
            "A statement the data never supports stays",
            "Removing it changes no probability (Santos et al. 2024, Prop. 5.1.1)",
        ),
    ),
    Section(
        "GroundingAndGaps",
        "Finding gaps: grounding",
        (
            "Start from the concepts the records show",
            "Pass 1",
            "Pass 2",
            "Never reached: ungrounded",
            "Gap = a (cell, statement) pair left ungrounded",
            f"This cell: 14 gaps, {len(model.gaps())} drawn",
            f"{model.CELL_RECORDS['<55/female/0']} records, no count gap, still 2 BKO gaps",
        ),
    ),
    Section(
        "WhatAContradictionIs",
        "Contradictions: data against ontology",
        (
            "HPO: ST depression is a kind of abnormal EKG",
            "One record shows the child without the parent",
            "A mapping says when a record shows a concept, nothing more",
            "Exercise test and resting EKG: different fields",
            "UCI: 40 of 92 records",
            "Diabetes 130: 757 of 838 records",
        ),
    ),
    Section(
        "FiveKinds",
        "Five kinds of contradiction",
        (
            f"narrow-without-broad: 1 of this cell's {_N} records",
            f"child-share-above-parent: {_ST_PRIOR} against {_EKG_PRIOR}",
            "all-show: both records show ST depression",
            "code-overlap: angina and chest pain share code 1",
            "empty-cell: none found; removing a whole group makes one",
        ),
    ),
    Section(
        "HandlingContradictions",
        "Our rule: report, never force",
        (
            "Fusion keeps both S-nodes: 1 and 0.5",
            "Gap detection: a held child grounds its parent",
            "Shares stay in [0, 1]; an empty cell uses its label group's",
            "Sampler: never forces a parent; nested codes stay nested",
            "No match in the cell: copy from the label group",
            "The contradictions command lists every kind",
            "Diabetes 130: forcing parents would rewrite most such records",
        ),
    ),
    Section(
        "LlmGuidedFill",
        "Filling the gap, path 1: a language model",
        (
            f"Ask for {model.COUNT_GAP_BELOW} records for this cell",
            "Prompt: cell, ranges, gaps, is_a chains, shares",
            "No patient record is sent",
            "Validation: each field's codes and ranges",
        ),
    ),
    Section(
        "BkoDirect",
        "Path 2: bko-direct, no model call",
        (
            f"share = (n·p_cell + {_W}·p_group) / (n + {_W})",
            (
                f"Calcification: ({_N}·0 + {_W}·{_CALCIFICATION_GROUP:.3f}) / {_N + _W}"
                f" = {model.share('HP:0001717'):.3f}"
            ),
            "Each concept drawn at its share",
            "Fields copied from matching real records",
        ),
    ),
    Section(
        "ClosingTheLoop",
        "Fusion 2: records rejoin the BKO",
        (
            f"{model.COUNT_GAP_BELOW} validated records join the cell",
            "Detection runs again",
            "Per-subgroup gaps: 16 before, 0 after",
        ),
    ),
    Section(
        "DoesItHelp",
        "Does it help a classifier?",
        (
            "UCI, 100 splits: no benefit shown, no harm shown",
            "Diabetes 130: remove every man aged 60-79 readmitted within 30 days",
            "Recall on the removed group",
            "bko-direct against CTGAN: +0.0595 [+0.0115, +0.1075]",
            "bko-direct was added after the plan: no verdict",
            "Boosted trees: 0.3591 against 0.5853",
            "19 real rows kept: level with ADASYN and CTGAN",
        ),
    ),
    Section(
        "Recap",
        "The loop",
        (
            "Represent, detect, generate, validate, evaluate",
            "Gains where a group was missing; none on UCI",
        ),
    ),
)
"""Every scene, in the order the video plays them."""

ORDER: Final[tuple[str, ...]] = tuple(section.name for section in SECTIONS)
"""The Scene class names, in play order: the one source of the concat list."""

KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("narrow-without-broad", "records showing a child concept without its parent"),
    ("child-share-above-parent", "a child concept more common than its parent in a cell"),
    ("empty-cell", "a cell with no records"),
    ("all-show", "every record of a cell shows a concept"),
    ("code-overlap", "two concepts claim the same code on one field"),
)
"""The five contradiction kinds, in the census's order, with their glosses."""

RECALL_BARS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "none": "0.0660",  # EVIDENCE E6
        "bko": "0.3072",  # EVIDENCE E6
        "ctgan": "0.4782",  # EVIDENCE E6
        "bko-direct": "0.5377",  # EVIDENCE E6
        "real": "0.5565",  # EVIDENCE E6
    }
)
"""Diabetes 130's removed-cell recall per arm at 1.0, as `docs/EVIDENCE.md` states it."""

EVIDENCE_FIGURES: Final[tuple[str, ...]] = (
    "+0.0595",  # EVIDENCE E6: bko-direct minus CTGAN
    "+0.0115",  # EVIDENCE E6: its lower bound
    "+0.1075",  # EVIDENCE E6: its upper bound
    "0.3591",  # EVIDENCE E6: boosted trees, bko-direct
    "0.5853",  # EVIDENCE E6: boosted trees, real
)
"""The other `docs/EVIDENCE.md` figures the captions state."""


def section(name: str) -> Section:
    """The section whose Scene class is `name`.

    Raises:
        KeyError: If no section has that name.
    """
    for each in SECTIONS:
        if each.name == name:
            return each
    raise KeyError(name)


def hold_seconds(caption: str) -> float:
    """How long `caption` holds after its fade, its beat's animation running inside the hold."""
    return max(MIN_HOLD_SECONDS, len(caption.split()) / WORDS_PER_SECOND + PAUSE_SECONDS)


def planned_seconds() -> float:
    """The video's planned length: every fade and hold, plus each scene's end."""
    return sum(
        sum(FADE_SECONDS + hold_seconds(caption) for caption in each.captions) + SCENE_END_SECONDS
        for each in SECTIONS
    )
