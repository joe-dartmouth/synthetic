"""The demo video's scene order, titles and on-screen text, and how long each beat holds.

Each beat shows a short label and, under it, a one-line explainer in plain words, for a class that
has not met BKBs before; a presenter talks over the video and pauses it. The video explains how the
system works and states no experiment result. A number of the running example is formatted from
`video.model`; a fact the example takes from the docs is a named constant whose sentence `SOURCED`
names, and `tests/test_video_captions.py` holds each to its doc. No module here imports ManimGL.
"""

from dataclasses import dataclass
from typing import Final

from video import model


WORDS_PER_SECOND: Final = 3.0
"""The reading speed a beat's hold allows for, label and explainer together."""

FADE_SECONDS: Final = 0.4
"""How long a beat's text takes to fade in."""

PAUSE_SECONDS: Final = 0.5
"""What each hold adds after the reading time."""

MIN_HOLD_SECONDS: Final = 3.0
"""The shortest hold."""

SCENE_END_SECONDS: Final = 1.5
"""How long each scene's finished diagram holds after its last beat."""


@dataclass(frozen=True, slots=True)
class Section:
    """One scene of the video.

    Attributes:
        name: The Scene class's name.
        title: The title shown at the top throughout.
        beats: Per beat, in order, its short label and its one-line explainer.
    """

    name: str
    title: str
    beats: tuple[tuple[str, str], ...]

    @property
    def captions(self) -> tuple[str, ...]:
        """The beats' labels, in order."""
        return tuple(label for label, _ in self.beats)


CELL_GAPS: Final = 14
"""The cell's BKO gaps over the whole mapped ontology (EVIDENCE C1, before fusion)."""
NEIGHBOUR_GAPS: Final = 2
"""The BKO gaps of `<55/female/0`, a cell with no count gap (EVIDENCE C1, before fusion)."""

_N = model.CELL_RECORDS[model.CELL]
_TOTAL = sum(model.CELL_RECORDS.values())
_ST_PRIOR = model.HOLDERS["HP:0012250"] / _N
_EKG_PRIOR = model.HOLDERS["HP:0003115"] / _N
_CALCIFICATION_GROUP = model.GROUP_HOLDERS["HP:0001717"] / model.GROUP_RECORDS
_CALCIFICATION = model.share("HP:0001717")
_W = model.GROUP_WEIGHT
_NEIGHBOUR = "<55/female/0"

SECTIONS: Final[tuple[Section, ...]] = (
    Section(
        "ThinCells",
        "The problem: thin cells",
        (
            (
                f"UCI Heart Disease: {_TOTAL} training records",
                "Each record is one patient: age, sex, test results and a diagnosis.",
            ),
            (
                "Cells: age band, sex, diagnosis",
                "We split the records into groups, one bar per group.",
            ),
            (
                f"Women under 55 with heart disease: {_N} records",
                "A classifier learns almost nothing about a group this small.",
            ),
            (
                f"Under {model.COUNT_GAP_BELOW} records: a count-based gap",
                "Counting says a group is thin, not what its records are missing.",
            ),
        ),
    ),
    Section(
        "Ontology",
        "Medical knowledge: an ontology",
        (
            (
                "HPO links each finding to broader ones",
                "HPO, the Human Phenotype Ontology, is a curated map of medical findings.",
            ),
            (
                "is_a: narrow to broad",
                "Each arrow reads 'is a kind of', from the narrow finding to the broad.",
            ),
        ),
    ),
    Section(
        "INodesAndSNodes",
        "BKB: I-nodes and S-nodes",
        (
            (
                "I-node: a component and one of its states",
                "A Bayesian Knowledge Base (BKB) has two node types. An I-node is a fact.",
            ),
            (
                f'Each class: one component, state "{model.STATE}"',
                f'State "{model.STATE}" stands for the patient: the patient has this finding.',
            ),
            (
                "S-node: P(head | tail)",
                "An S-node holds one probability: how likely the head is, given the tail.",
            ),
            (
                "is_a becomes P(parent | child) = 1",
                "A patient with the narrow finding surely has the broad one.",
            ),
            (
                "An empty tail is a prior",
                "A prior has no condition: it says how common the finding is.",
            ),
        ),
    ),
    Section(
        "DatasetFragment",
        "The cell's data as a fragment",
        (
            (
                "Prior = share of the cell's records showing the concept",
                "A mapping table turns each field value into the findings it shows.",
            ),
            (
                (
                    f"ST segment depression: {model.HOLDERS['HP:0012250']} of {_N},"
                    f" prior {_ST_PRIOR:.1f}"
                ),
                "Both records show ST depression on the exercise test.",
            ),
            (
                f"Abnormal EKG: {model.HOLDERS['HP:0003115']} of {_N}, prior {_EKG_PRIOR:.1f}",
                "Only one of the two shows an abnormal resting EKG.",
            ),
            (
                "Never shown: no prior",
                "No record shows calcification or chest pain, so the data is silent on them.",
            ),
        ),
    ),
    Section(
        "FusingFragments",
        "Fusion 1: one BKB from many sources",
        (
            (
                "Every S-node is kept",
                "Fusion merges the ontology's piece and the data's piece into one BKB.",
            ),
            (
                "Each gains a source I-node",
                "Each S-node is tagged with where it came from: HPO or the data.",
            ),
            (
                "Two sources at weight 1: prior 1/2 each",
                "Each source has a weight; here both are trusted equally.",
            ),
            (
                "Sources may disagree: both claims stay",
                "Abnormal EKG now has two supports, 1 and 0.5, and fusion keeps both.",
            ),
            (
                "UCI pipeline: MONDO, HPO and data, 1/3 each",
                "The full pipeline also fuses MONDO, an ontology of diseases.",
            ),
            (
                "Ontology + this cell's data = the cell's BKO",
                "Knowledge and data now sit in one probabilistic model per cell.",
            ),
        ),
    ),
    Section(
        "IncompletenessKept",
        "A BKO keeps what is missing",
        (
            (
                "BKO = ontology logic + BKB",
                "A Bayesian Knowledge-driven Ontology adds probabilities to an ontology.",
            ),
            (
                "A statement the data never supports stays",
                "No record backs the calcification chain, yet it stays in the model.",
            ),
            (
                "Removing it changes no probability (Santos et al. 2024, Prop. 5.1.1)",
                "So keeping it costs nothing, and it marks what the data lacks.",
            ),
        ),
    ),
    Section(
        "GroundingAndGaps",
        "Finding gaps: grounding",
        (
            (
                "Start from the concepts the records show",
                "Grounding asks which statements the data can reach.",
            ),
            ("Pass 1", "From ST depression, follow is_a up to Abnormal ST segment."),
            ("Pass 2", "Then up to Abnormal EKG. No further pass reaches anything new."),
            (
                "Never reached: ungrounded",
                "The data never touches these statements.",
            ),
            (
                "Gap = a (cell, statement) pair left ungrounded",
                "A gap is knowledge the cell's records do not cover.",
            ),
            (
                f"This cell: {CELL_GAPS} gaps, {len(model.gaps())} drawn",
                "The full ontology holds more statements than fit on screen.",
            ),
            (
                (
                    f"{model.CELL_RECORDS[_NEIGHBOUR]} records, no count gap,"
                    f" still {NEIGHBOUR_GAPS} BKO gaps"
                ),
                "A group can be big enough and still miss findings; counting cannot see it.",
            ),
        ),
    ),
    Section(
        "WhatAContradictionIs",
        "Contradictions: data against ontology",
        (
            (
                "HPO: ST depression is a kind of abnormal EKG",
                "By the ontology, every patient with ST depression has an abnormal EKG.",
            ),
            (
                "One record shows the child without the parent",
                "Record 1 shows ST depression with a normal EKG: a contradiction.",
            ),
            (
                "A mapping says when a record shows a concept, nothing more",
                "Each mapping reads its own field and does not check the ontology.",
            ),
            (
                "Exercise test and resting EKG: different fields",
                "Two tests at two times can disagree; neither record is an error.",
            ),
        ),
    ),
    Section(
        "FiveKinds",
        "Five kinds of contradiction",
        (
            (
                f"narrow-without-broad: 1 of this cell's {_N} records",
                "The case just shown: a narrow finding without its broad parent.",
            ),
            (
                f"child-share-above-parent: {_ST_PRIOR:.1f} against {_EKG_PRIOR:.1f}",
                "A narrow finding more common than its parent, which logic rules out.",
            ),
            (
                "all-show: both records show ST depression",
                "Every record shows it, so the data allows no patient without it.",
            ),
            (
                "code-overlap: angina and chest pain share code 1",
                "One field code maps to two concepts the ontology keeps apart.",
            ),
            (
                "empty-cell: a group with no records",
                "With no records, the data grounds nothing in that group.",
            ),
        ),
    ),
    Section(
        "HandlingContradictions",
        "Our rule: report, never force",
        (
            (
                "Fusion keeps both S-nodes: 1 and 0.5",
                "Conflicting supports sit side by side; neither is deleted.",
            ),
            (
                "Gap detection: a held child grounds its parent",
                "A record showing the child counts toward its parent too.",
            ),
            (
                "Shares stay in [0, 1]; an empty cell uses its label group's",
                "Each concept's rate stays a valid probability, even with no records.",
            ),
            (
                "Sampler: never forces a parent; nested codes stay nested",
                "New records are not rewritten to fit the ontology.",
            ),
            (
                "No match in the cell: copy from the label group",
                "If no record in the cell fits, it borrows from the same diagnosis.",
            ),
            (
                "The contradictions command lists every kind",
                "A report shows every contradiction so a person can review it.",
            ),
        ),
    ),
    Section(
        "LlmGuidedFill",
        "Filling the gap, path 1: a language model",
        (
            (
                f"Ask for {model.COUNT_GAP_BELOW} records for this cell",
                "A language model writes new records for the thin cell.",
            ),
            (
                "Prompt: cell, ranges, gaps, is_a chains, shares",
                "The prompt carries what the BKO found: gaps and how findings relate.",
            ),
            (
                "No patient record is sent",
                "Only summaries leave the machine, never a patient's row.",
            ),
            (
                "Validation: each field's codes and ranges",
                "Every returned record is checked; an invalid one is dropped.",
            ),
        ),
    ),
    Section(
        "BkoDirect",
        "Path 2: bko-direct, no model call",
        (
            (
                f"share = (n·p_cell + {_W}·p_group) / (n + {_W})",
                "Path 2 draws records straight from the BKO.",
            ),
            (
                (
                    f"Calcification: ({_N}·0 + {_W}·{_CALCIFICATION_GROUP:.3f}) / {_N + _W}"
                    f" = {_CALCIFICATION:.3f}"
                ),
                "Too few records, so the cell's rate is blended with its diagnosis group's.",
            ),
            (
                "Each concept drawn at its share",
                f"Each new record shows calcification with probability {_CALCIFICATION:.3f}.",
            ),
            (
                "Fields copied from matching real records",
                "Field values come from a real record with the same drawn findings.",
            ),
        ),
    ),
    Section(
        "ClosingTheLoop",
        "Fusion 2: records rejoin the BKO",
        (
            (
                f"{model.COUNT_GAP_BELOW} validated records join the cell",
                "The new records are added to the cell's data.",
            ),
            (
                "Detection runs again",
                "The new records show calcification and chest pain.",
            ),
            (
                "Statements the new records reach are no longer gaps",
                "Rerunning gap detection shows which gaps the fill closed.",
            ),
        ),
    ),
    Section(
        "HowWeEvaluate",
        "Evaluation: does it help a classifier?",
        (
            (
                "Hold out real records for testing",
                "The test patients are real, and the generator never sees them.",
            ),
            (
                "Train with and without the new records",
                "Same classifier and data; only the added records differ.",
            ),
            (
                "Score recall on the thin group",
                "Recall: of the group's real patients with the disease, how many are caught.",
            ),
            (
                "Compare with other ways to add records",
                "Baselines such as SMOTE and CTGAN fill the same group with no ontology.",
            ),
        ),
    ),
    Section(
        "Recap",
        "The loop",
        (
            (
                "Represent, detect, generate, validate, evaluate",
                "Build the BKO, find its gaps, fill them, check them, test them.",
            ),
            (
                "Fusion closes the loop",
                "Validated records are fused back, so detection sees what was filled.",
            ),
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

SOURCED: Final[tuple[tuple[str, str, str], ...]] = (
    (
        f"{CELL_GAPS} gaps",
        "docs/EVIDENCE.md",
        f"{CELL_GAPS} of them in the `<55`, female, target 1 cell",
    ),
    (
        f"still {NEIGHBOUR_GAPS} BKO gaps",
        "docs/EVIDENCE.md",
        f"{NEIGHBOUR_GAPS} in the `<55`, female, target 0 cell",
    ),
    ("Prop. 5.1.1", "docs/BACKGROUND.md", "Proposition 5.1.1 states that removing it"),
)
"""Each fact a label takes from the docs: the label's text, its doc and the doc's sentence."""


def section(name: str) -> Section:
    """The section whose Scene class is `name`.

    Raises:
        KeyError: If no section has that name.
    """
    for each in SECTIONS:
        if each.name == name:
            return each
    raise KeyError(name)


def hold_seconds(label: str, note: str) -> float:
    """How long a beat holds after its fade, its animation running inside the hold."""
    words = len(label.split()) + len(note.split())
    return max(MIN_HOLD_SECONDS, words / WORDS_PER_SECOND + PAUSE_SECONDS)


def planned_seconds() -> float:
    """The video's planned length: every fade and hold, plus each scene's end."""
    return sum(
        sum(FADE_SECONDS + hold_seconds(label, note) for label, note in each.beats)
        + SCENE_END_SECONDS
        for each in SECTIONS
    )
