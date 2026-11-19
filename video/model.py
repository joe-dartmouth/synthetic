"""The demo video's running example, built from real `bko.bkb` objects and no ManimGL.

The example is UCI Heart Disease's cell `<55/female/1` (women under 55 with heart disease) in the
default split, on an eight-class part of HPO. The ontology fragment encodes each `is_a` edge as
P(parent | child) = 1 (ADR `research/bko-fragment-encoding`); the dataset fragment gives each
concept the cell's records show its share of them as a prior (ADR
`research/dataset-fragment-encoding`); a concept's share for the sampler blends the cell's prior
with its label group's (ADR `research/bkb-direct-generation`). `tests/test_video_model.py` holds
this module to `bko.gaps`, and `tests/test_video_model_releases.py` holds its literals to `data/`.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.bkb.fusion import fuse, is_source
from bko.bkb.inference import instantiate
from bko.bkb.structure import BKB


STATE: Final = "a"
"""Each class's one state: the individual the BKO reasons about."""

CELL: Final = "<55/female/1"
"""The cell the video follows: age band, sex and diagnosis."""

LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "HP:0001717": "Coronary artery calcification",
        "HP:0003207": "Arterial calcification",
        "HP:0004934": "Vascular calcification",
        "HP:0100749": "Chest pain",
        "HP:0012531": "Pain",
        "HP:0012250": "ST segment depression",
        "HP:0012249": "Abnormal ST segment",
        "HP:0003115": "Abnormal EKG",
    }
)
"""The eight HPO classes on screen and their labels."""

EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("HP:0001717", "HP:0003207"),
    ("HP:0003207", "HP:0004934"),
    ("HP:0100749", "HP:0012531"),
    ("HP:0012250", "HP:0012249"),
    ("HP:0012249", "HP:0003115"),
)
"""The `is_a` edges between them, as (child, parent)."""

CELL_RECORDS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "<55/female/0": 30,
        "<55/female/1": 2,
        "<55/male/0": 47,
        "<55/male/1": 32,
        ">=55/female/0": 27,
        ">=55/female/1": 18,
        ">=55/male/0": 24,
        ">=55/male/1": 58,
    }
)
"""Each cell of the training split and its record count."""

HOLDERS: Final[Mapping[str, int]] = MappingProxyType(
    {"HP:0012250": 2, "HP:0003115": 1, "HP:0001717": 0, "HP:0100749": 0}
)
"""How many of the cell's records show each concept the video counts."""

GROUP_RECORDS: Final = 110
"""The records of the cell's label group (diagnosis 1)."""

GROUP_HOLDERS: Final[Mapping[str, int]] = MappingProxyType({"HP:0001717": 78})
"""How many of the label group's records show each concept the video blends."""

GROUP_WEIGHT: Final = 20
"""The label group's weight in a share: `bko.gaps.shares.GROUP_WEIGHT`."""

COUNT_GAP_BELOW: Final = 20
"""A cell under this many records is a count-based gap: `UNDERREPRESENTED_BELOW`."""


def ontology_fragment() -> BKB:
    """The HPO fragment: one S-node of probability 1 per `is_a` edge, child to parent."""
    fragment = BKB("hpo")
    for concept in LABELS:
        fragment.add_inode(concept, STATE)
    for child, parent in EDGES:
        fragment.add_snode(parent, STATE, 1.0, [(child, STATE)])
    return fragment


def dataset_fragment() -> BKB:
    """The cell's fragment: a prior per concept a record shows, its share of the records."""
    fragment = BKB(f"uci-heart-disease/{CELL}")
    for concept, holders in sorted(HOLDERS.items()):
        if holders > 0:
            fragment.add_inode(concept, STATE)
            fragment.add_snode(concept, STATE, holders / CELL_RECORDS[CELL])
    return fragment


def fused() -> BKB:
    """The cell's BKO: both fragments fused at reliability 1 each."""
    data = dataset_fragment()
    return fuse([ontology_fragment(), data], [1.0, 1.0], name=data.name)


def source_priors() -> dict[tuple[str, str], float]:
    """Each source I-node's prior, keyed by (source component, fragment name)."""
    return {
        (snode.head[0], str(snode.head[1])): snode.prob
        for snode in fused().snodes
        if is_source(snode.head[0])
    }


def passes() -> tuple[tuple[tuple[str, str], ...], ...]:
    """Each instantiation pass's statements, as (tail class, head class), from the held concepts.

    A pass's data priors are left out, since their tails name only a source; a pass left with
    nothing is dropped.
    """
    bkb = fused()
    held = [(concept, STATE) for concept, count in sorted(HOLDERS.items()) if count > 0]
    shown: list[tuple[tuple[str, str], ...]] = []
    for step in instantiate(bkb, held):
        statements = tuple(
            (tail[0], bkb.snodes[index].head[0])
            for index in step
            for tail in bkb.snodes[index].tail
            if not is_source(tail[0])
        )
        if statements:
            shown.append(statements)
    return tuple(shown)


def gaps() -> tuple[tuple[str, str], ...]:
    """The `is_a` edges no pass adds, sorted: the cell's ungrounded statements on screen."""
    added = {statement for step in passes() for statement in step}
    return tuple(sorted(edge for edge in EDGES if edge not in added))


def share(concept: str) -> float:
    """The sampler's share of `concept` in the cell, blended with its label group's prior."""
    n = CELL_RECORDS[CELL]
    p_cell = HOLDERS[concept] / n
    p_group = GROUP_HOLDERS[concept] / GROUP_RECORDS
    return (n * p_cell + GROUP_WEIGHT * p_group) / (n + GROUP_WEIGHT)
