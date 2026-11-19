"""The demo video's scenes in play order: the module `make video` hands ManimGL.

ManimGL renders `SCENES_IN_ORDER` as given. Importing this module raises when that order differs
from `video.captions.ORDER`, which the concat list is built from.
"""

from video.captions import ORDER
from video.sections.bkb import DatasetFragment, INodesAndSNodes, Ontology
from video.sections.contradictions import FiveKinds, HandlingContradictions, WhatAContradictionIs
from video.sections.evaluation import DoesItHelp, Recap
from video.sections.fusion import FusingFragments, GroundingAndGaps, IncompletenessKept
from video.sections.generation import BkoDirect, ClosingTheLoop, LlmGuidedFill
from video.sections.problem import ThinCells


SCENES_IN_ORDER = (
    ThinCells,
    Ontology,
    INodesAndSNodes,
    DatasetFragment,
    FusingFragments,
    IncompletenessKept,
    GroundingAndGaps,
    WhatAContradictionIs,
    FiveKinds,
    HandlingContradictions,
    LlmGuidedFill,
    BkoDirect,
    ClosingTheLoop,
    DoesItHelp,
    Recap,
)

_NAMES = tuple(scene.__name__ for scene in SCENES_IN_ORDER)
if _NAMES != ORDER:
    _MISSING = sorted(set(ORDER) - set(_NAMES))
    _EXTRA = sorted(set(_NAMES) - set(ORDER))
    msg = f"SCENES_IN_ORDER differs from captions.ORDER: missing {_MISSING}, extra {_EXTRA}"
    raise RuntimeError(msg)
