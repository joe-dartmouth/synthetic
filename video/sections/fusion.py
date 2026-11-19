"""Fusion 1, what a BKO keeps, and grounding."""

from video.layout import SectionScene


class FusingFragments(SectionScene):
    """The ontology and data fragments fuse into the cell's BKO."""

    name = "FusingFragments"


class IncompletenessKept(SectionScene):
    """A statement the data never supports stays in the BKO."""

    name = "IncompletenessKept"


class GroundingAndGaps(SectionScene):
    """Instantiation from the held concepts, and the ungrounded statements it leaves."""

    name = "GroundingAndGaps"
