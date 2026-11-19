"""Contradictions between the data and the ontology, and how each stage handles them."""

from video.layout import SectionScene


class WhatAContradictionIs(SectionScene):
    """A record showing a child concept without its parent."""

    name = "WhatAContradictionIs"


class FiveKinds(SectionScene):
    """The five kinds the contradictions census lists."""

    name = "FiveKinds"


class HandlingContradictions(SectionScene):
    """Each stage reports a contradiction and never forces the data."""

    name = "HandlingContradictions"
