"""Filling the gap: the language-model path, bko-direct, and Fusion 2."""

from video.layout import SectionScene


class LlmGuidedFill(SectionScene):
    """A prompt built from the cell's gaps, and validation of the records it returns."""

    name = "LlmGuidedFill"


class BkoDirect(SectionScene):
    """Each concept drawn at its share, fields copied from real records."""

    name = "BkoDirect"


class ClosingTheLoop(SectionScene):
    """Validated records rejoin the cell and detection runs again."""

    name = "ClosingTheLoop"
