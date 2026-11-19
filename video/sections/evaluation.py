"""Evaluation and the recap."""

from video.layout import SectionScene


class DoesItHelp(SectionScene):
    """Classifier results on UCI and on Diabetes 130's removed group."""

    name = "DoesItHelp"


class Recap(SectionScene):
    """The pipeline's loop."""

    name = "Recap"
