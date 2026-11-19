"""The scene frame every section shares: a title, one caption per beat, and the beat hook.

`SectionScene` reads its title and captions from `video.captions` and times each beat with
`hold_seconds`, so a section draws its diagram in `beat` and never sets its own timing.
"""

from typing import ClassVar

from manimlib import DOWN, UP, FadeIn, FadeOut, Scene, Text

from video.captions import FADE_SECONDS, SCENE_END_SECONDS, hold_seconds, section


class SectionScene(Scene):
    """One section of the video: its title throughout, then each caption with its beat.

    Attributes:
        name: The section's Scene class name in `video.captions.SECTIONS`.
    """

    name: ClassVar[str]

    def construct(self) -> None:
        """Show the title, then play each caption's fade, beat and hold, and end on the diagram."""
        this = section(self.name)
        self.add(Text(this.title, font_size=40).to_edge(UP))
        shown: Text | None = None
        for index, caption in enumerate(this.captions):
            label = Text(caption, font_size=32, line_width=12.0).to_edge(DOWN)
            fades = [FadeIn(label)] if shown is None else [FadeOut(shown), FadeIn(label)]
            self.play(*fades, run_time=FADE_SECONDS)
            shown = label
            start = self.time
            self.beat(index)
            remaining = hold_seconds(caption) - (self.time - start)
            if remaining > 0:
                self.wait(remaining)
        self.wait(SCENE_END_SECONDS)

    def beat(self, index: int) -> None:
        """Draw the beat of caption `index`; a section overrides it, and the default draws nothing.

        Args:
            index: The caption's position in the section.
        """
