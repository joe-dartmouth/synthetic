"""A probe scene proving ManimGL animates and encodes a movie from `video/.venv`."""

from manimlib import (
    LEFT,
    RIGHT,
    Arrow,
    Circle,
    FadeIn,
    RoundedRectangle,
    Scene,
    ShowCreation,
    Square,
    Text,
    Transform,
)


class Probe(Scene):
    """Text, an arrow between two shapes, and a transform."""

    def construct(self) -> None:
        """Play the three animations."""
        circle = Circle().shift(3 * LEFT)
        box = RoundedRectangle().shift(3 * RIGHT)
        self.play(FadeIn(Text("BKO")))
        self.play(ShowCreation(Arrow(circle.get_right(), box.get_left())))
        self.play(Transform(circle, Square().move_to(circle)))
