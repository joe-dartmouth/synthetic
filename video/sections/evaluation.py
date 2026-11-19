"""How the system is evaluated, and the recap."""

from itertools import pairwise, starmap

from manimlib import (
    DOWN,
    PI,
    RIGHT,
    UP,
    DashedLine,
    FadeIn,
    GrowFromCenter,
    LaggedStart,
    RoundedRectangle,
    ShowCreation,
    Text,
    Triangle,
    VGroup,
)

from video import layout
from video.layout import DATA_ORANGE, GREY, GROUNDED, HIGHLIGHT, INK, SMALL_FONT, SectionScene
from video.model import CELL


def _text(text: str, size: int = SMALL_FONT, color: str = INK) -> Text:
    return Text(text, font_size=size, fill_color=color)


class HowWeEvaluate(SectionScene):
    """The design: held-out real records, two classifiers and the thin group's recall."""

    name = "HowWeEvaluate"

    def _beat_1(self) -> None:
        self.train = layout.box("training records (real)", width=3.0).move_to((-5.0, 1.8, 0.0))
        self.test = layout.box("test records (held out)", width=3.0).move_to((5.0, 1.8, 0))
        self.play(FadeIn(self.train), FadeIn(self.test), run_time=0.8)

    def _beat_2(self) -> None:
        self.new = layout.box("new records", width=2.2, color=GROUNDED).move_to((-1.4, 1.8, 0))
        self.only = layout.box("classifier: real only", width=3.0).move_to((-5.0, -0.1, 0.0))
        self.plus = layout.box("classifier: real + new", width=3.2).move_to((-1.4, -0.1, 0.0))
        arrows = [
            layout.edge(self.train, self.only),
            layout.edge(self.train, self.plus),
            layout.edge(self.new, self.plus),
        ]
        self.play(FadeIn(self.new), FadeIn(self.only), FadeIn(self.plus), run_time=0.8)
        self.play(*(ShowCreation(arrow) for arrow in arrows), run_time=0.6)

    def _beat_3(self) -> None:
        score = layout.box(f"recall on {CELL}", width=3.2, color=HIGHLIGHT)
        score.move_to((4.6, -1.8, 0.0))
        arrows = [layout.edge(source, score) for source in (self.only, self.plus, self.test)]
        self.play(FadeIn(score), *(ShowCreation(arrow) for arrow in arrows), run_time=1.0)

    def _beat_4(self) -> None:
        names = ("SMOTE", "CTGAN", "other generators")
        others = VGroup(*(layout.box(n, width=2.0, color=DATA_ORANGE) for n in names))
        others.arrange(DOWN, buff=0.12).move_to((1.6, 2.05, 0.0), aligned_edge=UP)
        heading = _text("or, in its place:", 18, DATA_ORANGE).next_to(others, UP, buff=0.1)
        dashed = DashedLine(others[0].get_left(), self.new.get_right(), stroke_color=DATA_ORANGE)
        self.play(
            LaggedStart(*(GrowFromCenter(o) for o in others), lag_ratio=0.2),
            FadeIn(heading),
            run_time=1.0,
        )
        self.play(ShowCreation(dashed), run_time=0.5)


def _loop(start: VGroup, end: VGroup, label: str, *, below: bool) -> VGroup:
    """A dotted edge from `start` back to `end`, below or above the boxes, labelled."""
    side = DOWN if below else UP
    a = start.get_edge_center(side) + 0.1 * side
    b = end.get_edge_center(side) + 0.1 * side
    dotted = DashedLine(a, b, path_arc=-1.0 if below else 1.0, stroke_color=GREY)
    tip = Triangle(fill_color=GREY, fill_opacity=1, stroke_width=0).scale(0.1)
    tip.rotate(0.0 if below else PI).move_to(b)
    name = _text(label, 18, GREY).next_to(dotted, side, buff=0.1)
    return VGroup(dotted, tip, name)


STAGES = ("Represent", "Detect", "Generate", "Validate", "Evaluate")


class Recap(SectionScene):
    """The pipeline's loop."""

    name = "Recap"

    def _beat_1(self) -> None:
        self.boxes = VGroup()
        for stage in STAGES:
            label = _text(stage, 26)
            frame = RoundedRectangle(width=2.3, height=0.8, corner_radius=0.15, stroke_color=INK)
            self.boxes.add(VGroup(frame, label.move_to(frame)))
        self.boxes.arrange(RIGHT, buff=0.5).move_to((0.0, 0.2, 0.0))
        arrows = list(starmap(layout.edge, pairwise(self.boxes)))
        self.loops = VGroup(
            _loop(self.boxes[4], self.boxes[1], "BKO arms", below=True),
            _loop(self.boxes[3], self.boxes[0], "Fuse", below=False),
        )
        self.play(LaggedStart(*(FadeIn(b) for b in self.boxes), lag_ratio=0.3), run_time=1.2)
        self.play(*(ShowCreation(a) for a in arrows), run_time=0.6)
        self.play(FadeIn(self.loops), run_time=0.8)

    def _beat_2(self) -> None:
        fuse = self.loops[1]
        self.play(fuse.animate.set_color(HIGHLIGHT), run_time=0.8)
