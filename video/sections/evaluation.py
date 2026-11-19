"""Evaluation and the recap."""

from itertools import pairwise, starmap

from manimlib import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    DashedLine,
    FadeIn,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Line,
    RoundedRectangle,
    ShowCreation,
    Text,
    Triangle,
    VGroup,
)

from video import layout
from video.captions import RECALL_BARS
from video.layout import GREY, HIGHLIGHT, INK, SMALL_FONT, SectionScene


ARM_NAMES = {
    "none": "none",
    "bko": "bko\n(LLM)",
    "ctgan": "CTGAN",
    "bko-direct": "bko-direct\n(no LLM)",
    "real": "real",
}
"""Each recall bar's label."""


def _text(text: str, size: int = SMALL_FONT, color: str = INK) -> Text:
    return Text(text, font_size=size, fill_color=color)


class DoesItHelp(SectionScene):
    """Classifier results on UCI and on Diabetes 130's removed group."""

    name = "DoesItHelp"

    def _beat_1(self) -> None:
        axis = Line(LEFT * 2.0, RIGHT * 2.0, stroke_color=GREY).move_to((-4.4, 0.4, 0.0))
        zero = DashedLine(UP * 0.8, DOWN * 0.8, stroke_color=INK).move_to(axis)
        whisker = Line(LEFT * 0.9, RIGHT * 1.1, stroke_color=HIGHLIGHT, stroke_width=6)
        whisker.move_to(axis).shift(0.1 * RIGHT)
        ends = VGroup(*(Line(UP * 0.2, DOWN * 0.2, stroke_color=HIGHLIGHT) for _ in range(2)))
        ends[0].move_to(whisker.get_left())
        ends[1].move_to(whisker.get_right())
        heading = _text("UCI: bko minus none", 22).next_to(axis, UP, buff=1.0)
        labels = VGroup(
            _text("0", 16).next_to(zero, DOWN, buff=0.1),
            _text("worse", 16, GREY).next_to(axis, LEFT, buff=0.15),
            _text("better", 16, GREY).next_to(axis, RIGHT, buff=0.15),
        )
        self.play(FadeIn(heading), ShowCreation(axis), FadeIn(zero), FadeIn(labels))
        self.play(ShowCreation(whisker), FadeIn(ends), run_time=0.8)

    def _beat_2(self) -> None:
        self.heading = _text("Diabetes 130: men 60-79 readmitted < 30 days", 22)
        self.heading.move_to((3.0, 2.3, 0.0))
        removed = _text("removed from training", 18, GREY)
        self.removed = removed.next_to(self.heading, DOWN, buff=0.15)
        self.play(FadeIn(self.heading), FadeIn(removed), run_time=0.8)

    def _beat_3(self) -> None:
        values = {ARM_NAMES[arm]: float(figure) for arm, figure in RECALL_BARS.items()}
        shown = {ARM_NAMES[arm]: figure for arm, figure in RECALL_BARS.items()}
        self.chart = layout.bar_chart(values, unit=6.0, shown=shown).scale(0.85)
        self.chart.move_to((3.2, -0.6, 0.0))
        bars = self.chart[0]
        real = bars[-1]
        corners = [real.get_corner(DOWN + LEFT), real.get_corner(UP + LEFT)]
        corners += [real.get_corner(UP + RIGHT), real.get_corner(DOWN + RIGHT)]
        dashed = VGroup(*(DashedLine(a, b, stroke_color=INK) for a, b in pairwise(corners)))
        real.set_fill(opacity=0)
        self.play(
            *(GrowFromEdge(bar, DOWN) for bar in bars[:-1]),
            ShowCreation(dashed),
            FadeIn(self.chart[1]),
            run_time=1.2,
        )
        self.play(FadeIn(self.chart[2]), run_time=0.5)

    def _beat_4(self) -> None:
        names = list(RECALL_BARS)
        pair = [self.chart[0][names.index(arm)] for arm in ("ctgan", "bko-direct")]
        self.play(*(bar.animate.set_fill(HIGHLIGHT) for bar in pair), run_time=0.6)
        self.play(*(Indicate(bar, color=HIGHLIGHT) for bar in pair), run_time=1.0)

    def _beat_5(self) -> None:
        bar = self.chart[2][list(RECALL_BARS).index("bko-direct")]
        note = _text("no verdict", 16, GREY).next_to(bar, UP, buff=0.1)
        self.play(FadeIn(note, shift=0.1 * UP), run_time=0.5)

    def _beat_6(self) -> None:
        self.trees = _text("boosted trees: bko-direct 0.3591, real 0.5853", 18, GREY)
        self.trees.move_to((-4.4, -1.4, 0.0))
        self.play(FadeIn(self.trees), run_time=0.5)

    def _beat_7(self) -> None:
        kept = _text("19 real rows kept: level with ADASYN, CTGAN", 18, GREY)
        kept.next_to(self.trees, DOWN, buff=0.3)
        self.play(FadeIn(kept), run_time=0.5)


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
        loops = VGroup(
            _loop(self.boxes[4], self.boxes[1], "BKO arms", below=True),
            _loop(self.boxes[3], self.boxes[0], "Fuse", below=False),
        )
        self.play(LaggedStart(*(FadeIn(b) for b in self.boxes), lag_ratio=0.3), run_time=1.2)
        self.play(*(ShowCreation(a) for a in arrows), run_time=0.6)
        self.play(FadeIn(loops), run_time=0.8)

    def _beat_2(self) -> None:
        last = self.boxes[-1]
        self.play(Indicate(last, color=HIGHLIGHT), run_time=1.0)
        self.play(last[0].animate.set_stroke(HIGHLIGHT), run_time=0.4)
