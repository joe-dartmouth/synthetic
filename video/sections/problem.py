"""The problem: a thin cell of UCI Heart Disease."""

from manimlib import DOWN, LEFT, DashedLine, FadeIn, GrowFromEdge, Indicate, Text

from video import layout
from video.layout import GAP_RED, HIGHLIGHT, SMALL_FONT, SectionScene
from video.model import CELL, CELL_RECORDS, COUNT_GAP_BELOW


UNIT = 4.2 / max(CELL_RECORDS.values())
"""The height of one record."""


class ThinCells(SectionScene):
    """The cells of the training split and the count-based gap line."""

    name = "ThinCells"

    def _beat_1(self) -> None:
        self.chart = layout.bar_chart(CELL_RECORDS, unit=UNIT)
        self.chart.move_to((0.0, -0.1, 0.0))
        bars, names, numbers = self.chart
        self.play(*(GrowFromEdge(bar, DOWN) for bar in bars), run_time=1.2)
        self.play(FadeIn(names), FadeIn(numbers), run_time=0.6)

    def _beat_2(self) -> None:
        self.play(Indicate(self.chart[1], color=HIGHLIGHT), run_time=1.2)

    def _beat_3(self) -> None:
        index = list(CELL_RECORDS).index(CELL)
        bar = self.chart[0][index]
        group = [bar, self.chart[1][index], self.chart[2][index]]
        self.play(
            *(m.animate.set_color(HIGHLIGHT) for m in group),
            self.frame.animate.scale(0.45).move_to(bar.get_bottom()),
            run_time=1.2,
        )

    def _beat_4(self) -> None:
        bars = self.chart[0]
        y = bars.get_bottom()[1] + COUNT_GAP_BELOW * UNIT
        line = DashedLine(
            (bars.get_left()[0] - 0.4, y, 0.0),
            (bars.get_right()[0] + 0.4, y, 0.0),
            stroke_color=GAP_RED,
        )
        label = Text(str(COUNT_GAP_BELOW), font_size=SMALL_FONT, fill_color=GAP_RED)
        label.next_to(line, LEFT, buff=0.15)
        thin = [
            bars[i].animate.set_fill(GAP_RED)
            for i, count in enumerate(CELL_RECORDS.values())
            if count < COUNT_GAP_BELOW
        ]
        self.play(self.frame.animate.to_default_state(), run_time=0.8)
        self.play(FadeIn(line, shift=DOWN), FadeIn(label), run_time=0.6)
        self.play(*thin, run_time=0.6)
