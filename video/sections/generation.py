"""Filling the gap: the language-model path, bko-direct, and Fusion 2."""

from manimlib import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    ShowCreation,
    Text,
    Transform,
    VGroup,
    Write,
)

from video import layout
from video.captions import section
from video.layout import (
    DATA_ORANGE,
    GAP_RED,
    GREY,
    GROUNDED,
    HIGHLIGHT,
    INK,
    SMALL_FONT,
    SectionScene,
)
from video.model import CELL, CELL_RECORDS, COUNT_GAP_BELOW, LABELS, gaps, share


def _text(text: str, size: int = SMALL_FONT, color: str = INK) -> Text:
    return Text(text, font_size=size, fill_color=color)


def _box(text: str, *, width: float, color: str = INK) -> VGroup:
    label = _text(text, 20, color)
    frame = RoundedRectangle(width=width, height=0.5, corner_radius=0.1, stroke_color=color)
    return VGroup(frame, label.move_to(frame))


def _mini_record(color: str = INK) -> VGroup:
    """A small record icon: a card with three lines."""
    card = RoundedRectangle(width=0.4, height=0.5, corner_radius=0.05, stroke_color=color)
    lines = VGroup(*(Line(LEFT * 0.12, RIGHT * 0.12, stroke_color=color) for _ in range(3)))
    return VGroup(card, lines.arrange(DOWN, buff=0.09).move_to(card))


class LlmGuidedFill(SectionScene):
    """A prompt built from the cell's gaps, and validation of the records it returns."""

    name = "LlmGuidedFill"

    def _beat_1(self) -> None:
        heading = _text(f"{COUNT_GAP_BELOW} records for {CELL}", 22, HIGHLIGHT)
        self.card = RoundedRectangle(width=3.6, height=4.4, corner_radius=0.15, stroke_color=INK)
        self.card.move_to((-4.4, -0.1, 0.0))
        heading.next_to(self.card.get_top(), DOWN, buff=0.3)
        title = _text("prompt", 20, GREY).next_to(self.card, UP, buff=0.1)
        self.heading = heading
        self.play(ShowCreation(self.card), FadeIn(title), FadeIn(heading), run_time=1.0)

    def _beat_2(self) -> None:
        blocks = VGroup(
            *(
                _box(name, width=3.0, color=DATA_ORANGE)
                for name in ("cell", "ranges", "gaps", "is_a chains", "shares")
            )
        ).arrange(DOWN, buff=0.15)
        blocks.next_to(self.heading, DOWN, buff=0.3)
        self.play(
            LaggedStart(*(FadeIn(b, shift=0.5 * RIGHT) for b in blocks), lag_ratio=0.3),
            run_time=1.5,
        )

    def _beat_3(self) -> None:
        record = _mini_record().scale(1.6).next_to(self.card, RIGHT, buff=0.8).shift(1.2 * UP)
        mark = layout.cross(0.35).move_to(record)
        note = _text("patient records", 16, GREY).next_to(record, DOWN, buff=0.15)
        self.play(FadeIn(record), FadeIn(note), run_time=0.5)
        self.play(ShowCreation(mark), run_time=0.6)

    def _beat_4(self) -> None:
        model = _box("language model", width=2.6).move_to((0.4, -1.2, 0.0))
        gate = Rectangle(width=0.15, height=3.4, fill_color=HIGHLIGHT, fill_opacity=0.9)
        gate.set_stroke(width=0).move_to((3.0, -0.1, 0.0))
        gate_label = _text("validation", 16, HIGHLIGHT).next_to(gate, UP, buff=0.1)
        arrows = [layout.edge(self.card, model), layout.edge(model, gate)]
        self.play(FadeIn(model), *(ShowCreation(a) for a in arrows), run_time=0.8)
        self.play(FadeIn(gate), FadeIn(gate_label), run_time=0.4)
        records = VGroup(*(_mini_record(GROUNDED) for _ in range(COUNT_GAP_BELOW)))
        records.arrange_in_grid(5, 4, buff=0.15).move_to((5.2, -0.1, 0.0))
        start = model.get_center()
        self.play(
            LaggedStart(
                *(FadeIn(r, shift=r.get_center() - start) for r in records), lag_ratio=0.05
            ),
            run_time=1.5,
        )


class BkoDirect(SectionScene):
    """Each concept drawn at its share, fields copied from real records."""

    name = "BkoDirect"

    def _beat_1(self) -> None:
        texts = section(self.name).captions
        self.formula = _text(texts[0], 34).move_to((0.0, 2.0, 0.0))
        self.play(Write(self.formula), run_time=1.5)

    def _beat_2(self) -> None:
        texts = section(self.name).captions
        values = _text(texts[1], 28, HIGHLIGHT).next_to(self.formula, DOWN, buff=0.4)
        self.play(Write(values), run_time=1.5)

    def _beat_3(self) -> None:
        concept = "HP:0001717"
        value = share(concept)
        track = Rectangle(width=5.0, height=0.35, stroke_color=INK).move_to((-2.6, -0.5, 0.0))
        fill = Rectangle(width=5.0 * value, height=0.35, fill_color=GROUNDED, fill_opacity=0.9)
        fill.set_stroke(width=0).move_to(track, aligned_edge=LEFT)
        name = _text(LABELS[concept], 20).next_to(track, UP, buff=0.15, aligned_edge=LEFT)
        number = _text(f"{value:.3f}", 20).next_to(track, RIGHT, buff=0.2)
        self.play(FadeIn(track), FadeIn(name), run_time=0.4)
        self.play(GrowFromCenter(fill), FadeIn(number), run_time=1.0)

    def _beat_4(self) -> None:
        fields = ("cp", "restecg", "oldpeak, slope", "ca")
        donor = VGroup(*(_box(f, width=1.6, color=DATA_ORANGE) for f in fields))
        donor.arrange(DOWN, buff=0.12)
        new = VGroup(*(_box(f, width=1.6, color=GREY) for f in fields)).arrange(DOWN, buff=0.12)
        donor.move_to((2.6, -1.3, 0.0))
        new.move_to((5.4, -1.3, 0.0))
        labels = VGroup(
            _text("matching real record", 16, DATA_ORANGE).next_to(donor, UP, buff=0.12),
            _text("new record", 16, GREY).next_to(new, UP, buff=0.12),
        )
        self.play(FadeIn(donor), FadeIn(new), FadeIn(labels), run_time=0.6)
        copies = [
            Transform(target, source.copy().move_to(target))
            for source, target in zip(donor, new, strict=True)
        ]
        self.play(LaggedStart(*copies, lag_ratio=0.2), run_time=1.2)


class ClosingTheLoop(SectionScene):
    """Validated records rejoin the cell and detection runs again."""

    name = "ClosingTheLoop"

    def _beat_1(self) -> None:
        before = CELL_RECORDS[CELL]
        unit = 0.15
        bar = Rectangle(width=1.2, height=before * unit, fill_color=GAP_RED, fill_opacity=0.9)
        bar.set_stroke(width=0).move_to((-4.5, -2.4, 0.0), aligned_edge=DOWN)
        name = _text(CELL, 18).next_to(bar, DOWN, buff=0.15)
        count = _text(str(before), 22).next_to(bar, UP, buff=0.1)
        self.add(bar, name, count)
        cards = VGroup(*(Rectangle(width=1.2, height=unit) for _ in range(COUNT_GAP_BELOW)))
        for index, card in enumerate(cards):
            card.set_fill(GROUNDED, opacity=0.9).set_stroke(INK, width=1)
            card.move_to(bar.get_top() + (index + 0.5) * unit * UP)
        after = before + COUNT_GAP_BELOW
        total = _text(str(after), 22).move_to(cards.get_top() + 0.3 * UP)
        self.play(
            LaggedStart(*(FadeIn(c, shift=2.0 * DOWN) for c in cards), lag_ratio=0.08),
            Transform(count, total),
            run_time=1.8,
        )

    def _beat_2(self) -> None:
        rows = VGroup()
        for concept in ("HP:0001717", "HP:0100749"):
            mark = layout.tick()
            rows.add(VGroup(mark, _text(LABELS[concept], 22).next_to(mark, RIGHT, buff=0.2)))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to((-0.6, 1.6, 0.0))
        self.gap_rows = VGroup(
            *(_text(f"({CELL}, {c} -> {p})", SMALL_FONT, GAP_RED) for c, p in gaps())
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        self.gap_rows.move_to((3.6, -0.6, 0.0))
        heading = _text("gaps", 22).next_to(self.gap_rows, UP, buff=0.3, aligned_edge=LEFT)
        self.gap_rows.add(heading)
        self.add(self.gap_rows)
        self.play(LaggedStart(*(FadeIn(r, shift=0.3 * RIGHT) for r in rows)), run_time=1.0)

    def _beat_3(self) -> None:
        count = _text("per-subgroup gaps: 16 -> 0", 26, HIGHLIGHT)
        count.move_to(self.gap_rows, aligned_edge=LEFT)
        self.play(LaggedStart(*(FadeOut(r, shift=0.3 * RIGHT) for r in self.gap_rows)))
        self.play(FadeIn(count), run_time=0.5)
        stamp_text = _text("single draw", 30, GAP_RED)
        stamp = VGroup(
            RoundedRectangle(
                width=stamp_text.get_width() + 0.5,
                height=stamp_text.get_height() + 0.4,
                corner_radius=0.1,
                stroke_color=GAP_RED,
                stroke_width=5,
            ),
            stamp_text,
        )
        stamp.rotate(0.2).move_to((-0.6, -0.9, 0.0))
        self.play(GrowFromCenter(stamp), run_time=0.5)
