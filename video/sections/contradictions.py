"""Contradictions between the data and the ontology, and how each stage handles them."""

from itertools import pairwise, starmap

from manimlib import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line,
    Rectangle,
    RoundedRectangle,
    ShowCreation,
    Text,
    VGroup,
)

from video import layout
from video.captions import DIABETES_CHAIN, KINDS, UCI_CHAIN
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
from video.model import CELL, CELL_RECORDS, HOLDERS, fused, share


ST, ABNORMAL_ST, EKG = "HP:0012250", "HP:0012249", "HP:0003115"


def _text(text: str, size: int = SMALL_FONT, color: str = INK) -> Text:
    return Text(text, font_size=size, fill_color=color)


def _share_bar(label: str, part: int, whole: int) -> VGroup:
    """A bar of `whole` records with the `part` that contradict filled red, and its label."""
    outline = Rectangle(width=4.0, height=0.4, stroke_color=INK)
    filled = Rectangle(width=4.0 * part / whole, height=0.4, fill_color=GAP_RED, fill_opacity=0.9)
    filled.set_stroke(width=0).move_to(outline, aligned_edge=LEFT)
    caption = _text(f"{label}: {part} of {whole}", 22).next_to(outline, UP, buff=0.15)
    return VGroup(outline, filled, caption)


def _mini_node(text: str, *, fill: str | None = None) -> VGroup:
    circle = Circle(radius=0.18, stroke_color=INK, stroke_width=3)
    if fill is not None:
        circle.set_fill(fill, opacity=0.9)
    return VGroup(circle, _text(text, 14).next_to(circle, RIGHT, buff=0.1))


class WhatAContradictionIs(SectionScene):
    """A record showing a child concept without its parent."""

    name = "WhatAContradictionIs"

    def _beat_1(self) -> None:
        nodes, statements = layout.bkb_diagram()
        self.nodes = {c: nodes[c] for c in (ST, ABNORMAL_ST, EKG)}
        self.chain = VGroup(*self.nodes.values(), statements[ST, ABNORMAL_ST])
        self.chain.add(statements[ABNORMAL_ST, EKG])
        self.chain.shift(9.0 * LEFT)
        self.play(FadeIn(self.chain), run_time=0.6)
        ends = [self.nodes[c][0] for c in (ST, EKG)]
        self.play(*(e.animate.set_stroke(HIGHLIGHT) for e in ends), run_time=0.6)
        self.top = statements[ABNORMAL_ST, EKG]

    def _beat_2(self) -> None:
        n = CELL_RECORDS[CELL]
        rows = [("Abnormal EKG", False), ("ST segment depression", n - HOLDERS[ST] <= 0)]
        self.card = layout.record_card(rows, title=f"record 1 of {CELL}").move_to((0.6, 1.2, 0))
        mark = layout.cross(0.3).move_to(self.top[1])
        self.play(FadeIn(self.card, shift=0.3 * LEFT), run_time=0.8)
        self.play(ShowCreation(mark), run_time=0.6)

    def _beat_3(self) -> None:
        rows = self.card[1][1]
        x = self.card.get_left()[0]
        self.links = VGroup(
            *(
                layout.edge(Dot((x, row.get_y(), 0.0), radius=0.01), self.nodes[c][1])
                for row, c in zip(rows, (EKG, ST), strict=True)
            )
        )
        self.links.set_fill(DATA_ORANGE)
        self.play(*(ShowCreation(link) for link in self.links), run_time=1.0)

    def _beat_4(self) -> None:
        rows = self.card[1][1]
        fields = VGroup(
            _text("restecg 1-2  (resting EKG)", color=DATA_ORANGE),
            _text("oldpeak >= 1, slope 2-3  (exercise test)", color=DATA_ORANGE),
        )
        fields[0].next_to(self.card, RIGHT, buff=0.3).match_y(rows[0])
        fields[1].next_to(self.card, RIGHT, buff=0.3).match_y(rows[1])
        self.play(FadeIn(fields, shift=0.2 * LEFT), run_time=0.8)

    def _beat_5(self) -> None:
        self.uci = _share_bar("UCI", *UCI_CHAIN).move_to((1.6, -1.2, 0.0))
        self.play(FadeIn(self.uci[0]), FadeIn(self.uci[2]), GrowFromCenter(self.uci[1]))

    def _beat_6(self) -> None:
        diabetes = _share_bar("Diabetes 130", *DIABETES_CHAIN).next_to(self.uci, DOWN, buff=0.4)
        self.play(FadeIn(diabetes[0]), FadeIn(diabetes[2]), GrowFromCenter(diabetes[1]))


def _kind_pictures() -> list[VGroup]:
    """One small picture per kind, in `KINDS`'s order."""
    narrow = layout.record_card([("ST segment depression", True), ("Abnormal EKG", False)])
    ratio = layout.bar_chart(
        {"ST depression": HOLDERS[ST] / 2, "Abnormal EKG": HOLDERS[EKG] / 2}, unit=2.0
    )
    empty = Rectangle(width=1.0, height=2.0, stroke_color=GREY)
    empty_cell = VGroup(empty, _text("0 records", color=GREY).next_to(empty, DOWN))
    all_show = VGroup(
        *(
            layout.record_card([("ST segment depression", True)], title=f"record {i}")
            for i in (1, 2)
        )
    ).arrange(DOWN, buff=0.25)
    code = _text("cp = 1", 24, DATA_ORANGE)
    targets = VGroup(_text("Angina pectoris", 22), _text("Chest pain", 22))
    targets.arrange(DOWN, aligned_edge=LEFT, buff=0.6).next_to(code, RIGHT, buff=1.2)
    overlap = VGroup(code, targets, *(layout.edge(code, t) for t in targets))
    return [narrow, ratio, empty_cell, all_show, overlap]


class FiveKinds(SectionScene):
    """The five kinds the contradictions census lists."""

    name = "FiveKinds"

    def _beat_1(self) -> None:
        self.rows = {}
        for kind, gloss in KINDS:
            name = _text(kind, 26, INK)
            note = _text(gloss, SMALL_FONT, GREY)
            self.rows[kind] = VGroup(name, note).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        listing = VGroup(*self.rows.values()).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        listing.to_edge(LEFT, buff=0.8).shift(0.2 * UP)
        self.pictures = dict(zip((k for k, _ in KINDS), _kind_pictures(), strict=True))
        for picture in self.pictures.values():
            picture.move_to((4.2, 0.2, 0.0))
        self.shown = None
        self.play(FadeIn(listing), run_time=0.8)
        self._light("narrow-without-broad")

    def _light(self, kind: str) -> None:
        animations = [self.rows[kind][0].animate.set_fill(HIGHLIGHT)]
        if self.shown is not None:
            animations += [self.rows[self.shown][0].animate.set_fill(INK)]
            animations += [FadeOut(self.pictures[self.shown])]
        self.play(*animations, FadeIn(self.pictures[kind]), run_time=0.8)
        self.shown = kind

    def _beat_2(self) -> None:
        self._light("child-share-above-parent")

    def _beat_3(self) -> None:
        self._light("all-show")

    def _beat_4(self) -> None:
        self._light("code-overlap")

    def _beat_5(self) -> None:
        self._light("empty-cell")


STAGES = ("Fuse", "Detect", "Shares", "Sampler", "Census")


def _fuse_picture() -> VGroup:
    """Abnormal EKG with both S-nodes the fusion keeps, from `video.model.fused`."""
    node = _mini_node("Abnormal EKG")
    boxes = VGroup(
        *(layout.snode(f"{s.prob:g}") for s in fused().snodes if s.head == (EKG, "a"))
    ).arrange(RIGHT, buff=0.3)
    boxes.next_to(node, DOWN, buff=0.7).match_x(node[0])
    return VGroup(node, boxes, *(layout.edge(box, node[0]) for box in boxes))


def _detect_picture() -> VGroup:
    child = _mini_node("held child", fill=GROUNDED)
    parent = _mini_node("parent", fill=GROUNDED).next_to(child, UP, buff=0.8)
    return VGroup(child, parent, layout.edge(child[0], parent[0]))


def _shares_picture() -> VGroup:
    line = Line(LEFT * 1.0, RIGHT * 1.0, stroke_color=INK)
    ends = VGroup(_text("0").next_to(line, DOWN), _text("1").next_to(line, DOWN))
    ends[0].match_x(line.get_left())
    ends[1].match_x(line.get_right())
    value = share("HP:0001717")
    dot = Dot(line.point_from_proportion(value), radius=0.1, fill_color=HIGHLIGHT)
    return VGroup(line, ends, dot, _text("share", 16, GREY).next_to(line, UP, buff=0.3))


def _sampler_picture() -> VGroup:
    child = _mini_node("drawn", fill=GROUNDED)
    parent = _mini_node("not forced").next_to(child, UP, buff=0.8)
    parent.set_color(GREY)
    return VGroup(child, parent, layout.edge(child[0], parent[0]).set_fill(GREY))


def _census_picture() -> VGroup:
    lines = VGroup(*(_text(kind, 13) for kind, _ in KINDS))
    lines.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    sheet = RoundedRectangle(
        width=lines.get_width() + 0.4, height=lines.get_height() + 0.4, corner_radius=0.08
    )
    return VGroup(sheet.set_stroke(INK), lines.move_to(sheet))


class HandlingContradictions(SectionScene):
    """Each stage reports a contradiction and never forces the data."""

    name = "HandlingContradictions"

    def _beat_1(self) -> None:
        self.boxes = VGroup()
        for stage in STAGES:
            label = _text(stage, 24)
            box = RoundedRectangle(width=2.3, height=0.6, corner_radius=0.12, stroke_color=GREY)
            self.boxes.add(VGroup(box, label.move_to(box)))
        self.boxes.arrange(RIGHT, buff=0.35).move_to((0.0, 2.2, 0.0))
        arrows = list(starmap(layout.edge, pairwise(self.boxes)))
        self.play(FadeIn(self.boxes), *(FadeIn(a) for a in arrows), run_time=0.8)
        self._show(0, _fuse_picture())

    def _show(self, stage: int, picture: VGroup) -> None:
        box = self.boxes[stage]
        picture.next_to(box, DOWN, buff=0.6).match_x(box)
        self.play(box[0].animate.set_stroke(HIGHLIGHT), FadeIn(picture), run_time=0.8)
        self.last = picture

    def _beat_2(self) -> None:
        self._show(1, _detect_picture())

    def _beat_3(self) -> None:
        self._show(2, _shares_picture())

    def _beat_4(self) -> None:
        self._show(3, _sampler_picture())

    def _beat_5(self) -> None:
        note = _text("no match: copy from\nthe label group", 16, DATA_ORANGE)
        note.next_to(self.last, DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.6)

    def _beat_6(self) -> None:
        self._show(4, _census_picture())

    def _beat_7(self) -> None:
        note = _text("Diabetes 130:\n757 of 838 rewritten", 16, GAP_RED)
        note.next_to(self.last, DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.6)
