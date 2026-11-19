"""Fusion 1, what a BKO keeps, and grounding."""

from fractions import Fraction
from itertools import pairwise

from manimlib import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    Mobject,
    RoundedRectangle,
    ShowCreation,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
)

from bko.bkb.fusion import is_source
from video import layout
from video.captions import CELL_GAPS, NEIGHBOUR_GAPS
from video.layout import (
    GAP_RED,
    GREY,
    GROUNDED,
    HIGHLIGHT,
    INK,
    SMALL_FONT,
    SectionScene,
)
from video.model import CELL, CELL_RECORDS, COUNT_GAP_BELOW, fused, gaps, passes, source_priors


EKG = "HP:0003115"
CALCIFICATION_CHAIN = ("HP:0001717", "HP:0003207", "HP:0004934")


def _sources(
    statements: dict[tuple[str, str], VGroup], priors: dict[str, VGroup]
) -> list[tuple[Mobject, str, float]]:
    """Each fused S-node's box, its source's fragment name and that source's prior."""
    found: list[tuple[Mobject, str, float]] = []
    weights = source_priors()
    for snode in fused().snodes:
        if is_source(snode.head[0]):
            continue
        (source,) = [inode for inode in snode.tail if is_source(inode[0])]
        name = str(source[1])
        weight = weights[source[0], name]
        tail = [inode[0] for inode in snode.tail if not is_source(inode[0])]
        box = statements[tail[0], snode.head[0]][1] if tail else priors[snode.head[0]][0]
        found.append((box, "hpo" if name == "hpo" else "data", weight))
    return found


class FusingFragments(SectionScene):
    """The ontology and data fragments fuse into the cell's BKO."""

    name = "FusingFragments"

    def _beat_1(self) -> None:
        self.nodes, self.statements = layout.bkb_diagram()
        self.priors = layout.data_priors(self.nodes)
        self.add(*self.nodes.values(), *self.statements.values())
        loose = {}
        for row, concept in enumerate(layout.held_concepts()):
            node = layout.inode(concept, held=True).move_to((0.4, 2.3 - 1.0 * row, 0.0))
            loose[concept] = VGroup(node, layout.prior(node, str(self.priors[concept][0][1].text)))
        fragment = layout.tag("part of the cell's fragment")
        fragment.next_to(VGroup(*loose.values()), DOWN, buff=0.2)
        hpo = layout.tag("part of HPO").next_to(self.nodes["HP:0004934"], UP, buff=0.15)
        self.add(*loose.values(), fragment, hpo)
        merges = []
        for concept, group in loose.items():
            target = layout.inode(concept, held=True)
            merges += [Transform(group[0], target), Transform(group[1], self.priors[concept])]
        self.play(*merges, FadeOut(fragment), FadeOut(hpo), run_time=1.5)
        self.remove(*loose.values())
        filled = [self.nodes[c][0].set_fill(GROUNDED, opacity=0.8) for c in loose]
        self.add(*filled, *self.priors.values())

    def _beat_2(self) -> None:
        self.sources = []
        for box, name, weight in _sources(self.statements, self.priors):
            diamond = layout.source_node(name)
            side = LEFT if name == "data" else RIGHT
            diamond.next_to(box, side, buff=0.3)
            arrow = layout.edge(diamond[0], box)
            self.sources.append((VGroup(diamond, arrow), name, weight))
        self.play(LaggedStart(*(GrowFromCenter(s[0]) for s in self.sources)), run_time=1.5)

    def _beat_3(self) -> None:
        labels = [
            Text(
                f"P = {Fraction(weight).limit_denominator()}", font_size=16, fill_color=INK
            ).next_to(source[0][0], UP, buff=0.1)
            for source, _, weight in self.sources
        ]
        self.play(*(FadeIn(label) for label in labels), run_time=0.8)

    def _beat_4(self) -> None:
        both = [self.statements["HP:0012249", EKG][1], self.priors[EKG][0]]
        self.play(*(m.animate.set_color(HIGHLIGHT) for m in both), run_time=0.6)
        self.play(*(Indicate(m, color=HIGHLIGHT) for m in both), run_time=1.0)

    def _beat_5(self) -> None:
        row = VGroup()
        for name in ("mondo", "hpo", "data"):
            share = Text("1/3", font_size=16, fill_color=INK)
            row.add(VGroup(share, layout.source_node(name)).arrange(DOWN, buff=0.08))
        row.arrange(RIGHT, buff=0.6).move_to((-1.2, 2.25, 0.0))
        self.uci = row
        self.play(FadeIn(row, shift=0.2 * DOWN), run_time=0.8)

    def _beat_6(self) -> None:
        everything = VGroup(*self.nodes.values(), *self.statements.values(), *self.priors.values())
        frame = RoundedRectangle(
            width=everything.get_width() + 0.8,
            height=everything.get_height() + 0.5,
            corner_radius=0.2,
            stroke_color=HIGHLIGHT,
        ).move_to(everything)
        label = Text(f"BKO of {CELL}", font_size=24, fill_color=HIGHLIGHT)
        label.next_to(self.uci, DOWN, buff=0.25)
        self.play(ShowCreation(frame), FadeIn(label), run_time=1.2)


class IncompletenessKept(SectionScene):
    """A statement the data never supports stays in the BKO."""

    name = "IncompletenessKept"

    def _beat_1(self) -> None:
        nodes, statements, priors = layout.bko_diagram()
        self.add(*nodes.values(), *statements.values(), *priors.values())
        chain = [nodes[c] for c in CALCIFICATION_CHAIN]
        chain += [statements[e] for e in pairwise(CALCIFICATION_CHAIN)]
        self.chain = VGroup(*chain)
        self.formula = Text("BKO = ontology logic + BKB", font_size=26, fill_color=INK)
        self.formula.move_to((-0.4, 2.2, 0.0))
        self.play(FadeIn(self.formula, shift=0.2 * DOWN), run_time=0.8)

    def _beat_2(self) -> None:
        self.play(Indicate(self.chain, color=HIGHLIGHT, scale_factor=1.05), run_time=1.0)
        self.play(Indicate(self.chain, color=HIGHLIGHT, scale_factor=1.05), run_time=1.0)
        self.play(self.chain.animate.set_color(GREY), run_time=0.5)

    def _beat_3(self) -> None:
        note = Text("removed or kept: every probability the same", font_size=SMALL_FONT)
        note.set_fill(GREY).next_to(self.formula, DOWN, buff=0.3)
        self.play(FadeIn(note), FadeOut(self.chain), run_time=1.0)
        self.play(FadeIn(self.chain), run_time=1.0)


class GroundingAndGaps(SectionScene):
    """Instantiation from the held concepts, and the ungrounded statements it leaves."""

    name = "GroundingAndGaps"

    def _beat_1(self) -> None:
        self.nodes, self.statements, self.priors = layout.bko_diagram()
        for concept in layout.held_concepts():
            self.nodes[concept][0].set_fill(opacity=0.0)
        self.add(*self.nodes.values(), *self.statements.values(), *self.priors.values())
        anchors = [self.nodes[c][0] for c in layout.held_concepts()]
        self.play(*(a.animate.set_fill(GROUNDED, opacity=0.9) for a in anchors), run_time=0.6)
        self.play(*(Indicate(a, color=GROUNDED) for a in anchors), run_time=1.0)

    def _light(self, step: int) -> None:
        lit = []
        for edge in passes()[step]:
            lit += [
                self.statements[edge].animate.set_color(GROUNDED),
                self.nodes[edge[1]][0].animate.set_fill(GROUNDED, opacity=0.9),
            ]
        self.play(*lit, run_time=1.0)

    def _beat_2(self) -> None:
        self._light(0)

    def _beat_3(self) -> None:
        self._light(1)

    def _beat_4(self) -> None:
        self.play(*(self.statements[e].animate.set_color(GAP_RED) for e in gaps()), run_time=1.0)

    def _beat_5(self) -> None:
        diagram = VGroup(*self.nodes.values(), *self.statements.values(), *self.priors.values())
        self.play(diagram.animate.scale(0.62).to_edge(LEFT, buff=0.3), run_time=1.0)
        self.heading = Text(f"gaps of {CELL}", font_size=24, fill_color=INK)
        self.rows = VGroup(
            *(
                Text(f"({CELL}, {child} -> {parent})", font_size=SMALL_FONT, fill_color=GAP_RED)
                for child, parent in gaps()
            )
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        listing = VGroup(self.heading, self.rows).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        listing.move_to((3.9, 1.3, 0.0))
        copies = [
            TransformFromCopy(self.statements[e][1], r)
            for e, r in zip(gaps(), self.rows, strict=True)
        ]
        self.play(FadeIn(self.heading), *copies, run_time=1.2)

    def _beat_6(self) -> None:
        count = Text(f"{CELL_GAPS} gaps, {len(gaps())} drawn", font_size=24, fill_color=HIGHLIGHT)
        count.move_to(self.heading, aligned_edge=LEFT)
        more = Text(f"+ {CELL_GAPS - len(gaps())} more", font_size=SMALL_FONT, fill_color=GREY)
        more.next_to(self.rows, DOWN, aligned_edge=LEFT, buff=0.2)
        self.play(Transform(self.heading, count), FadeIn(more), run_time=0.8)

    def _beat_7(self) -> None:
        neighbour = "<55/female/0"
        chart = layout.bar_chart({neighbour: CELL_RECORDS[neighbour]}, unit=0.06)
        chart.move_to((3.0, -1.6, 0.0))
        bar = chart[0][0]
        y = bar.get_bottom()[1] + COUNT_GAP_BELOW * 0.06
        line = DashedLine(
            (bar.get_left()[0] - 0.3, y, 0.0), (bar.get_right()[0] + 0.3, y, 0.0)
        ).set_stroke(GAP_RED)
        note = Text(f"{NEIGHBOUR_GAPS} BKO gaps", font_size=24, fill_color=GAP_RED).next_to(
            bar, RIGHT, buff=0.6
        )
        self.play(FadeIn(chart, shift=0.2 * UP), FadeIn(line), run_time=0.8)
        self.play(FadeIn(note), run_time=0.5)
