"""The BKB vocabulary: the ontology, I-nodes and S-nodes, and the cell's data fragment."""

from manimlib import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    ReplacementTransform,
    ShowCreation,
    Text,
    Transform,
    VGroup,
)

from video import layout
from video.layout import GREY, HIGHLIGHT, SMALL_FONT, SectionScene
from video.model import CELL, CELL_RECORDS, EDGES, HOLDERS, LABELS


FIRST_EDGE = EDGES[0]


class Ontology(SectionScene):
    """The eight HPO classes as three is_a chains."""

    name = "Ontology"

    def _beat_1(self) -> None:
        nodes, self.arrows = layout.ontology_diagram()
        chains = [
            [nodes[c] for c in layout.CHAIN_LAYOUT if layout.CHAIN_LAYOUT[c][0] == x]
            for x in sorted({x for x, _ in layout.CHAIN_LAYOUT.values()})
        ]
        self.play(LaggedStart(*(FadeIn(VGroup(*chain)) for chain in chains)), run_time=1.2)
        self.play(*(ShowCreation(arrow) for arrow in self.arrows.values()), run_time=0.8)
        hpo = layout.tag("part of HPO").move_to((0.2, 2.1, 0.0))
        self.play(FadeIn(hpo), run_time=0.4)

    def _beat_2(self) -> None:
        labels = [
            Text("is_a", font_size=SMALL_FONT, fill_color=HIGHLIGHT).next_to(arrow, LEFT, 0.15)
            for arrow in self.arrows.values()
        ]
        self.play(*(FadeIn(label, shift=0.1 * UP) for label in labels), run_time=0.6)
        self.play(*(Indicate(arrow, color=HIGHLIGHT) for arrow in self.arrows.values()))


class INodesAndSNodes(SectionScene):
    """Each class becomes an I-node and each is_a edge an S-node."""

    name = "INodesAndSNodes"

    def _beat_1(self) -> None:
        self.nodes, self.arrows = layout.ontology_diagram()
        self.add(*self.nodes.values(), *self.arrows.values())
        child = FIRST_EDGE[0]
        self.play(self.frame.animate.scale(0.5).move_to(layout.point(child)), run_time=1.0)
        self.play(Transform(self.nodes[child], layout.inode(child, held=False)), run_time=1.0)

    def _beat_2(self) -> None:
        inodes, _ = layout.bkb_diagram()
        rest = [Transform(self.nodes[c], inodes[c]) for c in LABELS if c != FIRST_EDGE[0]]
        arrows = [
            Transform(self.arrows[c, p], layout.edge(inodes[c][0], inodes[p][0])) for c, p in EDGES
        ]
        self.play(self.frame.animate.to_default_state(), *rest, *arrows, run_time=1.5)

    def _beat_3(self) -> None:
        child, parent = FIRST_EDGE
        made = layout.statement(self.nodes[child], self.nodes[parent])
        tail = Text("tail", font_size=SMALL_FONT, fill_color=HIGHLIGHT)
        head = Text("head", font_size=SMALL_FONT, fill_color=HIGHLIGHT)
        tail.next_to(self.nodes[child][0], LEFT, buff=0.15)
        head.next_to(self.nodes[parent][0], LEFT, buff=0.15)
        self.play(ReplacementTransform(self.arrows[FIRST_EDGE], made), run_time=1.0)
        self.play(FadeIn(tail), FadeIn(head), run_time=0.6)

    def _beat_4(self) -> None:
        morphs = [
            ReplacementTransform(self.arrows[c, p], layout.statement(self.nodes[c], self.nodes[p]))
            for c, p in EDGES[1:]
        ]
        self.play(*morphs, run_time=1.5)

    def _beat_5(self) -> None:
        target = self.nodes["HP:0003115"][0]
        prior = layout.snode("p", color=HIGHLIGHT).next_to(target, LEFT, buff=1.0)
        note = Text("no tail", font_size=SMALL_FONT, fill_color=GREY)
        note.next_to(prior, LEFT, buff=0.15)
        arrow = layout.edge(prior, target)
        self.play(FadeIn(VGroup(prior, arrow, note), shift=0.2 * RIGHT), run_time=1.0)


class DatasetFragment(SectionScene):
    """The cell's records become priors on the concepts they show."""

    name = "DatasetFragment"

    def _beat_1(self) -> None:
        self.nodes, statements = layout.bkb_diagram()
        self.priors = layout.data_priors(self.nodes)
        diagram = VGroup(*self.nodes.values(), *statements.values(), *self.priors.values())
        diagram.scale(0.72).to_edge(LEFT, buff=0.3)
        self.add(*self.nodes.values(), *statements.values())
        n = CELL_RECORDS[CELL]
        cards = VGroup(
            *(
                layout.record_card(
                    [(LABELS[c], record >= n - count) for c, count in HOLDERS.items()],
                    title=f"record {record + 1} of {CELL}",
                )
                for record in range(n)
            )
        ).arrange(DOWN, buff=0.35)
        cards.move_to((4.7, 0.0, 0.0))
        self.play(LaggedStart(*(FadeIn(card, shift=0.3 * LEFT) for card in cards)), run_time=1.2)

    def _show_prior(self, concept: str) -> None:
        node = self.nodes[concept]
        self.play(
            GrowFromCenter(self.priors[concept]),
            node[0].animate.set_fill(layout.GROUNDED, opacity=0.8),
            run_time=1.0,
        )

    def _beat_2(self) -> None:
        self._show_prior("HP:0012250")

    def _beat_3(self) -> None:
        self._show_prior("HP:0003115")

    def _beat_4(self) -> None:
        never = [c for c, count in HOLDERS.items() if count == 0]
        fragment = layout.tag("part of the cell's fragment")
        fragment.next_to(self.priors["HP:0012250"], DOWN, buff=0.35)
        self.play(*(self.nodes[c].animate.set_color(GREY) for c in never), run_time=0.8)
        self.play(FadeIn(fragment), run_time=0.4)
