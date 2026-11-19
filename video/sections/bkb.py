"""The BKB vocabulary: the ontology, I-nodes and S-nodes, and the cell's data fragment."""

from manimlib import LEFT, RIGHT, FadeIn, ReplacementTransform, Text, Transform, VGroup

from video import layout
from video.layout import GREY, HIGHLIGHT, SMALL_FONT, SectionScene
from video.model import EDGES, LABELS


FIRST_EDGE = EDGES[0]


class Ontology(SectionScene):
    """The eight HPO classes as three is_a chains."""

    name = "Ontology"


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
