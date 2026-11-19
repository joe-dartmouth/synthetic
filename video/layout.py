"""The scene frame every section shares, and the diagram pieces more than one section draws.

`SectionScene` reads its title and beats from `video.captions` and times each beat with
`hold_seconds`; a section draws beat n's diagram in a `_beat_<n>` method and never sets its own
timing. The pieces below read the running example from `video.model`: `CHAIN_LAYOUT` places the
eight classes as three `is_a` chains, narrow at the bottom and broad at the top.
"""

import textwrap
from collections.abc import Collection, Mapping, Sequence
from typing import ClassVar, Final

from manimlib import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Mobject,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
    VMobject,
)

from video.captions import FADE_SECONDS, SCENE_END_SECONDS, hold_seconds, section
from video.model import EDGES, HOLDERS, LABELS, STATE, dataset_fragment


INK: Final = "#ECECEC"
GROUNDED: Final = "#5CB85C"
GAP_RED: Final = "#E0524D"
GREY: Final = "#7A7A7A"
HIGHLIGHT: Final = "#F0C419"
HPO_BLUE: Final = "#4A90D9"
DATA_ORANGE: Final = "#E8954A"
MONDO_PURPLE: Final = "#A57FD8"
SOURCE_COLORS: Final[Mapping[str, str]] = {"hpo": HPO_BLUE, "mondo": MONDO_PURPLE}
"""Each ontology source's color; a dataset source is orange."""

CHAIN_LAYOUT: Final[Mapping[str, tuple[float, float]]] = {
    "HP:0001717": (-5.3, -2.1),
    "HP:0003207": (-5.3, 0.0),
    "HP:0004934": (-5.3, 2.1),
    "HP:0100749": (-1.2, -2.1),
    "HP:0012531": (-1.2, 0.0),
    "HP:0012250": (3.7, -2.1),
    "HP:0012249": (3.7, 0.0),
    "HP:0003115": (3.7, 2.1),
}
"""Each class's position: one column per chain."""

LABEL_FONT: Final = 20
SMALL_FONT: Final = 18


def point(concept: str) -> tuple[float, float, float]:
    """Where `concept` sits on screen."""
    x, y = CHAIN_LAYOUT[concept]
    return (x, y, 0.0)


def _label(concept: str, second_line: str) -> Text:
    """The wrapped label of `concept` over `second_line`, in a left-aligned block."""
    lines = [*textwrap.wrap(LABELS[concept], 18), second_line]
    return Text("\n".join(lines), font_size=LABEL_FONT, alignment="LEFT", fill_color=INK)


def class_node(concept: str) -> VGroup:
    """An ontology class: a dot and its label and id."""
    dot = Dot(point(concept), radius=0.1, fill_color=INK)
    return VGroup(dot, _label(concept, concept).next_to(dot, RIGHT, buff=0.35))


def inode(concept: str, *, held: bool) -> VGroup:
    """An I-node: a circle, filled when the cell's records show it, its label and (id, state)."""
    circle = Circle(radius=0.25, stroke_color=INK, stroke_width=3).move_to(point(concept))
    if held:
        circle.set_fill(GROUNDED, opacity=0.8)
    label = _label(concept, f"({concept}, {STATE})").next_to(circle, RIGHT, buff=0.2)
    return VGroup(circle, label)


def snode(probability: str, *, color: str = INK) -> VGroup:
    """An S-node: a small rounded box holding its probability."""
    text = Text(f"P = {probability}", font_size=SMALL_FONT, fill_color=color)
    box = RoundedRectangle(
        width=text.get_width() + 0.3, height=0.42, corner_radius=0.1, stroke_color=color
    )
    return VGroup(box, text.move_to(box))


def edge(start: Mobject, end: Mobject) -> Arrow:
    """An arrow from `start`'s boundary to `end`'s."""
    return Arrow(start, end, buff=0.08, thickness=2.5, fill_color=INK)


def source_node(name: str) -> VGroup:
    """A source I-node: a small diamond tagged with its fragment's name."""
    color = SOURCE_COLORS.get(name, DATA_ORANGE)
    diamond = Square(side_length=0.26, fill_color=color, fill_opacity=1, stroke_width=0)
    diamond.rotate(0.785398)
    tag = Text(name, font_size=14, fill_color=color).next_to(diamond, DOWN, buff=0.06)
    return VGroup(diamond, tag)


def tick(color: str = GROUNDED) -> VMobject:
    """A check mark."""
    mark = VMobject(stroke_color=color, stroke_width=6)
    mark.set_points_as_corners([(-0.12, 0.0, 0.0), (-0.03, -0.1, 0.0), (0.14, 0.12, 0.0)])
    return mark


def cross(size: float = 0.14, color: str = GAP_RED) -> VGroup:
    """An X."""
    return VGroup(
        Line((-size, -size, 0.0), (size, size, 0.0), stroke_color=color, stroke_width=6),
        Line((-size, size, 0.0), (size, -size, 0.0), stroke_color=color, stroke_width=6),
    )


def record_card(rows: Sequence[tuple[str, bool]], title: str = "record") -> VGroup:
    """A record: its title, then per row a label and a tick or a grey dash."""
    lines = VGroup()
    for name, shown in rows:
        mark = tick() if shown else Line(LEFT * 0.1, RIGHT * 0.1, stroke_color=GREY)
        text = Text(name, font_size=SMALL_FONT, fill_color=INK if shown else GREY)
        lines.add(VGroup(mark, text.next_to(mark, RIGHT, buff=0.2)))
    lines.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    heading = Text(title, font_size=SMALL_FONT, fill_color=HIGHLIGHT)
    body = VGroup(heading, lines).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    frame = RoundedRectangle(
        width=body.get_width() + 0.5,
        height=body.get_height() + 0.4,
        corner_radius=0.12,
        stroke_color=GREY,
        fill_color="#262626",
        fill_opacity=1,
    )
    return VGroup(frame, body.move_to(frame))


def bar_chart(
    values: Mapping[str, float], *, unit: float, shown: Mapping[str, str] | None = None
) -> VGroup:
    """Bars on one baseline, a name under each and its value over it.

    Args:
        values: Each bar's name and value, in drawing order.
        unit: The height of one unit of value.
        shown: The text over each bar; the value itself when omitted.

    Returns:
        A group of three: the bars, the names and the values, each indexed in `values`'s order.
    """
    bars, names, numbers = VGroup(), VGroup(), VGroup()
    for index, (name, value) in enumerate(values.items()):
        bar = Rectangle(
            width=1.0, height=max(value * unit, 0.01), fill_color=INK, fill_opacity=0.85
        )
        bar.set_stroke(width=0)
        bar.move_to((index * 1.55, 0.0, 0.0), aligned_edge=DOWN)
        bars.add(bar)
        names.add(Text(name, font_size=16, fill_color=INK).next_to(bar, DOWN, buff=0.15))
        number = shown[name] if shown is not None else f"{value:g}"
        numbers.add(Text(number, font_size=SMALL_FONT, fill_color=INK).next_to(bar, UP, buff=0.1))
    return VGroup(bars, names, numbers)


def ontology_diagram() -> tuple[dict[str, VGroup], dict[tuple[str, str], Arrow]]:
    """The eight classes as dots and their `is_a` edges as arrows, narrow to broad."""
    nodes = {concept: class_node(concept) for concept in CHAIN_LAYOUT}
    arrows = {(child, parent): edge(nodes[child][0], nodes[parent][0]) for child, parent in EDGES}
    return nodes, arrows


def statement(child: VGroup, parent: VGroup, probability: str = "1") -> VGroup:
    """An `is_a` edge as an S-node: arrows from the tail to the box and from the box to the head."""
    box = snode(probability).move_to((child[0].get_center() + parent[0].get_center()) / 2)
    return VGroup(edge(child[0], box), box, edge(box, parent[0]))


def bkb_diagram(
    held: Collection[str] = (),
) -> tuple[dict[str, VGroup], dict[tuple[str, str], VGroup]]:
    """The eight I-nodes and an S-node per `is_a` edge; `held` I-nodes are filled."""
    nodes = {concept: inode(concept, held=concept in held) for concept in CHAIN_LAYOUT}
    statements = {(c, p): statement(nodes[c], nodes[p]) for c, p in EDGES}
    return nodes, statements


def prior(node: VGroup, probability: str) -> VGroup:
    """A prior S-node on `node`: a box to its left and an arrow into its circle."""
    box = snode(probability, color=DATA_ORANGE).next_to(node[0], LEFT, buff=0.7)
    return VGroup(box, edge(box, node[0]))


def data_priors(nodes: Mapping[str, VGroup]) -> dict[str, VGroup]:
    """The cell's dataset-fragment priors, each on its node, read from `video.model`."""
    return {s.head[0]: prior(nodes[s.head[0]], f"{s.prob}") for s in dataset_fragment().snodes}


def held_concepts() -> tuple[str, ...]:
    """The concepts the cell's records show, sorted."""
    return tuple(sorted(c for c, count in HOLDERS.items() if count > 0))


def bko_diagram() -> tuple[dict[str, VGroup], dict[tuple[str, str], VGroup], dict[str, VGroup]]:
    """The cell's BKO as drawn: held I-nodes filled, the S-node per edge and the data priors."""
    nodes, statements = bkb_diagram(held_concepts())
    return nodes, statements, data_priors(nodes)


def box(text: str, *, width: float, color: str = INK) -> VGroup:
    """A labelled rounded box, one line high."""
    label = Text(text, font_size=20, fill_color=color)
    frame = RoundedRectangle(width=width, height=0.5, corner_radius=0.1, stroke_color=color)
    return VGroup(frame, label.move_to(frame))


def tag(text: str) -> VGroup:
    """A small boxed tag, such as "part of HPO"."""
    label = Text(text, font_size=SMALL_FONT, fill_color=GREY)
    box = RoundedRectangle(
        width=label.get_width() + 0.3,
        height=label.get_height() + 0.2,
        corner_radius=0.08,
        stroke_color=GREY,
    )
    return VGroup(box, label.move_to(box))


class SectionScene(Scene):
    """One section of the video: its title throughout, then each beat's label and explainer.

    Attributes:
        name: The section's Scene class name in `video.captions.SECTIONS`.
    """

    name: ClassVar[str]

    def construct(self) -> None:
        """Show the title, then play each beat's fade, diagram and hold, and end on the diagram."""
        this = section(self.name)
        self.add(Text(this.title, font_size=40, fill_color=INK).to_edge(UP).fix_in_frame())
        shown: VGroup | None = None
        for index, (label, note) in enumerate(this.beats):
            text = VGroup(
                Text(label, font_size=30, line_width=13.0, fill_color=HIGHLIGHT),
                Text(note, font_size=22, line_width=13.0, fill_color=INK),
            ).arrange(DOWN, buff=0.12)
            text.to_edge(DOWN, buff=0.22).fix_in_frame()
            fades = [FadeIn(text)] if shown is None else [FadeOut(shown), FadeIn(text)]
            self.play(*fades, run_time=FADE_SECONDS)
            shown = text
            start = self.time
            self.beat(index)
            remaining = hold_seconds(label, note) - (self.time - start)
            if remaining > 0:
                self.wait(remaining)
        self.wait(SCENE_END_SECONDS)

    def beat(self, index: int) -> None:
        """Draw caption `index`'s beat through the section's `_beat_<index + 1>`, if it has one.

        Args:
            index: The caption's position in the section.
        """
        method = getattr(self, f"_beat_{index + 1}", None)
        if method is not None:
            method()
