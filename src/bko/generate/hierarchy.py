"""A broad concept drawn given its narrow one: the conditional `is_a` draw of the `bko-hier` arms.

For an `is_a` pair, narrow `N` under broad `B`, a cell of `n_c` records and its label group, each
source holds `N`'s prior `p` and the conditional `c` of `B` given `N`, the S-node
`bko.gaps.observe.conditional_fragment` holds. `pair_weights` reads `m = n_c p_c + w p_g` and
`b = n_c p_c c_c + w p_g c_g` from the counts those S-nodes come from, with `w` as
`bko.gaps.shares.GROUP_WEIGHT`; `conditional` shrinks `b / m` toward the `is_a` S-node's
probability 1 by a pseudo-count, and `broad_chance` reads `B`'s chance given whether `N` is
present, the complement keeping `B`'s marginal at its share unless the clamp at 0 or 1 fires (ADR
`research/bkb-direct-generation`, Decision 12). Each reads its inputs as exact binary fractions
and rounds its result once; a `PairPart`'s complement reads its rounded conditional, the value the
draw uses. A narrow share within one ulp of 1 counts as 1 (`always_present`), so no complement is
read there.

`Hierarchy` holds a draw's pairs; `Hierarchy.at` reads each pair's `PairPart` at a cell and
`Hierarchy.present` decides presence from one uniform per concept. `bko.generate.direct.sample`
draws through it and `bko.evaluate.redraw.Shares` traces through it, so the trace reads the
draw's own parts.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from bko.gaps.contradictions import is_a_pairs
from bko.gaps.observe import Cell, Observed, cell_key
from bko.gaps.shares import GROUP_WEIGHT, draw_order
from bko.ontology.terms import Ontology


logger = logging.getLogger(__name__)


type Exact = float | Fraction
"""A value read exactly: a float is its own binary fraction."""


def pair_weights(
    cell: Observed, group: Observed, pair: tuple[str, str]
) -> tuple[Fraction, Fraction]:
    """`(b, m)` of `pair`, `(narrow, broad)`, at a cell whose label group is `group`.

    `n_c p_c c_c` is the cell's records showing both concepts and `n_c p_c` its records showing
    the narrow one, so a source no record of which shows the narrow concept contributes 0; the
    group's terms are those counts over its records, times `GROUP_WEIGHT`. Both observations must
    count the pair (`Observed.both`).

    Raises:
        ValueError: If `group` holds no record, so it has no share to weigh.
    """
    if group.records == 0:
        msg = f"label group {group.name} holds no record, so it weighs no {pair} pair"
        raise ValueError(msg)
    narrow, _ = pair
    scale = Fraction(GROUP_WEIGHT, group.records)
    both = cell.both[pair] + scale * group.both[pair]
    held = cell.holders[narrow] + scale * group.holders[narrow]
    return both, held


def conditional(*, both: Exact, narrow: Exact, pseudo: float | None) -> float:
    """`c_h = (b + h) / (m + h)` at pseudo-count `h`; 1.0 when `pseudo` is None, the closed draw.

    It is 1.0 too where `b + h` and `m + h` are both 0, its limit as `h` falls to 0: no record
    shows the narrow concept, so none weighs against the `is_a` S-node.
    """
    if pseudo is None:
        return 1.0
    weight = Fraction(narrow) + Fraction(pseudo)
    if weight == 0:
        return 1.0
    return float((Fraction(both) + Fraction(pseudo)) / weight)


def always_present(share_narrow: Exact) -> bool:
    """Whether a narrow concept at this share is always present: its share at least 1's float below.

    No record lacks it there, so no complement is read.
    """
    return Fraction(share_narrow) >= Fraction(math.nextafter(1.0, 0.0))


def broad_chance(
    *, share_broad: Exact, share_narrow: Exact, given: Exact, narrow_present: bool
) -> tuple[float, bool]:
    """The broad concept's chance given the narrow one's presence, and whether the clamp fired.

    With the narrow concept present it is `given`, the conditional; absent, it is the complement
    `(s_B - s_N c) / (1 - s_N)`, clamped to [0, 1].

    Raises:
        ValueError: If the narrow concept is absent where it is `always_present`.
    """
    if narrow_present:
        return float(given), False
    chance, clamped = _complement(Fraction(share_broad), Fraction(share_narrow), Fraction(given))
    return float(chance), clamped


def pair_joint(
    *, share_broad: Exact, share_narrow: Exact, given: Exact
) -> dict[tuple[bool, bool], float]:
    """The chance of each (narrow present, broad present) under the conditional draw.

    The narrow concept is drawn at its share and the broad one at `broad_chance`; where the
    narrow concept is `always_present`, no complement is read and the broad one reads `given`
    either way, as `Hierarchy.present` does.
    """
    s_b, s_n, c = Fraction(share_broad), Fraction(share_narrow), Fraction(given)
    absent = c if always_present(s_n) else _complement(s_b, s_n, c)[0]
    return {
        (True, True): float(s_n * c),
        (True, False): float(s_n * (1 - c)),
        (False, True): float((1 - s_n) * absent),
        (False, False): float((1 - s_n) * (1 - absent)),
    }


def _complement(
    share_broad: Fraction, share_narrow: Fraction, given: Fraction
) -> tuple[Fraction, bool]:
    """`(s_B - s_N c) / (1 - s_N)` clamped to [0, 1], and whether the clamp fired.

    Raises:
        ValueError: If the narrow concept is `always_present`.
    """
    if always_present(share_narrow):
        msg = "the narrow concept has share 1, so no record lacks it and no complement is read"
        raise ValueError(msg)
    raw = (share_broad - share_narrow * given) / (1 - share_narrow)
    clamped = min(Fraction(1), max(Fraction(0), raw))
    return clamped, clamped != raw


@dataclass(frozen=True, slots=True)
class PairPart:
    """What a broad concept's draw reads at a cell: its pair's weights and its two chances.

    Attributes:
        given: The narrow concept it is drawn given.
        both: `b`, the weighted records showing both concepts.
        narrow: `m`, the weighted records showing the narrow one.
        pseudo: The pseudo-count; None for the closed draw.
        conditional: Its chance when the narrow concept is present (`conditional`).
        complement: Its chance when the narrow concept is absent (`broad_chance`); None where the
            narrow concept is `always_present`.
        clamped: Whether the complement was clamped to [0, 1].
    """

    given: str
    both: float
    narrow: float
    pseudo: float | None
    conditional: float
    complement: float | None
    clamped: bool


class Hierarchy:
    """The `is_a` pairs of a draw's concepts, the order presence is decided in, and the pseudo."""

    def __init__(
        self,
        order: Sequence[str],
        inner_of: Mapping[str, str],
        ontologies: Mapping[str, Ontology],
        *,
        pseudo: float | None,
    ) -> None:
        """`order` is the concepts drawn at a share, in the order their uniforms are drawn.

        `inner_of` maps each nested pair's outer concept to its inner one.

        Raises:
            ValueError: If `inner_of` names a nested pair, if no `is_a` pair joins two of `order`,
                or if a broad concept has more than one narrow one.
        """
        if inner_of:
            pairs = ", ".join(f"{inner} in {outer}" for outer, inner in sorted(inner_of.items()))
            msg = f"a conditional draw cannot share a stream with the nested pairs {pairs}"
            raise ValueError(msg)
        self.pairs = is_a_pairs(order, ontologies)
        if not self.pairs:
            msg = "a conditional draw needs an is_a pair among the drawn concepts, and none joins"
            raise ValueError(msg)
        narrow = Counter(broad for _, broad in self.pairs)
        wide = sorted(broad for broad, times in narrow.items() if times > 1)
        if wide:
            msg = f"{', '.join(wide)} each hold more than one narrow concept; one is drawn given"
            raise ValueError(msg)
        self.order = tuple(order)
        self.decided = draw_order(order, self.pairs)
        self.pseudo = pseudo

    def at(
        self, cell: Observed, group: Observed, share: Mapping[str, float]
    ) -> dict[str, PairPart]:
        """Each broad concept's `PairPart` at a cell, by broad concept, in pair order.

        `cell` and `group` must count the pairs (`Observed.both`); `share` is each concept's.
        """
        parts = {}
        for narrow, broad in self.pairs:
            both, held = pair_weights(cell, group, (narrow, broad))
            given = conditional(both=both, narrow=held, pseudo=self.pseudo)
            complement, clamped = None, False
            if not always_present(share[narrow]):
                complement, clamped = broad_chance(
                    share_broad=share[broad],
                    share_narrow=share[narrow],
                    given=given,
                    narrow_present=False,
                )
            parts[broad] = PairPart(
                narrow, float(both), float(held), self.pseudo, given, complement, clamped
            )
        return parts

    def present(
        self,
        present: set[str],
        uniforms: Sequence[float],
        share: Mapping[str, float],
        parts: Mapping[str, PairPart],
    ) -> set[str]:
        """`present` with each concept whose uniform falls below its chance.

        `uniforms` holds one per concept of `order`, in that order; presence is decided in
        `decided`, each narrow concept before its broad one. A concept outside every pair reads
        its share; a broad one its `PairPart`'s conditional or complement.
        """
        drawn = dict(zip(self.order, uniforms, strict=True))
        for concept in self.decided:
            part = parts.get(concept)
            if part is None:
                chance = share[concept]
            elif part.given in present or part.complement is None:
                chance = part.conditional
            else:
                chance = part.complement
            if drawn[concept] < chance:
                present.add(concept)
        return present


def log_pairs(holdout: int, fill: str, cell: Cell, parts: Mapping[str, PairPart]) -> None:
    """Log the pairs drawn at `cell` and how many of their complements were clamped."""
    clamped = sum(part.clamped for part in parts.values())
    logger.info(
        "hier holdout=%d fill=%s key=%s pairs=%d clamped=%d",
        holdout,
        fill,
        cell_key(cell),
        len(parts),
        clamped,
    )
