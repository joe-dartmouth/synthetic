"""A broad concept drawn given its narrow one: the conditional `is_a` draw of the `bko-hier` arms.

For an `is_a` pair, narrow `N` under broad `B`, a cell of `n_c` records and its label group, each
source holds `N`'s prior `p` and the conditional `c` of `B` given `N`, the S-node
`bko.gaps.observe.conditional_fragment` holds. `pair_weights` reads `m = n_c p_c + w p_g` and
`b = n_c p_c c_c + w p_g c_g` from the counts those S-nodes come from, with `w` as
`bko.gaps.shares.GROUP_WEIGHT`; `conditional` shrinks `b / m` toward the `is_a` S-node's
probability 1 by a pseudo-count, and `broad_chance` reads `B`'s chance given whether `N` is
present, the complement keeping `B`'s marginal at its share unless the clamp at 0 or 1 fires (ADR
`research/bkb-direct-generation`, Decision 12). Each is computed exactly and rounded once.
"""

from __future__ import annotations

from fractions import Fraction

from bko.gaps.observe import Observed
from bko.gaps.shares import GROUP_WEIGHT


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
    """
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


def broad_chance(
    *, share_broad: Exact, share_narrow: Exact, conditional: Exact, narrow_present: bool
) -> tuple[float, bool]:
    """The broad concept's chance given the narrow one's presence, and whether the clamp fired.

    With the narrow concept present it is `conditional`; absent, it is the complement
    `(s_B - s_N c) / (1 - s_N)`, clamped to [0, 1].

    Raises:
        ValueError: If the narrow concept is absent at a share of 1, where it is always present.
    """
    if narrow_present:
        return float(conditional), False
    chance, clamped = _complement(
        Fraction(share_broad), Fraction(share_narrow), Fraction(conditional)
    )
    return float(chance), clamped


def pair_joint(
    *, share_broad: Exact, share_narrow: Exact, conditional: Exact
) -> dict[tuple[bool, bool], float]:
    """The chance of each (narrow present, broad present) under the conditional draw.

    The narrow concept is drawn at its share and the broad one at `broad_chance`; at a narrow
    share of 1 the narrow concept is always present.
    """
    s_b, s_n, given = Fraction(share_broad), Fraction(share_narrow), Fraction(conditional)
    absent = Fraction(0) if s_n == 1 else _complement(s_b, s_n, given)[0]
    return {
        (True, True): float(s_n * given),
        (True, False): float(s_n * (1 - given)),
        (False, True): float((1 - s_n) * absent),
        (False, False): float((1 - s_n) * (1 - absent)),
    }


def _complement(
    share_broad: Fraction, share_narrow: Fraction, given: Fraction
) -> tuple[Fraction, bool]:
    """`(s_B - s_N c) / (1 - s_N)` clamped to [0, 1], and whether the clamp fired.

    Raises:
        ValueError: If `share_narrow` is 1.
    """
    if share_narrow == 1:
        msg = "the narrow concept has share 1, so no record lacks it and no complement is read"
        raise ValueError(msg)
    raw = (share_broad - share_narrow * given) / (1 - share_narrow)
    clamped = min(Fraction(1), max(Fraction(0), raw))
    return clamped, clamped != raw
