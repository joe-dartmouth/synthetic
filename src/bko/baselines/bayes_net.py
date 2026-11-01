"""A Bayesian network learned from the training rows and sampled ancestrally: PrivBayes, no noise.

Each measured field is cut into `BINS` equal-frequency bins, with every age band edge forced as a
cut so a sampled age bin names one band; a coded field (the label included) keeps its codes as its
states. GreedyBayes (Zhang et al. 2014) orders the fields: the first is given, and each next field
is the one whose best parent set, `min(k, placed)` fields already placed, shares the most mutual
information with it; ties fall to the first in the frame's order. Each field's distribution given
its parents is the BDeu posterior mean at equivalent sample size `ESS`, so a parent configuration
no training row shows gets the uniform prior. A record is drawn field by field in that order, and
each field's value is copied from a training row whose state for that field is the drawn one. ADR
`research/bayesian-network-baseline` records the choices and where they depart from PrivBayes.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from bko.datasets.frames import FrameSpec


BINS: Final = 5
"""The equal-frequency bins of a measured field, before the band edges are forced."""

ESS: Final = 1.0
"""BDeu's equivalent sample size."""

K: Final = 2
"""The most parents a field takes."""


@dataclass(frozen=True, slots=True)
class Bins:
    """Each field's states over the training rows.

    Attributes:
        cuts: Each measured field's cut points, ascending; a value's state is the number of cuts at
            or below it.
        codes: Each coded field's codes; a value's state is its code's position.
        states: The training rows' states, one integer column per field, on their index.
    """

    cuts: Mapping[str, tuple[float, ...]]
    codes: Mapping[str, tuple[int, ...]]
    states: pd.DataFrame

    def arity(self, field: str) -> int:
        """How many states `field` has."""
        if field in self.codes:
            return len(self.codes[field])
        return len(self.cuts[field]) + 1


@dataclass(frozen=True, slots=True)
class Node:
    """One field of the network: its parents and its distribution given each of their states.

    Attributes:
        field: The field.
        parents: The fields it is drawn given, in order.
        table: Each parent configuration's probabilities over the field's states.
        arity: How many states the field has.
    """

    field: str
    parents: tuple[str, ...]
    table: Mapping[tuple[int, ...], tuple[float, ...]]
    arity: int

    def chances(self, given: tuple[int, ...]) -> tuple[float, ...]:
        """The field's distribution at parent states `given`; the uniform prior if unseen."""
        return self.table.get(given, tuple([1.0 / self.arity] * self.arity))


@dataclass(frozen=True, slots=True)
class Network:
    """The learned network, its nodes in drawing order."""

    nodes: tuple[Node, ...]

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Each (parent, child) pair, in drawing order."""
        return [(parent, node.field) for node in self.nodes for parent in node.parents]


def discretize(train: pd.DataFrame, frame: FrameSpec) -> Bins:
    """Each field of `frame` cut into states over `train`: bins when measured, codes when coded.

    A measured field's cuts are its `BINS`-quantile values, with each of `frame.edges` added for
    `age`, then the cuts that would leave a bin empty removed, so every state holds a training row.
    """
    cuts: dict[str, tuple[float, ...]] = {}
    codes: dict[str, tuple[int, ...]] = {}
    for field in frame.columns:
        if field == frame.label:
            codes[field] = tuple(frame.label_codes)
        elif field in frame.categorical:
            codes[field] = tuple(frame.codebook[field])
        else:
            values = train[field].to_numpy(dtype="float64")
            quantiles = np.quantile(values, [i / BINS for i in range(1, BINS)])
            forced = [float(edge) for edge in frame.edges] if field == "age" else []
            cuts[field] = _occupied(values, sorted({*map(float, quantiles), *forced}))
    states = pd.DataFrame(
        {field: _states(train[field], field, cuts, codes) for field in frame.columns},
        index=train.index,
    )
    return Bins(cuts, codes, states)


def mutual_information(states: pd.DataFrame, field: str, parents: Sequence[str]) -> float:
    """The empirical mutual information, in nats, between `field` and the joint `parents`."""
    if not parents:
        return 0.0
    rows = len(states)
    child = states[field].to_numpy()
    joint = [tuple(row) for row in states.loc[:, list(parents)].to_numpy()]
    both = Counter(zip(joint, child, strict=True))
    left = Counter(joint)
    right = Counter(child)
    return sum(n / rows * math.log(n * rows / (left[p] * right[c])) for (p, c), n in both.items())


def learn(bins: Bins, *, k: int = K, first: str) -> Network:
    """GreedyBayes' order and parents over `bins`, each node's table the BDeu posterior mean.

    Raises:
        ValueError: If `first` is not a field of `bins`.
    """
    fields = list(bins.states.columns)
    if first not in fields:
        msg = f"the first field {first!r} is not one of {fields}"
        raise ValueError(msg)
    placed = [first]
    chosen: list[tuple[str, tuple[str, ...]]] = [(first, ())]
    while len(placed) < len(fields):
        best: tuple[float, str, tuple[str, ...]] | None = None
        for field in (f for f in fields if f not in placed):
            for parents in itertools.combinations(placed, min(k, len(placed))):
                score = mutual_information(bins.states, field, parents)
                if best is None or score > best[0]:
                    best = (score, field, parents)
        if best is None:
            msg = "no field is left to place"
            raise ValueError(msg)
        placed.append(best[1])
        chosen.append((best[1], best[2]))
    return Network(tuple(_node(bins, field, parents) for field, parents in chosen))


def draw(
    network: Network,
    bins: Bins,
    train: pd.DataFrame,
    rows: int,
    rng: np.random.Generator,
    frame: FrameSpec,
) -> pd.DataFrame:
    """`rows` records sampled from `network`, each field's value a training donor's in its state."""
    drawn: dict[str, np.ndarray] = {}
    for node in network.nodes:
        given = list(zip(*(drawn[parent] for parent in node.parents), strict=True))
        drawn[node.field] = np.array(
            [rng.choice(node.arity, p=node.chances(parents)) for parents in (given or [()] * rows)],
            dtype=int,
        )
    out = pd.DataFrame(index=range(rows))
    for field in frame.columns:
        values = train[field].to_numpy()
        donors = {
            state: values[bins.states[field].to_numpy() == state] for state in set(drawn[field])
        }
        out[field] = [rng.choice(donors[state]) for state in drawn[field]]
    return out.astype(dict(frame.dtypes))


def _occupied(values: np.ndarray, cuts: list[float]) -> tuple[float, ...]:
    """`cuts` less each one that leaves its bin, the values from the cut before it, empty."""
    kept: list[float] = []
    for cut in cuts:
        low = kept[-1] if kept else -math.inf
        if np.any((values >= low) & (values < cut)) and np.any(values >= cut):
            kept.append(cut)
    return tuple(kept)


def _states(
    column: pd.Series,
    field: str,
    cuts: Mapping[str, tuple[float, ...]],
    codes: Mapping[str, tuple[int, ...]],
) -> np.ndarray:
    if field in codes:
        position = {code: at for at, code in enumerate(codes[field])}
        return np.array([position[int(value)] for value in column], dtype=int)
    return np.searchsorted(np.asarray(cuts[field]), column.to_numpy(dtype="float64"), side="right")


def _node(bins: Bins, field: str, parents: tuple[str, ...]) -> Node:
    """`field`'s BDeu posterior mean given `parents`, over every configuration the rows show."""
    arity = bins.arity(field)
    configurations = math.prod(bins.arity(parent) for parent in parents)
    prior = ESS / (configurations * arity)
    counts: dict[tuple[int, ...], list[int]] = {}
    child = bins.states[field].to_numpy()
    given = bins.states.loc[:, list(parents)].to_numpy()
    for row, state in zip(given, child, strict=True):
        counts.setdefault(tuple(int(v) for v in row), [0] * arity)[int(state)] += 1
    table = {
        configuration: tuple((n + prior) / (sum(seen) + prior * arity) for n in seen)
        for configuration, seen in counts.items()
    }
    return Node(field, parents, table, arity)
