"""A Bayesian network learned from the training rows and sampled ancestrally: PrivBayes, no noise.

Each measured field is cut into `BINS` equal-frequency bins, with every age band edge offered as a
cut, so each age state's training rows lie in one band; a coded field (the label included) keeps
as its states the codes its training rows hold. GreedyBayes (Zhang et al. 2014) orders the fields:
the first is given, and each next field is the one whose best parent set, `min(k, placed)` fields
already placed, shares the most mutual information with it; ties fall to the first field in the
frame's order, then to the first parent set in placement order. Each field's distribution given
its parents is the BDeu posterior mean at equivalent sample size `ESS`, so a parent configuration
no training row shows gets the uniform prior. A record is drawn field by field in that order, and
each field's value is copied from a training row whose state for that field is the drawn one. ADR
`research/bayesian-network-baseline` records the choices and where they depart from PrivBayes.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.records import Baseline, derived_seed, unit_weights
from bko.baselines.request import Request
from bko.baselines.synthesizers import fill
from bko.datasets.frames import FrameSpec


logger = logging.getLogger(__name__)

BINS: Final = 5
"""The equal-frequency bins of a measured field, before the band edges are forced."""

ESS: Final = 1.0
"""BDeu's equivalent sample size."""

K: Final = 2
"""The most parents a field takes."""


@dataclass(frozen=True, slots=True, eq=False)
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


@dataclass(frozen=True, slots=True, eq=False)
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


@dataclass(frozen=True, slots=True, eq=False)
class Network:
    """The learned network, its nodes in drawing order."""

    nodes: tuple[Node, ...]

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Each (parent, child) pair, in drawing order."""
        return [(parent, node.field) for node in self.nodes for parent in node.parents]


def discretize(train: pd.DataFrame, frame: FrameSpec) -> Bins:
    """Each field of `frame` cut into states over `train`: bins when measured, codes when coded.

    A measured field's cuts are its `BINS - 1` interior quantiles, with each of `frame.edges` added
    for `age`, then each cut that would leave a bin empty removed; a coded field's states are the
    codes, in the frame's order, that `train` holds. Every state holds a training row, and each
    age state's rows lie in one band.
    """
    cuts: dict[str, tuple[float, ...]] = {}
    codes: dict[str, tuple[int, ...]] = {}
    for field in frame.columns:
        if field == frame.label or field in frame.categorical:
            known = frame.label_codes if field == frame.label else tuple(frame.codebook[field])
            present = set(train[field].astype(int))
            codes[field] = tuple(code for code in known if code in present)
        else:
            values = train[field].to_numpy(dtype="float64")
            quantiles = np.quantile(values, [i / BINS for i in range(1, BINS)])
            forced = [float(edge) for edge in frame.edges] if field == "age" else []
            cuts[field] = _occupied(values, sorted({*map(float, quantiles), *forced}))
    states = pd.DataFrame(
        {field: _states(train[field], field, cuts, codes) for field in frame.columns},
        index=train.index,
    )
    return Bins(MappingProxyType(cuts), MappingProxyType(codes), states)


def mutual_information(states: pd.DataFrame, field: str, parents: Sequence[str]) -> float:
    """The empirical mutual information, in nats, between `field` and the joint `parents`."""
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
    # A (field, parents) pair's information never changes, and later rounds offer it again.
    known: dict[tuple[str, tuple[str, ...]], float] = {}

    def information(field: str, parents: tuple[str, ...]) -> float:
        key = (field, parents)
        if key not in known:
            known[key] = mutual_information(bins.states, field, parents)
        return known[key]

    while len(placed) < len(fields):
        candidates = [
            (information(field, parents), field, parents)
            for field in fields
            if field not in placed
            for parents in itertools.combinations(placed, min(k, len(placed)))
        ]
        # The first of the largest: frame order of fields, then placement order of parent sets.
        _, field, parents = max(candidates, key=itemgetter(0))
        placed.append(field)
        chosen.append((field, parents))
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


def bayes_net(train: pd.DataFrame, request: Request) -> Baseline:
    """The network learned on `train`, records drawn until each stratum holds its quota.

    The first field is `frame.columns[derived_seed(seed, "bayes-net") % len(frame.columns)]`, and
    every draw reads `default_rng(derived_seed(seed, "bayes-net"))`, so the seed alone decides the
    rows and no global generator is read or moved. Logs one `fit method=bayes-net seconds=%.1f`
    line. The params name the request's seed, the first field, `k`, `BINS`, the equivalent sample
    size, each field's states and the learned edges, each `[parent, child]` in drawing order.

    Raises:
        BaselineError: As `fill`, if the draws cannot meet the quota; its shortfall holds no
            baseline, so the harness refuses it rather than filling partially.
    """
    frame = request.frame
    rows = train.loc[:, list(frame.columns)].reset_index(drop=True)
    seed = derived_seed(request.seed, "bayes-net")
    first = frame.columns[seed % len(frame.columns)]
    start = time.perf_counter()
    bins = discretize(rows, frame)
    network = learn(bins, first=first)
    logger.info("fit method=bayes-net seconds=%.1f", time.perf_counter() - start)
    rng = np.random.default_rng(seed)
    added = fill(
        lambda size: draw(network, bins, rows, size, rng, frame), request, method="bayes-net"
    )
    params = {
        "seed": request.seed,
        "first": first,
        "k": K,
        "bins": BINS,
        "ess": ESS,
        "states": {field: bins.arity(field) for field in frame.columns},
        "edges": [list(edge) for edge in network.edges],
    }
    return Baseline(added=added, weights=unit_weights(train), params=params)


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
        known = np.asarray(codes[field], dtype=np.int64)
        values = column.to_numpy().astype(np.int64)
        low = int(min(known.min(initial=0), values.min(initial=0)))
        lookup = np.full(int(max(known.max(initial=0), values.max(initial=0))) - low + 1, -1)
        lookup[known - low] = np.arange(len(known))
        found = lookup[values - low]
        if (found < 0).any():
            msg = (
                f"{field} holds codes {sorted(set(values[found < 0].tolist()))} outside its states"
            )
            raise KeyError(msg)
        return found.astype(int)
    return np.searchsorted(np.asarray(cuts[field]), column.to_numpy(dtype="float64"), side="right")


def _node(bins: Bins, field: str, parents: tuple[str, ...]) -> Node:
    """`field`'s BDeu posterior mean given `parents`, over every configuration the rows show."""
    arity = bins.arity(field)
    configurations = math.prod(bins.arity(parent) for parent in parents)
    prior = ESS / (configurations * arity)
    radices = [bins.arity(parent) for parent in parents]
    # Each row's parent configuration as one mixed-radix index, the last parent varying fastest.
    index = np.zeros(len(bins.states), dtype=np.int64)
    for parent, radix in zip(parents, radices, strict=True):
        index = index * radix + bins.states[parent].to_numpy(dtype=np.int64)
    cells = index * arity + bins.states[field].to_numpy(dtype=np.int64)
    counts = np.bincount(cells, minlength=configurations * arity).reshape(configurations, arity)
    table: dict[tuple[int, ...], tuple[float, ...]] = {}
    for at in np.flatnonzero(counts.sum(axis=1)):
        seen = [int(n) for n in counts[at]]
        configuration = tuple(int(v) for v in np.unravel_index(at, radices)) if radices else ()
        table[configuration] = tuple((n + prior) / (sum(seen) + prior * arity) for n in seen)
    return Node(field, parents, table, arity)
