"""The fields a pooled direct draw copies by the pooled rule, chosen once per fill.

A `Presence.POOLED` draw is `bko-nomap`'s record with a set of units pooled: each unit's donor is
a uniform cell record with probability n/(n + `GROUP_WEIGHT`), else a uniform record of the whole
label group (`bko.generate.donors._pick_pooled`). `pooled_units` says which units: the mapping
groups at `Pool.MAPPED`; at `Pool.RANDOM`, as many single fields as those groups hold, drawn
without replacement from the features less `age` and `sex` by their own generator, so the draw's
generator is untouched; at `Pool.ALL`, every such feature. ADR
`research/matched-knowledge-controls` records the design.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import numpy as np

from bko.datasets.frames import FrameSpec
from bko.generate.draw_spec import DrawSpec, Pool
from bko.generate.fields import field_groups
from bko.generate.presence import Presence
from bko.ontology.schema import FieldMapping


logger = logging.getLogger(__name__)

POOL_RANDOM_STREAM: Final = 19
"""The second seed word of `Pool.RANDOM`'s generator, `np.random.default_rng([seed, 19])`."""


def features(frame: FrameSpec) -> tuple[str, ...]:
    """`frame`'s columns less `age`, `sex` and the label, in column order."""
    return tuple(c for c in frame.columns if c not in {"age", "sex", frame.label})


def pooled_units(
    rows: Sequence[FieldMapping], frame: FrameSpec, pool: Pool, *, seed: int
) -> tuple[tuple[str, ...], ...]:
    """The units a pooled draw of `rows` at `pool` copies by the pooled rule.

    `Pool.MAPPED` gives each field group of `rows`, whole; `Pool.RANDOM` reads `rows` only for how
    many fields those groups hold, and draws that many from `features` at `seed`; `Pool.ALL` reads
    no row. A `RANDOM` or `ALL` unit is one field, in column order.

    Raises:
        ValueError: At `Pool.RANDOM`, if the groups hold more fields than `features` holds.
    """
    if pool is Pool.MAPPED:
        return tuple(group.fields for group in field_groups(rows, frame=frame))
    candidates = features(frame)
    if pool is Pool.ALL:
        return tuple((field,) for field in candidates)
    held = sum(len(group.fields) for group in field_groups(rows, frame=frame))
    if held > len(candidates):
        msg = f"pool random draws {held} fields but frame {frame.name} has {len(candidates)} "
        raise ValueError(msg + "features besides age and sex")
    rng = np.random.default_rng([seed, POOL_RANDOM_STREAM])
    chosen = {int(k) for k in rng.choice(len(candidates), size=held, replace=False)}
    return tuple((field,) for k, field in enumerate(candidates) if k in chosen)


def pooled_for(
    rows: Sequence[FieldMapping],
    frame: FrameSpec,
    spec: DrawSpec,
    *,
    seed: int,
    tag: tuple[int, str],
) -> tuple[tuple[str, ...], ...]:
    """`pooled_units` at `spec`'s pool when its presence is pooled, else none.

    At `Pool.RANDOM` it logs `pool-random holdout=%d fill=%s fields=%s`, `tag`'s holdout and
    fill, the drawn fields comma-joined in column order.
    """
    if spec.presence is not Presence.POOLED:
        return ()
    units = pooled_units(rows, frame, spec.pool, seed=seed)
    if spec.pool is Pool.RANDOM:
        fields = ",".join(field for unit in units for field in unit)
        logger.info("pool-random holdout=%d fill=%s fields=%s", *tag, fields)
    return units
