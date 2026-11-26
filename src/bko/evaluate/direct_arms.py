"""The direct arms by name, each with the draw spec it reads: a light module the run table imports.

`bko.evaluate.experiment` re-exports `DIRECT_ARM`, `DIRECT_ARMS`, `LOGIC_ARM`, `NOMAP_ARM`,
`PRAM_ARM` and `COHERENT_ARM`, the last for `bko.evaluate.holdout_candidates`; every other name
here is imported from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.gaps.shares import GROUP_WEIGHT
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec, Pool
from bko.generate.presence import Presence


DIRECT_ARM: Final = "bko-direct"
"""The free arm that draws records from the fused BKB with no API call (`bko.evaluate.direct`)."""

LOGIC_ARM: Final = "bko-logic"
"""The free arm that draws as `DIRECT_ARM` does with each concept's presence from logic alone, no
data prior (ADR `research/logic-only-ablation`)."""

PRAM_ARM: Final = "bko-direct-pram"
"""The free arm that redraws `DIRECT_ARM`'s unmapped fields by PRAM toward each field's label-group
training marginal (`bko.evaluate.pram`; ADR `research/identifier-perturbation`)."""

NOMAP_ARM: Final = "bko-nomap"
"""The free arm that draws as `DIRECT_ARM` does with an empty mapping table: no concept, and every
column but `age`, `sex` and the label a free field (ADR `research/no-mapping-ablation`)."""

CAPPED_ARM: Final = "bko-capped"
"""The free arm that draws as `DIRECT_ARM` does under a donor floor of 2 and a reuse cap of 1 per
unit (ADR `research/sampler-donor-limits`)."""

MICRO_ARM: Final = "bko-micro"
"""The free arm that draws as `CAPPED_ARM` does with each numeric field but `age` the mean of 3
distinct donors of one key class (`bko.generate.micro`; ADR `research/sampler-donor-limits`)."""

TREE_ARM: Final = "bko-tree"
"""The free arm that draws as `DIRECT_ARM` does with each tree field drawn as one leaf of its
ontology tree at its ancestor-shrunk share (ADR `research/ancestor-shrunk-shares`)."""

HIER_0_ARM: Final = "bko-hier-0"
"""The free arm that draws as `DIRECT_ARM` does with each `is_a` pair's broad concept drawn given
its narrow one at the data's conditional, pseudo-count 0 (ADR `research/bkb-direct-generation`,
Decision 12)."""

HIER_W_ARM: Final = "bko-hier-w"
"""As `HIER_0_ARM`, the conditional shrunk toward 1 at pseudo-count `GROUP_WEIGHT`."""

HIER_CLOSED_ARM: Final = "bko-hier-closed"
"""As `HIER_0_ARM`, the conditional set to 1, so each pair's closure is forced: an ablation arm."""

COHERENT_ARM: Final = "bko-coherent"
"""The free arm that draws as `DIRECT_ARM` does, then redraws its free fields by sequential CART
given the cell, the label and the mapped fields (`bko.evaluate.coherent`; ADR
`research/free-field-synthesis`)."""

DONOR_ARM: Final = "bko-donor"
"""The free arm that draws `NOMAP_ARM`'s record with each mapping group copied whole by the pooled
rule: a cell record with probability n/(n + `GROUP_WEIGHT`), else a label-group record (ADR
`research/matched-knowledge-controls`)."""

POOL_RANDOM_ARM: Final = "bko-pool-random"
"""As `DONOR_ARM`, the pooled rule on as many fields as the mapping's groups hold, drawn at random
once per fill: the arm `DIRECT_ARM` is read against for the mapping's choice of fields."""

POOL_ALL_ARM: Final = "bko-pool-all"
"""As `DONOR_ARM`, the pooled rule on every feature but `age` and `sex`."""

KB_ARM: Final = "bko-kb"
"""The free arm that draws from K alone, copying no training value: presence as `LOGIC_ARM` draws
it, each mapping group's codes uniform over the codebook tuples showing the drawn pattern, and
`age` inside the training split's range (`bko.generate.knowledge_draw`; ADR
`research/matched-knowledge-controls`)."""

DIRECT_ARMS: Final[Mapping[str, DrawSpec]] = MappingProxyType(
    {
        DIRECT_ARM: DEFAULT_SPEC,
        LOGIC_ARM: DrawSpec(presence=Presence.LOGIC),
        PRAM_ARM: DEFAULT_SPEC,
        NOMAP_ARM: DrawSpec(mapped=False),
        CAPPED_ARM: DrawSpec(min_donors=2, reuse_cap=1),
        MICRO_ARM: DrawSpec(min_donors=2, reuse_cap=1, microaggregate=3),
        TREE_ARM: DrawSpec(tree=True),
        HIER_0_ARM: DrawSpec(hierarchy=0.0),
        HIER_W_ARM: DrawSpec(hierarchy=float(GROUP_WEIGHT)),
        HIER_CLOSED_ARM: DrawSpec(closed=True),
        COHERENT_ARM: DEFAULT_SPEC,
        DONOR_ARM: DrawSpec(presence=Presence.POOLED),
        POOL_RANDOM_ARM: DrawSpec(presence=Presence.POOLED, pool=Pool.RANDOM),
        POOL_ALL_ARM: DrawSpec(presence=Presence.POOLED, pool=Pool.ALL),
        KB_ARM: DrawSpec(presence=Presence.LOGIC, donors=False),
    }
)
"""Every direct arm, with the spec its draw reads (`bko.generate.draw_spec.DrawSpec`)."""
