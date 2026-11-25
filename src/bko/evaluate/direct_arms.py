"""The direct arms by name, each with the draw spec it reads: a light module the run table imports.

`bko.evaluate.experiment` re-exports every name here but `CAPPED_ARM`, `TREE_ARM` and the three
`bko-hier` arms; those and any arm added later are imported from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.gaps.shares import GROUP_WEIGHT
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
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

DIRECT_ARMS: Final[Mapping[str, DrawSpec]] = MappingProxyType(
    {
        DIRECT_ARM: DEFAULT_SPEC,
        LOGIC_ARM: DrawSpec(presence=Presence.LOGIC),
        PRAM_ARM: DEFAULT_SPEC,
        NOMAP_ARM: DrawSpec(mapped=False),
        CAPPED_ARM: DrawSpec(min_donors=2, reuse_cap=1),
        TREE_ARM: DrawSpec(tree=True),
        HIER_0_ARM: DrawSpec(hierarchy=0.0),
        HIER_W_ARM: DrawSpec(hierarchy=float(GROUP_WEIGHT)),
        HIER_CLOSED_ARM: DrawSpec(closed=True),
    }
)
"""Every direct arm, with the spec its draw reads (`bko.generate.draw_spec.DrawSpec`)."""
