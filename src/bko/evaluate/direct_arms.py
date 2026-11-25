"""The direct arms by name, each with the draw spec it reads: a light module the run table imports.

`bko.evaluate.experiment` re-exports every name here but `CAPPED_ARM`; `CAPPED_ARM` and any arm
added later are imported from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

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

DIRECT_ARMS: Final[Mapping[str, DrawSpec]] = MappingProxyType(
    {
        DIRECT_ARM: DEFAULT_SPEC,
        LOGIC_ARM: DrawSpec(presence=Presence.LOGIC),
        PRAM_ARM: DEFAULT_SPEC,
        NOMAP_ARM: DrawSpec(mapped=False),
        CAPPED_ARM: DrawSpec(min_donors=2, reuse_cap=1),
    }
)
"""Every direct arm, with the spec its draw reads (`bko.generate.draw_spec.DrawSpec`)."""
