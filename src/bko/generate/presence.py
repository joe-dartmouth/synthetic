"""How the direct draw decides each concept's presence: a light module the run table imports.

`bko.generate.direct.sample` takes a `Presence`; `bko.generate.logic` holds the logic-only draw.
"""

from __future__ import annotations

import enum


class Presence(enum.StrEnum):
    """How the direct draw decides each concept's presence."""

    SHARES = "shares"
    """At its share, read from the fused BKB's data priors (`bko.gaps.shares`)."""

    LOGIC = "logic"
    """Uniformly over its component's consistent assignments, with no prior
    (`bko.generate.logic`)."""
