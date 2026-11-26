"""What a direct draw reads, as one value: a light module the run table imports.

`bko.generate.direct.sample` takes a `DrawSpec`. Its default is `bko-direct`'s draw; each other
field value names an ablation or a variant (ADRs `research/logic-only-ablation`,
`research/no-mapping-ablation`, `research/sampler-donor-limits`, `research/ancestor-shrunk-shares`,
`research/bkb-direct-generation`, Decision 12, and `research/matched-knowledge-controls`).
`params` and `tokens` hold only the fields that differ from their defaults, so the default spec's
sampler lines and `arm` params keep their bytes; `values` holds every field, so
`attribution spec=` gains each field the spec gains.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from dataclasses import dataclass
from typing import Final

from bko.generate.presence import Presence


class Pool(enum.StrEnum):
    """Which fields a `Presence.POOLED` draw copies by the pooled rule (`bko.generate.pooling`)."""

    MAPPED = "mapped"
    """Each mapping group, whole."""

    RANDOM = "random"
    """As many fields as the mapping's groups hold, drawn at random once per fill."""

    ALL = "all"
    """Every feature but `age` and `sex`."""


@dataclass(frozen=True, slots=True)
class DrawSpec:
    """How a direct draw decides presence, reads the mapping table, picks donors and codes.

    Attributes:
        presence: How each concept's presence is drawn.
        mapped: Whether the draw reads the mapping table; without it there is no concept and
            every column but `age`, `sex` and the label is a free field.
        min_donors: The fewest cell records showing a unit's drawn pattern that the unit may be
            copied from; below it the unit reads the label group less the cell's records.
        reuse_cap: The most times one record may give one unit's values in a fill attempt; None
            sets no cap.
        tree: Whether each tree field (`bko.generate.tree.tree_fields`) is drawn as one leaf of
            its ontology tree at its ancestor-shrunk share, from a donor of that column alone.
        hierarchy: The pseudo-count that shrinks each `is_a` pair's conditional toward 1, the broad
            concept drawn given its narrow one (`bko.generate.hierarchy`); None draws each concept
            at its share.
        closed: Whether each `is_a` pair's broad concept is drawn present whenever its narrow
            one is, the conditional set to 1.
        pool: Which fields a `Presence.POOLED` draw pools; off its default it needs that presence.
        donors: Whether the draw copies fields from training records; false draws every field
            from the mapping table and the codebook alone (`bko.generate.knowledge_draw`) and
            needs `Presence.LOGIC`.
        microaggregate: How many distinct donors of one key class each numeric field but `age`
            is averaged over (`bko.generate.micro`); None copies it from one donor.
    """

    presence: Presence = Presence.SHARES
    mapped: bool = True
    min_donors: int = 1
    reuse_cap: int | None = None
    tree: bool = False
    hierarchy: float | None = None
    closed: bool = False
    pool: Pool = Pool.MAPPED
    donors: bool = True
    microaggregate: int | None = None

    def __post_init__(self) -> None:
        """Refuse a field combination the draw cannot read.

        Raises:
            ValueError: If `presence` is not `Presence.SHARES` and `mapped` is false (pooled
                reads rows through `pool` alone, so it keeps `mapped` true); if a limit
                is below 1, or `microaggregate` below 2; if `hierarchy` is not a finite
                non-negative float, or is set beside `closed`; if `tree` is set beside
                `hierarchy` or `closed`; if a field of `_VARIANTS` is off its default without the
                mapping table or at a presence other than shares; if `pool` is off its default at
                a presence other than pooled; or if `donors` is false at a presence other than
                logic.
        """
        if self.presence is Presence.POOLED and not self.mapped:
            msg = "presence pooled keeps mapped true: pool, not mapped, says which rows it reads"
            raise ValueError(msg)
        if self.presence is not Presence.SHARES and not self.mapped:
            msg = f"presence {self.presence.value} has no meaning without the mapping table"
            raise ValueError(msg)
        for name, value in (("min_donors", self.min_donors), ("reuse_cap", self.reuse_cap)):
            if value is not None and value < 1:
                msg = f"{name} {value} must be at least 1"
                raise ValueError(msg)
        if self.microaggregate is not None and self.microaggregate < 2:
            msg = f"microaggregate {self.microaggregate} must be at least 2"
            raise ValueError(msg)
        pseudo = self.hierarchy
        if pseudo is not None and (
            type(pseudo) is not float or not math.isfinite(pseudo) or pseudo < 0
        ):
            msg = f"hierarchy {pseudo!r} must be a finite non-negative float"
            raise ValueError(msg)
        if pseudo is not None and self.closed:
            msg = "closed sets the conditional to 1, so it takes no hierarchy pseudo-count"
            raise ValueError(msg)
        if self.tree and self.conditional:
            msg = "tree draws each tree field as one leaf, so it takes no conditional is_a draw"
            raise ValueError(msg)
        defaults = {field.name: field.default for field in dataclasses.fields(self)}
        changed = [name for name in _VARIANTS if getattr(self, name) != defaults[name]]
        if changed and (self.presence is not Presence.SHARES or not self.mapped):
            msg = f"{', '.join(changed)} needs the mapping table and presence shares"
            raise ValueError(msg)
        if self.pool is not Pool.MAPPED and self.presence is not Presence.POOLED:
            msg = f"pool {self.pool.value} needs presence pooled"
            raise ValueError(msg)
        if not self.donors and self.presence is not Presence.LOGIC:
            msg = "donors false needs presence logic"
            raise ValueError(msg)

    @property
    def limited(self) -> bool:
        """Whether the draw sets a donor floor above 1 or a reuse cap."""
        return self.min_donors > 1 or self.reuse_cap is not None

    @property
    def reads_concepts(self) -> bool:
        """Whether the draw reads concepts from the mapping table: mapped, and not pooled."""
        return self.mapped and self.presence is not Presence.POOLED

    @property
    def conditional(self) -> bool:
        """Whether the draw reads each `is_a` pair's conditional: `hierarchy` set, or `closed`."""
        return self.hierarchy is not None or self.closed

    def values(self) -> dict[str, object]:
        """Every field, an enum as its value, in field order: what `params` logs it as."""
        return {field.name: _plain(getattr(self, field.name)) for field in dataclasses.fields(self)}

    def params(self) -> dict[str, object]:
        """Each field that differs from its default, an enum as its value, in field order."""
        return {name: _plain(value) for name, value in self._changed()}

    def tokens(self) -> str:
        """A ` name=value` token per changed field but `presence`, a bool in lower case.

        The sampler's lines name `presence` in their own field, after these tokens.
        """
        return "".join(
            f" {name}={_text(value)}" for name, value in self._changed() if name != "presence"
        )

    def _changed(self) -> list[tuple[str, object]]:
        default = type(self)()
        return [
            (field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
            if getattr(self, field.name) != getattr(default, field.name)
        ]


_VARIANTS: Final = ("min_donors", "reuse_cap", "tree", "hierarchy", "closed", "microaggregate")
"""The fields that need `mapped=True` and `Presence.SHARES` off their defaults."""


def _plain(value: object) -> object:
    return value.value if isinstance(value, enum.Enum) else value


def _text(value: object) -> str:
    plain = _plain(value)
    return str(plain).lower() if isinstance(plain, bool) else str(plain)


DEFAULT_SPEC: Final[DrawSpec] = DrawSpec()
"""`bko-direct`'s draw: concepts at their shares, read through the mapping table."""
