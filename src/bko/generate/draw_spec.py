"""What a direct draw reads, as one value: a light module the run table imports.

`bko.generate.direct.sample` takes a `DrawSpec`. Its default is `bko-direct`'s draw; each other
field value names an ablation or a variant (ADRs `research/logic-only-ablation`,
`research/no-mapping-ablation`, `research/sampler-donor-limits` and
`research/ancestor-shrunk-shares`). Only fields that differ from
their defaults are logged, so the default spec's lines and params keep their bytes.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import Final

from bko.generate.presence import Presence


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
    """

    presence: Presence = Presence.SHARES
    mapped: bool = True
    min_donors: int = 1
    reuse_cap: int | None = None
    tree: bool = False

    def __post_init__(self) -> None:
        """Refuse a field combination the draw cannot read.

        Raises:
            ValueError: If `presence` is not `Presence.SHARES` and `mapped` is false; if a limit
                is below 1; or if a field of `_VARIANTS` is off its default without the mapping
                table or at a presence other than shares.
        """
        if self.presence is not Presence.SHARES and not self.mapped:
            msg = f"presence {self.presence.value} has no meaning without the mapping table"
            raise ValueError(msg)
        for name, value in (("min_donors", self.min_donors), ("reuse_cap", self.reuse_cap)):
            if value is not None and value < 1:
                msg = f"{name} {value} must be at least 1"
                raise ValueError(msg)
        defaults = {field.name: field.default for field in dataclasses.fields(self)}
        changed = [name for name in _VARIANTS if getattr(self, name) != defaults[name]]
        if changed and (self.presence is not Presence.SHARES or not self.mapped):
            msg = f"{', '.join(changed)} needs the mapping table and presence shares"
            raise ValueError(msg)

    @property
    def limited(self) -> bool:
        """Whether the draw sets a donor floor above 1 or a reuse cap."""
        return self.min_donors > 1 or self.reuse_cap is not None

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


_VARIANTS: Final = ("min_donors", "reuse_cap", "tree")
"""The fields that need `mapped=True` and `Presence.SHARES` off their defaults."""


def _plain(value: object) -> object:
    return value.value if isinstance(value, enum.Enum) else value


def _text(value: object) -> str:
    plain = _plain(value)
    return str(plain).lower() if isinstance(plain, bool) else str(plain)


DEFAULT_SPEC: Final[DrawSpec] = DrawSpec()
"""`bko-direct`'s draw: concepts at their shares, read through the mapping table."""
