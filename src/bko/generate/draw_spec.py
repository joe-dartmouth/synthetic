"""What a direct draw reads, as one value: a light module the run table imports.

`bko.generate.direct.sample` takes a `DrawSpec`. Its default is `bko-direct`'s draw; each other
field value names an ablation (ADRs `research/logic-only-ablation` and
`research/no-mapping-ablation`). Only fields that differ from their defaults are logged, so the
default spec's lines and params keep their bytes.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import Final

from bko.generate.presence import Presence


@dataclass(frozen=True, slots=True)
class DrawSpec:
    """How a direct draw decides presence and whether it reads the mapping table.

    Attributes:
        presence: How each concept's presence is drawn.
        mapped: Whether the draw reads the mapping table; without it there is no concept and
            every column but `age`, `sex` and the label is a free field.
    """

    presence: Presence = Presence.SHARES
    mapped: bool = True

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
        return [
            (field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
            if getattr(self, field.name) != field.default
        ]


def _plain(value: object) -> object:
    return value.value if isinstance(value, enum.Enum) else value


def _text(value: object) -> str:
    plain = _plain(value)
    return str(plain).lower() if isinstance(plain, bool) else str(plain)


DEFAULT_SPEC: Final[DrawSpec] = DrawSpec()
"""`bko-direct`'s draw: concepts at their shares, read through the mapping table."""
