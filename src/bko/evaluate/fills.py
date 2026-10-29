"""The holdout and the fill a runner call serves, for log attribution only.

The harness sets `CURRENT_FILL` around each runner call (`bko.evaluate.run`), so an arm can name
both in its log lines without a parameter on `Runner`. Nothing a score reads depends on it.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True, slots=True)
class Fill:
    """The holdout and the fill a runner call serves, for log attribution only.

    Attributes:
        holdout: The holdout's seed.
        kind: `gap` for the gap request, `replica` for the replica request.
    """

    holdout: int
    kind: Literal["gap", "replica"]


CURRENT_FILL: Final[ContextVar[Fill | None]] = ContextVar("current_fill", default=None)
"""The fill the running runner call serves; None outside a harness fill."""
