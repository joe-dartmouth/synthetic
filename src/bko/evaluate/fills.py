"""The holdout and the fill a runner call serves, for log attribution only.

The harness sets `CURRENT_FILL` around each runner call (`bko.evaluate.run`), so an arm can name
both in its log lines without a parameter on `Runner`. Nothing a score reads depends on it. An arm
reads it synchronously when called, as `bko.evaluate.direct` and `bko.evaluate.skeleton` do: a
task on an event loop started earlier, or a thread pool, would not see the value.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, Literal


type FillKind = Literal["gap", "replica"]
"""`gap` for the gap request, `replica` for the replica request."""


@dataclass(frozen=True, slots=True)
class Fill:
    """The holdout and the fill a runner call serves, for log attribution only.

    Attributes:
        holdout: The holdout's seed.
        kind: `gap` for the gap request, `replica` for the replica request.
    """

    holdout: int
    kind: FillKind


CURRENT_FILL: Final[ContextVar[Fill | None]] = ContextVar("current_fill", default=None)
"""The fill the running runner call serves; None outside a harness fill."""
