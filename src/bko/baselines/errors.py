"""What `bko.baselines` raises when a baseline cannot be built.

The exit codes follow `bko.datasets.errors`: 2 the command could not run, 3 the output contradicts
its contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar


if TYPE_CHECKING:
    import pandas as pd

    from bko.baselines.records import Baseline


class BaselineError(Exception):
    """A baseline could not be built: an infeasible request, or a library that refused it."""

    exit_code: ClassVar[int] = 2


class OutputContractError(BaselineError):
    """A method's output breaks the contract `bko.baselines.records.Baseline` documents."""

    exit_code: ClassVar[int] = 3


class ShortfallError(BaselineError):
    """A generator drew its row limit with strata still short; it holds what it kept.

    Attributes:
        short: Each short stratum's missing rows, keyed `<band>/<sex>/<label>`.
        added: The rows kept, numbered as a `Baseline`'s added rows.
        baseline: The rows kept as the method's whole output, once the method has built it; a
            harness may score it as a partial fill on its last attempt.
    """

    def __init__(
        self,
        message: str,
        short: Mapping[str, int],
        added: pd.DataFrame,
        baseline: Baseline | None = None,
    ) -> None:
        """Hold the message, the shortfall and the kept rows."""
        super().__init__(message)
        self.short = short
        self.added = added
        self.baseline = baseline
