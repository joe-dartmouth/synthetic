"""What `bko.baselines` raises when a baseline cannot be built.

The exit codes follow `bko.datasets.errors`: 2 the command could not run, 3 the output contradicts
its contract.
"""

from __future__ import annotations

from typing import ClassVar


class BaselineError(Exception):
    """A baseline could not be built: an infeasible request, or a library that refused it."""

    exit_code: ClassVar[int] = 2


class OutputContractError(BaselineError):
    """A method's output breaks the contract `bko.baselines.records.Baseline` documents."""

    exit_code: ClassVar[int] = 3
