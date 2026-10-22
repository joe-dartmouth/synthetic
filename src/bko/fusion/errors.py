"""The failure `bko.fusion` adds to `bko.datasets.errors`; the exit codes follow it."""

from __future__ import annotations

from bko.datasets.errors import ContractError


class ClosureError(ContractError):
    """A candidates file does not answer the completeness request.

    Its header names another request, UCI archive, split or releases; its plan's per-cell sums
    differ from the request; or, read call by call, a line belongs to an unplanned call or names
    another cell than its call's plan, or a planned call wrote fewer lines than it asked, as a run
    that stopped early leaves it. Exit 3: the input contradicts its contract.
    """
