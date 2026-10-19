"""The failure `bko.fusion` adds to `bko.datasets.errors`; the exit codes follow it."""

from __future__ import annotations

from bko.datasets.errors import ContractError


class ClosureError(ContractError):
    """A candidates file does not answer the completeness request.

    It names a cell the request does not ask for, or holds fewer lines for a requested cell than
    its rows, as a run that stopped early leaves it. Exit 3: the input contradicts its contract.
    """
