"""The failures `bko.datasets` reports, each carrying the CLI exit code it maps to.

Exit 1 is left to Python, which uses it for an uncaught exception, so a defect never reads as a
contract failure.
"""

from __future__ import annotations

from typing import ClassVar


class DatasetError(Exception):
    """A dataset command could not run: a bad request, or an input it cannot obtain."""

    exit_code: ClassVar[int] = 2


class UnavailableError(DatasetError):
    """An input could not be obtained: no network, an HTTP error, or no snapshot on disk."""


class ContractError(DatasetError):
    """The bytes obtained contradict their pin or their documented schema."""

    exit_code: ClassVar[int] = 3
