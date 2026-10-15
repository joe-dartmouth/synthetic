"""The failures `bko.datasets` reports, each carrying the CLI exit code it maps to."""

from __future__ import annotations

from typing import ClassVar


class DatasetError(Exception):
    """A dataset command could not run: a bad request, or an input it cannot obtain."""

    exit_code: ClassVar[int] = 2


class UnavailableError(DatasetError):
    """An input could not be obtained: no network, an HTTP error, or no snapshot on disk."""


class ContractError(DatasetError):
    """The bytes obtained contradict their pin or their documented schema."""

    exit_code: ClassVar[int] = 1
