"""The errors generation raises: a call with no readable response, and a malformed file."""

from __future__ import annotations

from typing import ClassVar

from bko.datasets.errors import ContractError


class GenerationError(Exception):
    """A call failed before any response arrived, so it owes no records.

    Attributes:
        exit_code: The command line's exit status for this failure.
    """

    exit_code: ClassVar[int] = 2


class CandidatesFormatError(ContractError):
    """A candidates file is not the JSON lines `python -m bko.generate` writes. Exit 3."""
