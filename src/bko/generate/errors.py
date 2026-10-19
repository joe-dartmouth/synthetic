"""The errors generation raises: a call with no readable response, and a malformed file."""

from __future__ import annotations

from typing import ClassVar

from bko.datasets.errors import ContractError


class GenerationError(Exception):
    """A call failed before any response arrived, so it owes no records.

    Attributes:
        exit_code: The command line's exit status for this failure.
        attempts: The HTTP requests the SDK made for the call, or None when nothing counted them.
    """

    exit_code: ClassVar[int] = 2

    def __init__(self, *args: object, attempts: int | None = None) -> None:
        """Keep the message and the attempt count."""
        super().__init__(*args)
        self.attempts = attempts


class CandidatesFormatError(ContractError):
    """A candidates file is not the JSON lines `python -m bko.generate` writes. Exit 3."""


class ReplayError(ContractError):
    """A rerun's calls do not match the call log it replays. Exit 3."""
