"""The errors generation raises: a call with no response, a malformed file, a refused draw."""

from __future__ import annotations

from typing import ClassVar

from bko.datasets.errors import ContractError, DatasetError


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
    """A rerun's calls do not match the call log it replays, or a call log a resume cannot continue.

    Exit 3.
    """


class SamplingError(DatasetError):
    """The direct sampler found no training record to read a drawn field from. Exit 2."""
