"""The error generation raises when a call gets no response it can read."""

from __future__ import annotations

from typing import ClassVar


class GenerationError(Exception):
    """A call failed before any response arrived, so it owes no records.

    Attributes:
        exit_code: The command line's exit status for this failure.
    """

    exit_code: ClassVar[int] = 2
