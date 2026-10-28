"""The errors generation raises: no reply, a malformed file, a crossed wave, a refused draw."""

from __future__ import annotations

from typing import ClassVar

from bko.datasets.errors import ContractError, DatasetError


class GenerationError(Exception):
    """A call failed before any response arrived, so it owes no records.

    Attributes:
        exit_code: The command line's exit status for this failure.
        attempts: The HTTP requests the SDK made for the call, or None when nothing counted them.
        retryable: Whether the API caused it, so a resume may succeed: a connection error, a
            timeout, a 408, 409 or 429, or a status of 500 or above (`bko.generate.api`). False
            otherwise.
        rate_limited: Whether any retry of the call got status 429, or the call failed on one.
    """

    exit_code: ClassVar[int] = 2

    def __init__(
        self,
        *args: object,
        attempts: int | None = None,
        retryable: bool = False,
        rate_limited: bool = False,
    ) -> None:
        """Keep the message, the attempt count and what the API caused."""
        super().__init__(*args)
        self.attempts = attempts
        self.retryable = retryable
        self.rate_limited = rate_limited


class IdleError(GenerationError):
    """A call got no reply within the run's idle limit, its retries included. Exit 2.

    `bko.generate.flight` raises it; the call log's `end` line names `stop_rule` under `stop`. It
    is `retryable` unless told otherwise: a stalled call is the API's to retry.

    Attributes:
        stop_rule: The stop rule that fired, `idle`.
    """

    stop_rule: ClassVar[str] = "idle"

    def __init__(
        self,
        *args: object,
        attempts: int | None = None,
        retryable: bool = True,
        rate_limited: bool = False,
    ) -> None:
        """Keep the message, the attempt count and what the API caused; `retryable` by default."""
        super().__init__(*args, attempts=attempts, retryable=retryable, rate_limited=rate_limited)


class CandidatesFormatError(ContractError):
    """A candidates file is not the JSON lines `python -m bko.generate` writes. Exit 3."""


class ReplayError(ContractError):
    """A rerun's calls do not match the call log it replays, or a call log a resume cannot continue.

    Exit 3.
    """


class WaveError(ContractError):
    """A reply kept a record in a cell a later call of its wave already asked for. Exit 3.

    Structured outputs pin each record's sex and label, so this means a reply crossed them, and
    the wave no longer asks what a one-call-at-a-time run asks (ADR
    `research/concurrent-call-waves`).
    """


class SamplingError(DatasetError):
    """The direct sampler found no training record to read a drawn field from. Exit 2."""
