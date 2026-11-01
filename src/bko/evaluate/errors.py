"""Why `python -m bko.evaluate` stops; the exit codes follow `bko.datasets.errors`.

2 means the command could not run, a stop rule included, and 3 that a scores or candidates file
breaks its format or a run folder fails a `shared` identity check. Exit 1
is left to Python, so a traceback still means a defect.
"""

from __future__ import annotations

from typing import ClassVar


class EvaluationError(Exception):
    """The harness could not score or judge the arms it was given."""

    exit_code: ClassVar[int] = 2


class ScoresFormatError(EvaluationError):
    """A scores file is not the CSV `python -m bko.evaluate score` writes."""

    exit_code: ClassVar[int] = 3


class CandidatesFormatError(EvaluationError):
    """A candidates file is not the JSONL `bko.evaluate.holdout_candidates` writes.

    `bko.evaluate.holdout_candidates.read` raises it.
    """

    exit_code: ClassVar[int] = 3


class IdentityError(EvaluationError):
    """A run folder `bko.evaluate.shared` rescores fails one of its identity checks."""

    exit_code: ClassVar[int] = 3


class StopRuleError(EvaluationError):
    """A paid run's stop rule fired: `ceiling`, `projection` or `idle`.

    The call log's `end` line records `stop_rule` under `stop` (`bko.generate.log`), so a resume
    can refuse a run the projection or the ceiling stopped, and `retryable` under `retryable`.

    Attributes:
        stop_rule: The rule that fired.
        retryable: Whether the API caused the stop, so a supervised run resumes it: an idle stop
            carries its `IdleError`'s; False otherwise.
    """

    def __init__(self, message: str, stop_rule: str, *, retryable: bool = False) -> None:
        """Hold the message, the rule and whether the API caused it."""
        super().__init__(message)
        self.stop_rule = stop_rule
        self.retryable = retryable
