"""Why `python -m bko.evaluate` stops; the exit codes follow `bko.datasets.errors`.

2 means the command could not run and 3 that a scores file breaks its format. Exit 1 is left to
Python, so a traceback still means a defect.
"""

from __future__ import annotations

from typing import ClassVar


class EvaluationError(Exception):
    """The harness could not score or judge the arms it was given."""

    exit_code: ClassVar[int] = 2


class ScoresFormatError(EvaluationError):
    """A scores file is not the CSV `python -m bko.evaluate score` writes."""

    exit_code: ClassVar[int] = 3
