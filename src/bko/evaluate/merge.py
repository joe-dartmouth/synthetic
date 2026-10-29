"""Join holdout slices' scores into the scores one run writes: `python -m bko.evaluate merge`."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from bko.evaluate.errors import EvaluationError
from bko.evaluate.scores import Score, parse, render


logger = logging.getLogger(__name__)


def merge(seeds: Sequence[int], paths: Sequence[Path]) -> str:
    """Join the scores files at `paths` into the bytes one run over `seeds` writes.

    Each file is read as `bko.evaluate.scores.parse` reads one, so a file with other columns is
    refused. The rows are written in seed order under one header, each holdout's rows in the order
    its file holds them.

    Raises:
        EvaluationError: If no file is given, two files hold the same holdout, or the files'
            holdouts are not exactly `seeds`.
        ScoresFormatError: If a file is not the CSV `score` writes.
        OSError: If a file cannot be read.
    """
    if not paths:
        msg = "merge needs at least one scores file"
        raise EvaluationError(msg)
    owner: dict[int, Path] = {}
    scores: list[Score] = []
    for path in paths:
        read = parse(path.read_text(encoding="utf-8"))
        for holdout in sorted({s.holdout for s in read}):
            if holdout in owner:
                msg = f"holdout {holdout} is in both {owner[holdout]} and {path}"
                raise EvaluationError(msg)
            owner[holdout] = path
        scores.extend(read)
    missing = sorted(set(seeds) - owner.keys())
    extra = sorted(owner.keys() - set(seeds))
    if missing or extra:
        msg = f"the files' holdouts are not the run's: missing {missing}, not the run's {extra}"
        raise EvaluationError(msg)
    ordered = sorted(scores, key=lambda s: s.holdout)
    text = render(ordered)
    logger.info("merge files=%d holdouts=%d rows=%d", len(paths), len(owner), len(ordered))
    return text
