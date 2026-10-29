"""Join holdout slices' scores into the scores one run writes: `python -m bko.evaluate merge`."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from bko.evaluate.errors import EvaluationError, ScoresFormatError
from bko.evaluate.scores import Measure, Scope, Score, Source, parse, render


logger = logging.getLogger(__name__)

type _Keys = frozenset[tuple[str, Source, Measure, Scope]]


def merge(seeds: Sequence[int], paths: Sequence[Path]) -> str:
    """Join the scores files at `paths` into the bytes one run over `seeds` writes.

    Each file is read as `bko.evaluate.scores.parse` reads one, so a file with other columns is
    refused, and logged with its path, its lowest and highest holdout, its row count and its
    sha256. Every holdout must hold the same (arm, source, measure, scope) keys, as one run's
    holdouts do, so slices scored under different code or configuration are refused. The rows are
    written in the order of `seeds` under one header, each holdout's rows in the order its file
    holds them, and the result is logged with its sha256.

    Raises:
        EvaluationError: If no file is given, two files hold the same holdout, a holdout holds
            other keys than the first file's first holdout, or the files' holdouts are not
            exactly `seeds`.
        ScoresFormatError: If a file is not UTF-8 or not the CSV `score` writes.
        OSError: If a file cannot be read.
    """
    if not paths:
        msg = "merge needs at least one scores file"
        raise EvaluationError(msg)
    owner: dict[int, Path] = {}
    scores: list[Score] = []
    reference: tuple[int, Path, _Keys] | None = None
    for path in paths:
        data = path.read_bytes()
        try:
            read = parse(data.decode("utf-8"))
        except UnicodeDecodeError as exc:
            msg = f"{path} is not UTF-8: {exc}"
            raise ScoresFormatError(msg) from exc
        held = sorted({s.holdout for s in read})
        logger.info(
            "merge file=%s holdouts=%s rows=%d sha256=%s",
            path,
            f"{held[0]}-{held[-1]}" if held else "none",
            len(read),
            hashlib.sha256(data).hexdigest(),
        )
        for holdout in held:
            if holdout in owner:
                msg = f"holdout {holdout} is in both {owner[holdout]} and {path}"
                raise EvaluationError(msg)
            owner[holdout] = path
            keys = frozenset(
                (s.arm, s.source, s.measure, s.scope) for s in read if s.holdout == holdout
            )
            if reference is None:
                reference = (holdout, path, keys)
            elif keys != reference[2]:
                msg = (
                    f"holdout {holdout} of {path} holds other scores than holdout {reference[0]} "
                    f"of {reference[1]}, so the files are not slices of one run"
                )
                raise EvaluationError(msg)
        scores.extend(read)
    missing = sorted(set(seeds) - owner.keys())
    extra = sorted(owner.keys() - set(seeds))
    if missing or extra:
        msg = f"the files' holdouts are not the run's: missing {missing}, not the run's {extra}"
        raise EvaluationError(msg)
    position = {seed: index for index, seed in enumerate(seeds)}
    ordered = sorted(scores, key=lambda s: position[s.holdout])
    text = render(ordered)
    logger.info(
        "merge files=%d holdouts=%d rows=%d sha256=%s",
        len(paths),
        len(owner),
        len(ordered),
        hashlib.sha256(text.encode()).hexdigest(),
    )
    return text
