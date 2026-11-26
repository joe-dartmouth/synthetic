"""`python -m bko.evaluate contradictions RUN --rows PATH...`: contradictions in each fill's rows.

The added-rows basis of ADR `research/data-contradiction-handling` (Decision 2). `run_added`
reads a run's `rows` stores (`bko.evaluate.privacy_stores.read`), one per holdout slice, and for
each holdout and fill, gap and replica fills alike, `added_counts` counts the five kinds of
`bko.gaps.contradictions` and the three coherence rules of `bko.evaluate.perturb.rule_breaks` in
the fill's added rows. It writes one CSV row per holdout, fill and kind, and makes no API call.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from bko.baselines.request import Method, Stratum
from bko.datasets.cells import cell_keys
from bko.datasets.frames import FrameSpec
from bko.evaluate.data import Data
from bko.evaluate.experiment import (
    PRIOR_SHIFT_ARM,
    REAL_ARM,
    Experiment,
    gap_request,
    replica_request,
)
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.perturb import RULE_COLUMNS, RULES, rule_breaks
from bko.evaluate.privacy_stores import Store, checked, read, read_fills
from bko.gaps.contradictions import KINDS, survey
from bko.gaps.observe import in_cell
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.evaluate")

ADDED_COLUMNS: Final = ("holdout", "arm", "rows", "kind", "count")
"""The CSV's columns: one row per holdout, fill and kind."""

ADDED_KINDS: Final = (*KINDS, *RULES)
"""What each fill counts, in the order the CSV writes them."""

REPLICA: Final = "-replica"
"""The suffix of a replica fill's arm."""


def added_counts(
    added: pd.DataFrame,
    rows: Sequence[FieldMapping],
    ontologies: Mapping[str, Ontology],
    *,
    frame: FrameSpec,
    asked: Collection[Stratum] | None,
    name: str,
) -> dict[str, int | None]:
    """Each of `ADDED_KINDS` counted in one fill's `added` rows.

    `narrow-without-broad` counts the rows showing a narrow concept without its broad one, summed
    over the `is_a` pairs; `child-share-above-parent` and `all-show` count `survey`'s findings
    over the cells `added` holds rows in; `empty-cell` counts the cells of `asked` holding no row,
    None for a fill that answers no request (`asked` None); `code-overlap` counts the rows whose
    code lies in an unresolved overlap; each rule counts the rows breaking it, None where `added`
    lacks the rules' columns. A fill with no rows counts 0 on every other kind and every cell of
    `asked` as empty; its rules stay None off the rules' columns.

    Args:
        added: The fill's rows, in `frame`'s columns.
        rows: The field mappings.
        ontologies: The loaded releases, by concept prefix.
        frame: The rows' frame.
        asked: The cells of the request the fill answered, None if it answers none.
        name: The rows' name, for an error.

    Raises:
        GapError: As `survey`.
        ContractError: As `survey`.
        ValueError: As `survey`.
    """
    coded = set(RULE_COLUMNS) <= set(added.columns)
    if not len(added):
        empty = None if asked is None else len(asked)
        found = {**dict.fromkeys(KINDS, 0), "empty-cell": empty}
        return found | dict.fromkeys(RULES, 0 if coded else None)
    findings = survey(added, rows, ontologies, frame=frame, name=name).findings
    overlapping = pd.Series(False, index=added.index)
    for finding in findings:
        if finding.kind == "code-overlap":
            overlapping |= added[finding.field].isin(finding.codes)
    keys = cell_keys(added, edges=frame.edges)
    empty = None if asked is None else sum(not in_cell(keys, cell).any() for cell in asked)
    found: dict[str, int | None] = {
        "narrow-without-broad": sum(
            finding.without_parent or 0
            for finding in findings
            if finding.kind == "narrow-without-broad"
        ),
        "child-share-above-parent": sum(f.kind == "child-share-above-parent" for f in findings),
        "empty-cell": empty,
        "all-show": sum(f.kind == "all-show" for f in findings),
        "code-overlap": int(overlapping.sum()),
    }
    if not coded:
        return found | dict.fromkeys(RULES)
    return found | {rule: int(mask.sum()) for rule, mask in rule_breaks(added).items()}


def run_added(
    run: str,
    stores: Sequence[Path],
    *,
    data: Data,
    ontologies: Mapping[str, Ontology],
    experiment: Experiment,
    runs: Mapping[str, NamedRun],
    root: Path,
) -> str:
    """Count `ADDED_KINDS` in every fill of `run`'s `stores`; return the CSV.

    Fills come per holdout in the experiment's order, the arms' fills in order, then the replica
    arms'. It logs `added run=%s stores=%d holdouts=%d root=%s`, `read`'s `added store` lines,
    an `added holdout=%d arm=%s rows=0 asked=%d` line per fill that answers a request with no
    row, an `added holdout=%d done=%d/%d seconds=%.1f` line per holdout and, last,
    `added rows=%d sha256=%s stores=%d holdouts=%d`.

    Raises:
        IdentityError: As `read` and `read_fills`, naming the store, or if a holdout's `split`
            line is not the one its data prepares.
        EvaluationError: As `gap_request`, if a holdout with a gap fill that answers the request
            flags no cell.
        GapError: As `added_counts`.
    """
    begun = time.perf_counter()
    total = len(experiment.holdout_seeds)
    logger.info("added run=%s stores=%d holdouts=%d root=%s", run, len(stores), total, root)
    held = read(run, stores, data, experiment=experiment, runs=runs, root=root, verb="added")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(ADDED_COLUMNS)
    written = 0
    for done, seed in enumerate(experiment.holdout_seeds, start=1):
        for arm, added, counts in _holdout(seed, held[seed], data, experiment, ontologies):
            for kind, count in counts.items():
                writer.writerow((seed, arm, len(added), kind, "" if count is None else count))
                written += 1
        logger.info(
            "added holdout=%d done=%d/%d seconds=%.1f",
            seed,
            done,
            total,
            time.perf_counter() - begun,
        )
    text = out.getvalue()
    sha = hashlib.sha256(text.encode()).hexdigest()
    logger.info("added rows=%d sha256=%s stores=%d holdouts=%d", written, sha, len(stores), total)
    return text


def answers(arm: str) -> bool:
    """Whether the gap fill of `arm` answers the gap request, so its `empty-cell` counts.

    `REAL_ARM` adds the removed rows, and `PRIOR_SHIFT_ARM` and the methods that add no rows
    (`none`, `class-weight`) add none, so none of them answers it.
    """
    if arm in {REAL_ARM, PRIOR_SHIFT_ARM}:
        return False
    return arm not in Method or Method(arm).adds_rows


def _holdout(
    seed: int,
    store: Store,
    data: Data,
    experiment: Experiment,
    ontologies: Mapping[str, Ontology],
) -> list[tuple[str, pd.DataFrame, dict[str, int | None]]]:
    """Each fill of holdout `seed` in `store`: its arm, its rows and their counts.

    A gap fill that `answers` counts its `empty-cell` against the gap request, a replica fill
    against the replica request; any other fill's is None.

    Raises:
        IdentityError: As `checked`, or as `read_fills`, naming the store.
        EvaluationError: As `gap_request`.
    """
    frame = data.frame
    train = checked(seed, store, data, experiment).ablated.train
    listed = [
        *((arm, False) for arm in experiment.arms),
        *((f"{arm}{REPLICA}", True) for arm in experiment.replica_arms),
    ]
    held = read_fills(seed, store, [arm for arm, _ in listed], frame)
    asked: dict[bool, tuple[Stratum, ...]] = {}
    found = []
    for arm, replica in listed:
        added, cells = held[arm], None
        if replica or answers(arm):
            if replica not in asked:
                request = (
                    replica_request(train, experiment, seed=seed, frame=frame)
                    if replica
                    else gap_request(
                        train,
                        experiment,
                        seed=seed,
                        frame=frame,
                        rows=data.field_rows,
                        name=data.name,
                    )
                )
                asked[replica] = tuple(request.quota)
            cells = asked[replica]
            if not len(added):
                logger.info("added holdout=%d arm=%s rows=0 asked=%d", seed, arm, len(cells))
        counts = added_counts(
            added, data.field_rows, ontologies, frame=frame, asked=cells, name=f"{arm}/{seed}"
        )
        found.append((arm, added, counts))
    return found
