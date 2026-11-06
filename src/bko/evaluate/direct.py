"""The direct arm: a holdout's request filled from the fused BKB's concept shares, with no API call.

A `DirectArm` is a `bko.baselines.methods.Runner`. Given a holdout's training rows and a request,
the gap request or the replica request alike, it draws every requested row once with
`bko.generate.direct.sample`, seeded by the request's seed, in quota order. Every record goes
through the same `bko.evaluate.attempts.Attempt` as a BKO arm's candidates: validated, dropped
where `check` would refuse it, and counted toward the cell its own values name. A drawn record
copies training values into its own cell, so none is dropped; an attempt that leaves a cell short
anyway, or whose draw no training record matches, refuses with a plain `BaselineError`, which the
harness retries with a derived seed.
The arm keeps its validity counts itself and writes every record it drew, with its outcome and a
null call, to the run's candidates file when the harness drains it
(`bko.evaluate.holdout_candidates`). `run_direct` runs a run's arms with its direct arms among
them, one per name of `DIRECT_ARMS`, each drawing presence its own way. ADR
`research/bkb-direct-generation` records the sampler, and ADR `research/logic-only-ablation` the
logic-only draw of `bko-logic`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, check, unit_weights
from bko.baselines.request import Request, Strata
from bko.datasets.frames import FrameSpec
from bko.evaluate.attempts import Attempt, Tally
from bko.evaluate.classifier import Factory, Model
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARMS, Experiment
from bko.evaluate.guided import Knowledge
from bko.evaluate.holdout_candidates import FILLS, Candidate, CandidateFile, Seen
from bko.evaluate.run import Drain, run
from bko.evaluate.scores import Measure, Score, Source
from bko.generate.direct import sample
from bko.generate.errors import SamplingError
from bko.generate.logic import Presence
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology


logger = logging.getLogger(__name__)


class DirectArm:
    """The direct arm: a `Runner` that keeps its validity counts over every attempt it makes.

    Attributes:
        name: The arm's name.
        totals: Every attempt's counts over the run.
    """

    def __init__(
        self,
        rows: Sequence[FieldMapping],
        ontologies: Mapping[str, Ontology],
        *,
        name: str,
        budget: int,
        sink: CandidateFile | None,
        dataset: str,
        presence: Presence = Presence.SHARES,
    ) -> None:
        """Hold what each attempt needs; nothing is drawn until the arm runs.

        Args:
            rows: The field mappings.
            ontologies: The loaded ontologies, by concept prefix, for each draw's closure.
            name: The arm's name.
            budget: The records an attempt may draw, as a multiple of the request's rows; one draw
                of the request's rows always fits it.
            sink: The run's candidates file; None keeps no lines.
            dataset: The dataset's source name; each cell's fragment takes it.
            presence: How each draw decides a concept's presence (`bko.generate.direct.sample`).
        """
        self.name = name
        self.totals = Tally()
        self._rows = tuple(rows)
        self._ontologies = ontologies
        self._budget = budget
        self._sink = sink
        self._dataset = dataset
        self._presence = presence
        self._pending = Tally()
        self._attempts = 0
        self._seen: list[tuple[int, bool, Seen]] = []

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Fill `request` from `train`'s concept shares, seeded by the request's seed.

        Raises:
            BaselineError: Plain, when a cell is left short or no training record matches a draw.
            EvaluationError: If the request is not by cell.
        """
        if request.strata is not Strata.CELL:
            msg = f"{self.name} fills cells only, not {request.strata}"
            raise EvaluationError(msg)
        attempt = Attempt(train, request, self._budget * request.rows)
        fill = CURRENT_FILL.get()
        self._attempts += 1
        outcome = "failed"
        try:
            self._draw(attempt, train, request, fill)
            outcome = "returned"
        except BaselineError:
            outcome = "refused"
            raise
        finally:
            self._pending.add(attempt.tally)
            self.totals.add(attempt.tally)
            returned = outcome == "returned"
            self._seen += [(self._attempts, returned, seen) for seen in attempt.seen]
            logger.info(
                "direct arm=%s holdout=%d fill=%s seed=%d outcome=%s %s",
                self.name,
                -1 if fill is None else fill.holdout,
                "none" if fill is None else fill.kind,
                request.seed,
                outcome,
                attempt.line(),
            )
        params: dict[str, object] = {"generator": "direct", "seed": request.seed}
        if self._presence is not Presence.SHARES:
            params["presence"] = str(self._presence)
        baseline = Baseline(attempt.rows(), unit_weights(train), params)
        check(baseline, train, request)
        return baseline

    def drain(self, holdout: int, source: Source) -> dict[Measure, float]:
        """The counts since the last drain, as candidate measures; then start again.

        The fill's records go to the candidates file as lines of `holdout` and the fill `source`
        counts, each attempt numbered from 1 within the fill, each line's call null.
        """
        if self._sink is not None:
            fill = FILLS[source]
            self._sink.write(
                Candidate(holdout, self.name, fill, attempt, returned, seen)
                for attempt, returned, seen in self._seen
            )
        measures = self._pending.measures()
        self._pending = Tally()
        self._attempts = 0
        self._seen = []
        return measures

    def summary(self) -> None:
        """Log the run's sums over every fill: the validity rate's counts, and every drop."""
        logger.info("validity arm=%s %s", self.name, self.totals.line())

    def _draw(
        self, attempt: Attempt, train: pd.DataFrame, request: Request, fill: Fill | None
    ) -> None:
        """Draw every requested row once; a draw copies training values, so none is dropped.

        The sampler's lines name `fill`'s holdout and kind, or none outside a harness fill.
        """
        try:
            drawn = sample(
                train,
                request.quota,
                rows=self._rows,
                ontologies=self._ontologies,
                name=self._dataset,
                seed=request.seed,
                frame=request.frame,
                tag=None if fill is None else (fill.holdout, fill.kind),
                presence=self._presence,
            )
        except SamplingError as exc:
            raise BaselineError(str(exc)) from exc
        for cell, draws in drawn.items():
            attempt.add(None, cell, len(draws), [dict(draw.record) for draw in draws])
        if short := attempt.short():
            msg = f"{self.name} left {len(short)} cells short: {attempt.line()}"
            raise BaselineError(msg)


def run_direct(
    records: pd.DataFrame,
    experiment: Experiment,
    arms: Mapping[str, Runner],
    knowledge: Knowledge,
    *,
    names: Sequence[str],
    sink: CandidateFile,
    models: Mapping[Model, Factory],
    drains: Mapping[str, Drain] = MappingProxyType({}),
    on_holdout: Callable[[], None] | None = None,
    frame: FrameSpec,
) -> list[Score]:
    """Score every arm of `experiment` with a `DirectArm` per name of `names` writing `sink`.

    Each name is a key of `DIRECT_ARMS`, which says how its arm draws presence. Each arm logs its
    run's validity sums on the way out, a stopped run included.

    Raises:
        As `bko.evaluate.run.run`.
    """
    direct = {
        name: DirectArm(
            knowledge.field_rows,
            knowledge.ontologies,
            name=name,
            budget=experiment.budget,
            sink=sink,
            dataset=knowledge.name,
            presence=DIRECT_ARMS[name],
        )
        for name in names
    }
    try:
        return run(
            records,
            experiment,
            {**arms, **direct},
            models=models,
            drains={**drains, **{name: arm.drain for name, arm in direct.items()}},
            on_holdout=on_holdout,
            frame=frame,
        )
    finally:
        for arm in direct.values():
            arm.summary()
