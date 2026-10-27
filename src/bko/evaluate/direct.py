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
(`bko.evaluate.holdout_candidates`). `run_direct` runs a run's arms with the direct arm among
them. ADR `research/bkb-direct-generation` records the sampler.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, check, unit_weights
from bko.baselines.request import Request, Strata
from bko.evaluate.attempts import Attempt, Tally
from bko.evaluate.classifier import Factory, Model
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, Experiment
from bko.evaluate.guided import Knowledge
from bko.evaluate.holdout_candidates import FILLS, Candidate, CandidateFile, Seen
from bko.evaluate.run import Drain, run
from bko.evaluate.scores import Measure, Score, Source
from bko.generate.direct import sample
from bko.generate.errors import SamplingError
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
        """
        self.name = name
        self.totals = Tally()
        self._rows = tuple(rows)
        self._ontologies = ontologies
        self._budget = budget
        self._sink = sink
        self._dataset = dataset
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
        self._attempts += 1
        outcome = "failed"
        try:
            self._draw(attempt, train, request)
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
                "direct arm=%s seed=%d outcome=%s %s",
                self.name,
                request.seed,
                outcome,
                attempt.line(),
            )
        params = {"generator": "direct", "seed": request.seed}
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

    def _draw(self, attempt: Attempt, train: pd.DataFrame, request: Request) -> None:
        """Draw every requested row once; a draw copies training values, so none is dropped."""
        try:
            drawn = sample(
                train,
                request.quota,
                rows=self._rows,
                ontologies=self._ontologies,
                name=self._dataset,
                seed=request.seed,
                frame=request.frame,
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
    sink: CandidateFile,
    models: Mapping[Model, Factory],
    drains: Mapping[str, Drain] = MappingProxyType({}),
    on_holdout: Callable[[], None] | None = None,
) -> list[Score]:
    """Score every arm of `experiment` with a `DirectArm` named `DIRECT_ARM` writing `sink`.

    The arm logs its run's validity sums on the way out, a stopped run included.

    Raises:
        As `bko.evaluate.run.run`.
    """
    arm = DirectArm(
        knowledge.field_rows,
        knowledge.ontologies,
        name=DIRECT_ARM,
        budget=experiment.budget,
        sink=sink,
        dataset=knowledge.name,
    )
    try:
        return run(
            records,
            experiment,
            {**arms, DIRECT_ARM: arm},
            models=models,
            drains={**drains, DIRECT_ARM: arm.drain},
            on_holdout=on_holdout,
        )
    finally:
        arm.summary()
