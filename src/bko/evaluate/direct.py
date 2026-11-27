"""The direct arms: a holdout's request drawn from the fused BKB at a draw spec.

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
them, one per name of `DIRECT_ARMS`, each drawing at its own spec. ADR
`research/bkb-direct-generation` records the sampler, ADR `research/logic-only-ablation` the
logic-only draw of `bko-logic`, whose components `run_direct` checks before any arm fits, and ADR
`research/no-mapping-ablation` the draw of `bko-nomap` without the mapping table. `bko-direct-pram`
and `bko-coherent` wrap a `DirectArm` and change its draw (`WRAPPERS`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, check, unit_weights
from bko.baselines.request import Request, Strata
from bko.datasets.frames import FrameSpec
from bko.evaluate.attempts import Attempt, Tally
from bko.evaluate.classifier import Factory, Model
from bko.evaluate.coherent import CoherentArm
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import COHERENT_ARM, DIRECT_ARMS, PRAM_ARM, Experiment
from bko.evaluate.guided import Knowledge
from bko.evaluate.holdout_candidates import FILLS, Candidate, CandidateFile, Seen
from bko.evaluate.perturb import Change, Perturbing, unmapped
from bko.evaluate.pram import THETA, PramArm
from bko.evaluate.run import Drain, run
from bko.evaluate.scores import Measure, Score, Source
from bko.generate.direct import logic_components, sample
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.generate.errors import SamplingError
from bko.generate.knowledge_draw import kb_check
from bko.generate.presence import Presence
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
        spec: DrawSpec = DEFAULT_SPEC,
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
            spec: What each draw reads (`bko.generate.direct.sample`).
        """
        self.name = name
        self.totals = Tally()
        self._rows = tuple(rows)
        self._ontologies = ontologies
        self._budget = budget
        self._sink = sink
        self._dataset = dataset
        self._spec = spec
        self._pending = Tally()
        self._attempts = 0
        self._seen: list[tuple[int, bool, Seen]] = []

    @property
    def rows(self) -> tuple[FieldMapping, ...]:
        """The field mappings it draws by."""
        return self._rows

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Fill `request` from `train`'s concept shares, seeded by the request's seed.

        Raises:
            BaselineError: Plain, when a cell is left short or no training record matches a draw.
            EvaluationError: If the request is not by cell.
        """
        return self.fill(train, request)

    def fill(
        self,
        train: pd.DataFrame,
        request: Request,
        *,
        change: Change | None = None,
        params: Mapping[str, object] = MappingProxyType({}),
    ) -> Baseline:
        """Fill `request` as `__call__` does, each cell's drawn records first passed to `change`.

        Args:
            train: The holdout's training rows.
            request: The request to fill.
            change: Takes the training rows, the request and every cell's drawn records, in quota
                order, and returns each cell's rows in the same order, each carrying the call that
                changed it; None keeps the draw (`bko.evaluate.perturb`).
            params: What the baseline's params add.

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
        fell = 0
        try:
            fell = self._draw(attempt, train, request, fill, change)
            outcome = "returned"
        except BaselineError:
            outcome = "refused"
            raise
        finally:
            self._pending.add(attempt.tally)
            self.totals.add(attempt.tally)
            returned = outcome == "returned"
            self._seen += [(self._attempts, returned, seen) for seen in attempt.seen]
            # `bko-direct`'s line stays as its runs of record logged it; an arm at another
            # spec, which falls back far more often, adds its count of fallen-back draws.
            logger.info(
                "direct arm=%s holdout=%d fill=%s seed=%d outcome=%s %s%s",
                self.name,
                -1 if fill is None else fill.holdout,
                "none" if fill is None else fill.kind,
                request.seed,
                outcome,
                attempt.line(),
                "" if self._spec == DEFAULT_SPEC else f" fallback_draws={fell}",
            )
        logged = {"generator": "direct", "seed": request.seed, **self._spec.params()}
        baseline = Baseline(attempt.rows(), unit_weights(train), {**logged, **params})
        check(baseline, train, request)
        return baseline

    def drain(self, holdout: int, source: Source) -> dict[Measure, float]:
        """The counts since the last drain, as candidate measures; then start again.

        The fill's records go to the candidates file as lines of `holdout` and the fill `source`
        counts, each attempt numbered from 1 within the fill, each line's call null unless a call
        changed its row.
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
        self,
        attempt: Attempt,
        train: pd.DataFrame,
        request: Request,
        fill: Fill | None,
        change: Change | None,
    ) -> int:
        """Draw every requested row once; a draw copies training values, so none is dropped.

        The sampler's lines name `fill`'s holdout and kind, or none outside a harness fill. With
        `change`, each cell's rows are its changed rows, and each one a call changed carries that
        call and its position there on its candidate line, and each one a step read its `perturb`.

        Returns:
            How many draws read a field from the label group (`Draw.fallback`).
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
                spec=self._spec,
            )
        except SamplingError as exc:
            raise BaselineError(str(exc)) from exc
        records = {cell: [dict(draw.record) for draw in draws] for cell, draws in drawn.items()}
        changed = None if change is None else change(train, request, records)
        for cell, rows in records.items():
            if changed is None:
                attempt.add(None, cell, len(rows), rows)
                continue
            first = len(attempt.seen)
            attempt.add(None, cell, len(rows), [row.record for row in changed[cell]])
            for offset, row in enumerate(changed[cell]):
                at = first + offset
                called = {} if row.call is None else {"call": row.call, "position": row.position}
                attempt.seen[at] = replace(attempt.seen[at], **called, perturb=row.result)
        if short := attempt.short():
            msg = f"{self.name} left {len(short)} cells short: {attempt.line()}"
            raise BaselineError(msg)
        return sum(bool(draw.fallback) for draws in drawn.values() for draw in draws)


type Wrapper = Callable[[DirectArm, Sequence[str]], Perturbing]
"""Builds a perturbing arm over the direct arm that draws for it and the fields it perturbs."""

WRAPPERS: Final[Mapping[str, Wrapper]] = MappingProxyType(
    {
        PRAM_ARM: lambda arm, fields: PramArm(arm, THETA, fields=fields),
        COHERENT_ARM: lambda arm, fields: CoherentArm(arm, fields=fields),
    }
)
"""The direct arms that perturb their draw, by name, each with what wraps its `DirectArm`:
`bko.evaluate.pram.PramArm` and `bko.evaluate.coherent.CoherentArm`."""


def run_direct(  # ruff: ignore[too-many-arguments]  (`run`'s keywords beside the direct ones)
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
    held: frozenset[int] | None = None,
) -> list[Score]:
    """Score every arm of `experiment` with a `DirectArm` per name of `names` writing `sink`.

    `held` is `bko.evaluate.run.run`'s.

    Each name is a key of `DIRECT_ARMS`, which says what its arm's draw reads; an arm of
    `WRAPPERS` then perturbs its draw's unmapped fields. Each arm logs its run's validity sums on
    the way out, a stopped run included.

    Raises:
        EvaluationError: If a logic-only arm's components cannot be enumerated
            (`bko.generate.direct.logic_components`), a knowledge-only arm's frame holds a value
            K cannot give (`bko.generate.knowledge_draw.kb_check`), or a perturbing arm's fields
            are not the frame's pinned ones (`bko.evaluate.perturb.unmapped`); raised before any
            arm fits.
        As `bko.evaluate.run.run`.
    """
    for name in names:
        spec = DIRECT_ARMS[name]
        try:
            if not spec.donors:
                kb_check(knowledge.field_rows, frame)
            if spec.presence is Presence.LOGIC:
                logic_components(knowledge.field_rows, knowledge.ontologies, frame)
        except ValueError as exc:
            msg = f"{name} cannot draw dataset {knowledge.name} on frame {frame.name}: {exc}"
            raise EvaluationError(msg) from exc
    direct: dict[str, DirectArm | Perturbing] = {}
    for name in names:
        arm = direct_arm(name, knowledge, experiment, sink)
        wrapper = WRAPPERS.get(name)
        if wrapper is None:
            direct[name] = arm
        else:
            direct[name] = wrapper(arm, unmapped(knowledge.field_rows, frame=frame))
    try:
        return run(
            records,
            experiment,
            {**arms, **direct},
            models=models,
            drains={**drains, **{name: arm.drain for name, arm in direct.items()}},
            on_holdout=on_holdout,
            frame=frame,
            held=held,
            rows=knowledge.field_rows,
            source_name=knowledge.name,
        )
    finally:
        for arm in direct.values():
            arm.summary()


def direct_arm(
    name: str, knowledge: Knowledge, experiment: Experiment, sink: CandidateFile | None
) -> DirectArm:
    """The `DirectArm` named `name` over `knowledge`, drawing at the spec `DIRECT_ARMS` gives it.

    Its draw is `DIRECT_ARM`'s whenever its spec is the default, whatever its name.
    """
    return DirectArm(
        knowledge.field_rows,
        knowledge.ontologies,
        name=name,
        budget=experiment.budget,
        sink=sink,
        dataset=knowledge.name,
        spec=DIRECT_ARMS.get(name, DEFAULT_SPEC),
    )
