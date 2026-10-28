"""The BKO arms: each reads a holdout's gaps under its definition and generates the request's rows.

A `GuidedArm` is a `bko.baselines.methods.Runner`. Given a holdout's training rows and a request,
the gap request or the replica request alike, it observes the rows' cells, finds their gaps under
its definition and guides each requested cell by them. It then asks the Messages API for each cell's
shortfall, round by round in quota order. Each round goes out in waves: a wave is the longest run of
the round's short cells whose (sex, label) chains are distinct, its calls are in flight together
(`bko.generate.flight`), and its batches are taken in call order before the next wave is read, so
the calls are those a one-call-at-a-time run makes (ADR `research/concurrent-call-waves`). Every
reply's candidates are validated (`bko.validate`).
Each accepted one that `check` would refuse is dropped by `bko.baselines.records.drops`, and the
rest count toward the cell their own values name. An attempt asks for at most `budget` times the
request's rows, and refuses with a plain `BaselineError` as soon as the rest of its budget cannot
cover the rows still short. The arm keeps its validity counts itself, since a refused attempt
returns no `Baseline`, and writes every candidate it saw, with its outcome, to the run's candidates
file when the harness drains it (`bko.evaluate.holdout_candidates`). ADR
`research/e1-evaluation-protocol` records the rule. `guided_arms` builds a run's BKO arms over one
call log and one candidates file, for the evaluate command and the app alike.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.records import Baseline, check, unit_weights
from bko.baselines.request import Request, Strata
from bko.bkb.structure import BKB
from bko.datasets.frames import FrameSpec
from bko.evaluate.attempts import Attempt, Tally
from bko.evaluate.errors import EvaluationError, StopRuleError
from bko.evaluate.experiment import BKO_ARMS, Experiment
from bko.evaluate.holdout_candidates import FILLS, Candidate, CandidateFile, Origin, Seen
from bko.evaluate.scores import Measure, Source
from bko.evaluate.watch import Watch
from bko.gaps.definitions import Definition, gaps
from bko.gaps.observe import Cell, cell_key, record_cells
from bko.generate.calls import Batch
from bko.generate.errors import IdleError, ReplayError, WaveError
from bko.generate.flight import Flight, Limits, Planned
from bko.generate.guidance import Guidance, guide
from bko.generate.log import CallLog, utc_now
from bko.generate.messages import SETTINGS, AnySend, Settings
from bko.generate.replay import Answers, Replay
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology


logger = logging.getLogger(__name__)

SEQUENTIAL: Final = Limits(in_flight=1, idle=None)
"""One call at a time with no idle limit."""


@dataclass(frozen=True, slots=True, eq=False)
class Knowledge:
    """What every holdout's gap detection reads, loaded once per run.

    Attributes:
        name: The dataset's source name; each cell's fragment takes it.
        ontologies: The loaded ontologies, by concept prefix.
        fragments: The ontology fragments the field mappings' concepts reach.
        field_rows: The field mappings.
    """

    name: str
    ontologies: Mapping[str, Ontology]
    fragments: tuple[BKB, ...]
    field_rows: tuple[FieldMapping, ...]


class GuidedArm:
    """One BKO arm: a `Runner` that keeps its validity counts over every attempt it makes.

    Attributes:
        name: The arm's name.
        totals: Every attempt's counts over the run.
    """

    def __init__(
        self,
        name: str,
        definition: Definition,
        knowledge: Knowledge,
        *,
        log: CallLog,
        budget: int,
        flight: Flight,
        candidates: CandidateFile | None = None,
    ) -> None:
        """Hold what each attempt needs; nothing is called until the arm runs.

        Args:
            name: The arm's name.
            definition: The gap definition that guides it.
            knowledge: The run's releases and field mappings.
            log: The run's call log, shared by every BKO arm.
            budget: The records an attempt may ask for, as a multiple of the request's rows.
            flight: The run's `Flight` over its send and `log`, which makes each call and holds
                it to the run's ceiling and idle limit, shared by every BKO arm, so an async
                client's calls stay on the one event loop its connections are bound to.
            candidates: The run's candidates file, shared by every BKO arm; None keeps no lines.
        """
        self.name = name
        self.totals = Tally()
        self._definition = definition
        self._knowledge = knowledge
        self._log = log
        self._budget = budget
        self._candidates = candidates
        self._flight = flight
        self._pending = Tally()
        self._attempts = 0
        self._seen: list[tuple[int, bool, Seen]] = []

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Fill `request` from the API, guided by `train`'s gaps under the arm's definition.

        Raises:
            BaselineError: Plain, when the rest of the budget cannot cover the rows still short.
            EvaluationError: If the request is not by cell; a `StopRuleError` if the run's spend
                reached the ceiling, or if a call got no reply within the idle limit.
            GenerationError: As `bko.generate.flight.Flight.wave`: a call with no response.
            WaveError: If a reply kept a record in a cell a later call of its wave asked for.
        """
        if request.strata is not Strata.CELL:
            msg = f"{self.name} fills cells only, not {request.strata}"
            raise EvaluationError(msg)
        guidance = self._guidance(train, request.frame)
        attempt = Attempt(train, request, self._budget * request.rows)
        self._attempts += 1
        first = self._log.calls + 1
        outcome = "failed"
        try:
            self._rounds(attempt, guidance, request.frame)
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
                "guided arm=%s seed=%d outcome=%s calls=%d-%d %s usd=%.4f",
                self.name,
                request.seed,
                outcome,
                first,
                self._log.calls,
                attempt.line(),
                self._log.usd,
            )
        params = {
            "definition": self._definition.value,
            "budget": self._budget,
            "calls": [first, self._log.calls],
        }
        baseline = Baseline(attempt.rows(), unit_weights(train), params)
        check(baseline, train, request)
        return baseline

    def drain(self, holdout: int, source: Source) -> dict[Measure, float]:
        """The counts since the last drain, as candidate measures; then start again.

        The harness drains after each fill, since the arm cannot tell a gap request from a replica
        request. The fill's candidates go to the candidates file as lines of `holdout` and the fill
        `source` counts, each attempt numbered from 1 within the fill.
        """
        if self._candidates is not None:
            fill = FILLS[source]
            self._candidates.write(
                Candidate(holdout, self.name, fill, attempt, returned, seen)
                for attempt, returned, seen in self._seen
            )
        measures = self._pending.measures()
        self._pending = Tally()
        self._attempts = 0
        self._seen = []
        return measures

    def summary(self) -> None:
        """Log the run's sums over every fill.

        They are the validity rate's numerator and denominator, and every drop.
        """
        logger.info("validity arm=%s %s", self.name, self.totals.line())

    def _guidance(self, train: pd.DataFrame, frame: FrameSpec) -> dict[Cell, Guidance]:
        known = self._knowledge
        cells = record_cells(train, known.field_rows, name=known.name, edges=frame.edges)
        found = gaps(self._definition, known.fragments, cells)
        return guide(
            train,
            cells,
            found,
            definition=self._definition,
            fragments=known.fragments,
            ontologies=known.ontologies,
            rows=known.field_rows,
            frame=frame,
        )

    def _rounds(
        self, attempt: Attempt, guidance: Mapping[Cell, Guidance], frame: FrameSpec
    ) -> None:
        per_call = self._log.settings.per_call
        while short := attempt.short():
            if attempt.asked + sum(short.values()) > attempt.limit:
                msg = (
                    f"{self.name} cannot fill {len(short)} cells inside its budget of "
                    f"{attempt.limit} records: {attempt.line()}"
                )
                raise BaselineError(msg)
            cells = list(short)
            while cells:
                wave = _wave(cells, guidance)
                cells = cells[len(wave) :]
                # Read from committed state: the cells of one wave cannot change each other's.
                rows = {cell: attempt.short().get(cell, 0) for cell in wave}
                planned = [
                    Planned(guidance[cell], min(per_call, count - start))
                    for cell, count in rows.items()
                    for start in range(0, count, per_call)
                ]
                if not planned:
                    continue
                commit = _committer(attempt, [item.guidance.cell for item in planned])
                try:
                    self._flight.wave(planned, frame=frame, on_batch=commit)
                except IdleError as exc:
                    raise StopRuleError(str(exc), "idle", retryable=exc.retryable) from None


def _wave(cells: Sequence[Cell], guidance: Mapping[Cell, Guidance]) -> list[Cell]:
    """The longest run of `cells` from the first whose (sex, label) chains are distinct."""
    chains: set[tuple[int, int]] = set()
    wave: list[Cell] = []
    for cell in cells:
        chain = (guidance[cell].sex, guidance[cell].label)
        if chain in chains:
            break
        chains.add(chain)
        wave.append(cell)
    return wave


def _committer(attempt: Attempt, plan: Sequence[Cell]) -> Callable[[Batch], None]:
    """Take a wave's batches, given in call order, refusing a row kept in a later group's cell.

    Args:
        attempt: The attempt the wave fills.
        plan: Each planned call's cell, in plan order; a group is one cell's calls.

    Returns:
        The callable that takes each batch. It raises `WaveError` if a batch keeps a row in a
        cell a later group of the wave asked for, so a one-call-at-a-time run would have asked
        that group differently.
    """
    groups = list(dict.fromkeys(plan))
    taken: list[Batch] = []

    def commit(batch: Batch) -> None:
        first = batch.index - len(taken)
        taken.append(batch)
        before = attempt.held()
        attempt.take(batch)
        after = attempt.held()
        for cell in groups[groups.index(batch.cell) + 1 :]:
            if after.get(cell, 0) > before.get(cell, 0):
                msg = (
                    f"call {batch.index} kept a record in cell {cell_key(cell)}, which call "
                    f"{first + plan.index(cell)} of the same wave already asked for: a reply "
                    "crossed sex or label, so the wave changed the call plan"
                )
                raise WaveError(msg)

    return commit


def spendable(log: CallLog, ceiling: float) -> None:
    """Log `log`'s spend against `ceiling`, and refuse the next call once it is reached.

    Raises:
        StopRuleError: If the spend reached the ceiling.
    """
    spent = log.usd
    logger.info("spend usd=%.4f ceiling=%.2f", spent, ceiling)
    if spent >= ceiling:
        logger.warning("stop rule=ceiling usd=%.4f ceiling=%.2f", spent, ceiling)
        msg = f"the run's spend {spent:.4f} USD reached its ceiling of {ceiling} USD"
        raise StopRuleError(msg, "ceiling")


@dataclass(frozen=True, slots=True)
class GuidedRun:
    """A run's BKO arms over one call log and one candidates file.

    Attributes:
        log: The call log every arm writes.
        arms: Each BKO arm, by name.
        watch: The run's stop rules, which the harness calls after each holdout.
        candidates: The candidates file every arm writes; a direct arm in the run shares it.
    """

    log: CallLog
    arms: Mapping[str, GuidedArm]
    watch: Watch
    candidates: CandidateFile


@contextmanager
def guided_arms(  # ruff: ignore[too-many-arguments]
    names: Sequence[str],
    knowledge: Knowledge,
    experiment: Experiment,
    *,
    settings: Settings = SETTINGS,
    log: Path,
    candidates: Path,
    origin: Origin,
    send: Callable[[], AnySend],
    run: str,
    limits: Limits,
    answers: Answers | None = None,
    rehearsal: bool = False,
    now: Callable[[], datetime] = utc_now,
) -> Generator[GuidedRun]:
    """Build a `GuidedArm` for each of `names` over one new call log, for the body to run.

    The candidates file at `candidates` is created before the call log, with a header naming
    `origin` and `settings`, and every arm writes its lines there. The call log names the run
    `run` and records `limits.in_flight`, and every arm's calls go through one `Flight` at
    `limits`. Every call goes through
    `answers` when given, else through `send()`; `rehearsal` marks the call log as a rehearsal's,
    whose `send()` answers with canned replies, and a replay of a rehearsal's log keeps the mark.
    Once the body returns, answers left over are refused. Each arm logs its validity sums on the
    way out, a stopped run included.

    Raises:
        ReplayError: If the body returns while `answers` holds calls it did not answer.
        GenerationError: As `CallLog`, if the settings model has no price, before `log` exists;
            the candidates file is then removed.
        importlib.metadata.PackageNotFoundError: As `CallLog`, before `log` exists; the
            candidates file is then removed.
        FileExistsError: If `candidates` exists, or `log` does; the candidates file made is then
            removed.
        Exception: Anything building `send()` or the arms raises, before the first call; both
            files are then removed.
    """
    answered = None if answers is None else answers.answered
    rehearsal = rehearsal or (answers is not None and answers.rehearsal)
    kept = CandidateFile(candidates, origin, settings)
    try:
        call_log = CallLog(
            log,
            settings,
            answered=answered,
            rehearsal=rehearsal,
            run=run,
            in_flight=limits.in_flight,
            now=now,
        )
    except BaseException:
        kept.discard()
        raise
    try:
        sender = answers if answers is not None else send()
        watch = Watch(experiment, lambda: call_log.usd, projecting=not isinstance(answers, Replay))
        ceiling = experiment.spend_ceiling
        flight = Flight(
            sender,
            call_log,
            limits,
            spendable=partial(spendable, call_log, ceiling),
            ceiling=ceiling,
        )
        arms = {
            name: GuidedArm(
                name,
                BKO_ARMS[name],
                knowledge,
                log=call_log,
                budget=experiment.budget,
                flight=flight,
                candidates=kept,
            )
            for name in names
        }
    except BaseException:
        call_log.discard()
        kept.discard()
        raise
    with kept, call_log, flight:
        try:
            yield GuidedRun(call_log, MappingProxyType(arms), watch, kept)
            if answers is not None and answers.remaining:
                msg = f"{answers.remaining} logged calls were not replayed"
                raise ReplayError(msg)
        finally:
            for arm in arms.values():
                arm.summary()
