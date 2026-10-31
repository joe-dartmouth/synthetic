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
file when the harness drains it (`bko.evaluate.holdout_candidates`). An arm given a prescribe step
(`bko.evaluate.skeleton`) draws each attempt's lines once, hands each planned call its cell's next
lines at wave planning, drops a record outside the call's cell or matching no line of the call
left unconsumed, and returns every line no kept record consumed to its cell's queue. ADR
`research/e1-evaluation-protocol` records the rule. `guided_arms` builds a run's BKO arms over one
call log and one candidates file, for the evaluate command and the app alike.
"""

from __future__ import annotations

import dataclasses
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
from bko.baselines.records import Baseline, Drop, check, unit_weights
from bko.baselines.request import Request, Strata
from bko.bkb.structure import BKB
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import permits_sending
from bko.evaluate.attempts import Attempt, Tally
from bko.evaluate.cllm import SHOW, Shower
from bko.evaluate.curation import classifier_discard
from bko.evaluate.errors import EvaluationError, StopRuleError
from bko.evaluate.experiment import MESSAGES_ARMS, Curation, Experiment
from bko.evaluate.holdout_candidates import FILLS, Candidate, CandidateFile, Origin, Seen
from bko.evaluate.scores import Measure, Source
from bko.evaluate.skeleton import Prescribe, Queues, prescriber
from bko.evaluate.watch import Watch
from bko.evaluate.waves import committer, give_up, lines_requeued, next_wave, spendable
from bko.gaps.definitions import Definition, gaps
from bko.gaps.observe import Cell, cell_key, record_cells
from bko.generate.errors import IdleError, ReplayError
from bko.generate.flight import Flight, Limits, Planned
from bko.generate.guidance import Guidance, cell_guidance, guide
from bko.generate.log import CallLog, utc_now
from bko.generate.messages import SETTINGS, AnySend, Settings
from bko.generate.replay import Answers, Replay
from bko.generate.skeleton import skeleton
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
        spent: The run's spend on the arm's attempts, in USD, summed from each attempt's change in
            the call log's spend.
    """

    def __init__(
        self,
        name: str,
        definition: Definition | None,
        knowledge: Knowledge,
        *,
        log: CallLog,
        budget: int,
        flight: Flight,
        candidates: CandidateFile | None = None,
        prescribe: Prescribe | None = None,
        shows: bool = False,
        curation: Curation | None = None,
    ) -> None:
        """Hold what each attempt needs; nothing is called until the arm runs.

        Args:
            name: The arm's name.
            definition: The gap definition that guides it; None guides each cell by itself alone
                (`bko.generate.guidance.cell_guidance`), reading no gap and no ontology.
            knowledge: The run's releases and field mappings.
            log: The run's call log, shared by every BKO arm.
            budget: The records an attempt may ask for, as a multiple of the request's rows.
            flight: The run's `Flight` over its send and `log`, which makes each call and holds
                it to the run's ceiling and idle limit, shared by every BKO arm, so an async
                client's calls stay on the one event loop its connections are bound to.
            candidates: The run's candidates file, shared by every BKO arm; None keeps no lines.
            prescribe: Draws each attempt's lines, one per requested record, which its calls
                carry; None for an arm that prescribes nothing.
            shows: Whether each call shows training records (`bko.evaluate.cllm`).
            curation: How each call's accepted rows are curated before they count
                (`bko.evaluate.curation`); None keeps every one.

        Raises:
            EvaluationError: If `curation` names a curator no arm has yet.
        """
        if curation is not None and curation is not Curation.CLASSIFIER:
            msg = f"{name} names curation {curation}, which no arm has yet"
            raise EvaluationError(msg)
        self.name = name
        self._prescribe = prescribe
        self._shows = shows
        self._curation = curation
        self.totals = self._tally()
        self.spent = 0.0
        self._definition = definition
        self._knowledge = knowledge
        self._log = log
        self._budget = budget
        self._candidates = candidates
        self._flight = flight
        self._pending = self._tally()
        self._attempts = 0
        self._seen: list[tuple[int, bool, Seen]] = []

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Fill `request` from the API, guided by `train`'s gaps under the arm's definition.

        A curating arm may return fewer rows than the quota: it gives up a cell its own budget
        cannot cover, logs the attempt as `partial` and records each such cell's missing rows
        under `short` in the params.

        Raises:
            BaselineError: Plain, for an arm that does not curate, when the rest of the budget
                cannot cover the rows still short.
            EvaluationError: If the request is not by cell; a `StopRuleError` if the run's spend
                reached the ceiling, or if a call got no reply within the idle limit.
            GenerationError: As `bko.generate.flight.Flight.wave`: a call with no response.
            WaveError: If a reply kept a record in a cell a later call of its wave asked for.
        """
        if request.strata is not Strata.CELL:
            msg = f"{self.name} fills cells only, not {request.strata}"
            raise EvaluationError(msg)
        guidance = self._guidance(train, request.frame)
        prescribing = self._prescribe is not None
        discard = None if self._curation is None else classifier_discard(train, frame=request.frame)
        attempt = Attempt(
            train, request, self._budget * request.rows, prescribing=prescribing, discard=discard
        )
        shower = Shower(train, frame=request.frame, seed=request.seed) if self._shows else None
        self._attempts += 1
        first = self._log.calls + 1
        before = self._log.usd
        outcome = "failed"
        queues: Queues | None = None
        try:
            queues = None if self._prescribe is None else Queues(self._prescribe(train, request))
            short = self._rounds(attempt, guidance, request.frame, queues, shower)
            outcome = "partial" if short else "returned"
        except BaselineError:
            outcome = "refused"
            raise
        finally:
            self._pending.add(attempt.tally)
            self.totals.add(attempt.tally)
            self.spent += self._log.usd - before
            returned = outcome in {"returned", "partial"}
            self._seen += [(self._attempts, returned, seen) for seen in attempt.seen]
            last = self._log.calls
            requeued = "" if self._prescribe is None else f" requeued={lines_requeued(queues)}"
            logger.info(
                "guided arm=%s seed=%d outcome=%s calls=%s %s%s usd=%.4f",
                self.name,
                request.seed,
                outcome,
                f"{first}-{last}" if last >= first else "none",
                attempt.line(),
                requeued,
                self._log.usd,
            )
        return self._baseline(attempt, train, first, short)

    def _baseline(
        self, attempt: Attempt, train: pd.DataFrame, first: int, short: Mapping[Cell, int]
    ) -> Baseline:
        """The attempt's rows as a `Baseline`, checked against its quota less `short`."""
        params: dict[str, object] = {
            "definition": None if self._definition is None else self._definition.value,
            "budget": self._budget,
            "calls": [first, self._log.calls],
        }
        if self._shows:
            params["shows"] = SHOW
        if self._curation is not None:
            params["curation"] = self._curation.value
        if short:
            params["short"] = {cell_key(cell): rows for cell, rows in short.items()}
        request = attempt.request
        baseline = Baseline(attempt.rows(), unit_weights(train), params)
        filled = {
            cell: rows - short.get(cell, 0)
            for cell, rows in request.quota.items()
            if rows > short.get(cell, 0)
        }
        check(baseline, train, dataclasses.replace(request, quota=filled) if short else request)
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
        self._pending = self._tally()
        self._attempts = 0
        self._seen = []
        return measures

    def summary(self) -> None:
        """Log the run's sums over every fill.

        They are the validity rate's numerator and denominator, every drop, and the arm's spend.
        """
        logger.info("validity arm=%s %s usd=%.4f", self.name, self.totals.line(), self.spent)

    def _tally(self) -> Tally:
        return Attempt.tally_for(
            prescribing=self._prescribe is not None, curating=self._curation is not None
        )

    def _guidance(self, train: pd.DataFrame, frame: FrameSpec) -> dict[Cell, Guidance]:
        known = self._knowledge
        if self._definition is None:
            cells = record_cells(train, (), name=known.name, edges=frame.edges)
            return cell_guidance(train, cells, frame=frame)
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
        self,
        attempt: Attempt,
        guidance: Mapping[Cell, Guidance],
        frame: FrameSpec,
        queues: Queues | None,
        shower: Shower | None = None,
    ) -> dict[Cell, int]:
        """Ask for each short cell round by round until the quota is filled.

        Returns:
            Each cell a curating arm leaves short, with its missing rows, once that cell's own
            budget cannot cover them (`bko.evaluate.waves.give_up`); empty when the quota is filled.

        Raises:
            BaselineError: Plain, for an arm that does not curate, when the rest of the attempt's
                budget cannot cover the rows still short.
        """
        per_call = self._log.settings.per_call
        waves = 0
        given_up: dict[Cell, int] = {}
        while short := {c: n for c, n in attempt.short().items() if c not in given_up}:
            if self._curation is not None:
                if self._budget * attempt.request.rows != attempt.limit:
                    msg = f"{self.name}'s cell budgets do not sum to its limit {attempt.limit}"
                    raise EvaluationError(msg)
                given_up |= give_up(attempt, short, name=self.name, budget=self._budget)
                short = {c: n for c, n in short.items() if c not in given_up}
                if not short:
                    break
            elif attempt.asked + sum(short.values()) > attempt.limit:
                msg = (
                    f"{self.name} cannot fill {len(short)} cells inside its budget of "
                    f"{attempt.limit} records: {attempt.line()}"
                )
                raise BaselineError(msg)
            cells = list(short)
            while cells:
                wave = next_wave(cells, guidance)
                cells = cells[len(wave) :]
                # Read from committed state: the cells of one wave cannot change each other's.
                rows = {cell: attempt.short().get(cell, 0) for cell in wave}
                # Lines are taken in plan order, so every in-flight count asks the same calls.
                planned = [
                    self._planned(
                        guidance[cell], min(per_call, count - start), frame, queues, shower
                    )
                    for cell, count in rows.items()
                    for start in range(0, count, per_call)
                ]
                if not planned:
                    continue
                commit = committer(attempt, planned, queues)
                waves += 1
                kept, concepts = sum(attempt.held().values()), attempt.tally.dropped[Drop.CONCEPTS]
                requeued = lines_requeued(queues)
                try:
                    self._flight.wave(planned, frame=frame, on_batch=commit)
                except IdleError as exc:
                    raise StopRuleError(str(exc), "idle", retryable=exc.retryable) from None
                if queues is not None:
                    logger.info(
                        "prescribed arm=%s seed=%d wave=%d asked=%d kept=%d dropped_concepts=%d "
                        "requeued=%d",
                        self.name,
                        attempt.request.seed,
                        waves,
                        sum(item.rows for item in planned),
                        sum(attempt.held().values()) - kept,
                        attempt.tally.dropped[Drop.CONCEPTS] - concepts,
                        queues.requeued - requeued,
                    )
        # A row kept in a given-up cell from another cell's call shrinks its shortfall.
        return {cell: rows for cell, rows in attempt.short().items() if cell in given_up}

    def _planned(
        self,
        guidance: Guidance,
        rows: int,
        frame: FrameSpec,
        queues: Queues | None,
        shower: Shower | None,
    ) -> Planned:
        if shower is not None:
            return Planned(guidance, rows, shown=shower.take(guidance.cell))
        if queues is None:
            return Planned(guidance, rows)
        lines = queues.take(guidance.cell, rows)
        return Planned(guidance, rows, skeleton(lines, self._knowledge.field_rows, frame=frame))


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
    `origin` and `settings`, and every arm writes its lines there. The call log names the run `run`
    and `origin`'s frame and records `limits.in_flight`, and every arm's calls go through one
    `Flight` at `limits`. Every call goes through `answers` when given, else through `send()`;
    `rehearsal` marks the call log as a rehearsal's, whose `send()` answers with canned replies, and
    a replay of a rehearsal's log keeps the mark. Once the body returns, answers left over are
    refused. Each arm logs its validity sums on the way out, a stopped run included.

    Raises:
        EvaluationError: If the data is held under an agreement that grants no permission to send
            its records, before either file exists.
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
    agreement = origin.data.agreement
    if agreement is not None and not permits_sending(agreement):
        msg = (
            f"{origin.data.name} is held under the agreement {agreement.name!r}, which grants no "
            "permission to send its records to the Messages API"
        )
        raise EvaluationError(msg)
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
            frame=origin.data.frame.name,
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
                spec.definition,
                knowledge,
                log=call_log,
                budget=experiment.budget if spec.budget is None else spec.budget,
                flight=flight,
                candidates=kept,
                prescribe=prescriber(knowledge) if spec.prescribes else None,
                shows=spec.shows,
                curation=spec.curation,
            )
            for name in names
            for spec in (MESSAGES_ARMS[name],)
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
