"""The BKO arms: each reads a holdout's gaps under its definition and generates the request's rows.

A `GuidedArm` is a `bko.baselines.methods.Runner`. Given a holdout's training rows and the gap
request, it observes the rows' cells, finds their gaps under its definition and guides each
requested cell by them. It then asks the Messages API for each cell's shortfall, round by round in
quota order. Every reply's candidates are validated (`bko.validate`). Each accepted one that
`check` would refuse is dropped by `bko.baselines.records.drops`, and the rest count toward the
cell their own values name. An attempt asks for at most `budget` times the request's rows, and
refuses with a plain `BaselineError` as soon as the rest of its budget cannot cover the rows still
short. The arm keeps its validity counts itself, since a refused attempt returns no `Baseline`. ADR
`research/e1-evaluation-protocol` records the rule.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.records import Baseline, Drop, check, drops, number, unit_weights
from bko.baselines.request import Request, Strata, Stratum, stratum_keys
from bko.bkb.structure import BKB
from bko.evaluate.errors import EvaluationError
from bko.evaluate.scores import Measure
from bko.gaps.definitions import Definition, gaps
from bko.gaps.observe import Cell, record_cells
from bko.generate.calls import Batch, generate
from bko.generate.guidance import Guidance, guide
from bko.generate.log import CallLog
from bko.generate.messages import Send
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology
from bko.validate.checks import validate


logger = logging.getLogger(__name__)

DROPPED: Final[Mapping[Drop, Measure]] = MappingProxyType(
    {
        Drop.RANGE: Measure.DROPPED_RANGE,
        Drop.CELL: Measure.DROPPED_CELL,
        Drop.QUOTA: Measure.DROPPED_QUOTA,
    }
)
"""The measure each drop reason is counted under."""


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


@dataclass(slots=True)
class Tally:
    """Candidates counted over some calls.

    Attributes:
        generated: Candidates validated, undecoded placeholders included.
        passed: Those that passed every check.
        dropped: Those passed then dropped, by reason.
    """

    generated: int = 0
    passed: int = 0
    dropped: Counter[Drop] = field(default_factory=Counter)

    def add(self, other: Tally) -> None:
        """Add `other`'s counts to these."""
        self.generated += other.generated
        self.passed += other.passed
        self.dropped.update(other.dropped)

    def measures(self) -> dict[Measure, float]:
        """The counts as `Source.CANDIDATES` measures, every drop reason included."""
        return {
            Measure.GENERATED: float(self.generated),
            Measure.PASSED: float(self.passed),
            **{measure: float(self.dropped[drop]) for drop, measure in DROPPED.items()},
        }

    def line(self) -> str:
        """The counts as `key=value` pairs for a log line."""
        dropped = " ".join(f"dropped_{drop}={self.dropped[drop]}" for drop in Drop)
        return f"generated={self.generated} passed={self.passed} {dropped}"


class _Attempt:
    """One attempt's records asked, its counts, and its rows kept per cell."""

    def __init__(self, train: pd.DataFrame, request: Request, limit: int) -> None:
        self.train = train
        self.request = request
        self.limit = limit
        self.asked = 0
        self.tally = Tally()
        self.kept: dict[Stratum, list[pd.DataFrame]] = defaultdict(list)

    def held(self) -> dict[Stratum, int]:
        return {cell: sum(len(frame) for frame in frames) for cell, frames in self.kept.items()}

    def short(self) -> dict[Stratum, int]:
        """Each requested cell's rows still missing, in the quota's order."""
        held = self.held()
        return {
            cell: rows - held.get(cell, 0)
            for cell, rows in self.request.quota.items()
            if rows > held.get(cell, 0)
        }

    def take(self, batch: Batch) -> None:
        """Validate one call's candidates, drop what `check` would refuse, and keep the rest."""
        self.asked += batch.asked
        checked = validate(batch.candidates)
        self.tally.generated += checked.generated
        self.tally.passed += checked.passed
        rows = checked.accepted
        reasons = drops(rows, self.train, self.request, held=self.held())
        self.tally.dropped.update(reason for reason in reasons if reason is not None)
        # `.loc`: a plain `frame[[]]` would select no columns when nothing passed.
        kept = rows.loc[[reason is None for reason in reasons]]
        keys = stratum_keys(kept, Strata.CELL)
        for cell in dict.fromkeys(keys):
            self.kept[cell].append(kept.loc[[key == cell for key in keys]])

    def rows(self) -> pd.DataFrame:
        """The kept rows, cell by cell in the quota's order, in generation order within one."""
        frames = [frame for cell in self.request.quota for frame in self.kept[cell]]
        return number(pd.concat(frames))

    def line(self) -> str:
        return f"asked={self.asked} {self.tally.line()} kept={sum(self.held().values())}"


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
        send: Send,
        log: CallLog,
        budget: int,
        ceiling: float,
    ) -> None:
        """Hold what each attempt needs; nothing is called until the arm runs.

        Args:
            name: The arm's name.
            definition: The gap definition that guides it.
            knowledge: The run's releases and field mappings.
            send: Makes each call.
            log: The run's call log, shared by every BKO arm.
            budget: The records an attempt may ask for, as a multiple of the request's rows.
            ceiling: The run's spend in USD at which no further call is made.
        """
        self.name = name
        self.totals = Tally()
        self._definition = definition
        self._knowledge = knowledge
        self._send = send
        self._log = log
        self._budget = budget
        self._ceiling = ceiling
        self._pending = Tally()

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Fill `request` from the API, guided by `train`'s gaps under the arm's definition.

        Raises:
            BaselineError: Plain, when the rest of the budget cannot cover the rows still short.
            EvaluationError: If the request is not by cell, or the run's spend reached the ceiling.
            GenerationError: As `bko.generate.calls.generate`: a call with no response.
        """
        if request.strata is not Strata.CELL:
            msg = f"{self.name} fills cells only, not {request.strata}"
            raise EvaluationError(msg)
        guidance = self._guidance(train)
        attempt = _Attempt(train, request, self._budget * request.rows)
        first = self._log.calls + 1
        outcome = "failed"
        try:
            self._rounds(attempt, guidance)
            outcome = "returned"
        except BaselineError:
            outcome = "refused"
            raise
        finally:
            self._pending.add(attempt.tally)
            self.totals.add(attempt.tally)
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

    def drain(self) -> dict[Measure, float]:
        """The counts since the last drain, as `Source.CANDIDATES` measures; then start again."""
        measures = self._pending.measures()
        self._pending = Tally()
        return measures

    def summary(self) -> None:
        """Log the run's sums: the validity rate's numerator and denominator, and every drop."""
        logger.info("validity arm=%s %s", self.name, self.totals.line())

    def _guidance(self, train: pd.DataFrame) -> dict[Cell, Guidance]:
        known = self._knowledge
        cells = record_cells(train, known.field_rows, name=known.name)
        found = gaps(self._definition, known.fragments, cells)
        return guide(
            train,
            cells,
            found,
            definition=self._definition,
            fragments=known.fragments,
            ontologies=known.ontologies,
            rows=known.field_rows,
        )

    def _rounds(self, attempt: _Attempt, guidance: Mapping[Cell, Guidance]) -> None:
        while short := attempt.short():
            if attempt.asked + sum(short.values()) > attempt.limit:
                msg = (
                    f"{self.name} cannot fill {len(short)} cells inside its budget of "
                    f"{attempt.limit} records: {attempt.line()}"
                )
                raise BaselineError(msg)
            for cell in short:
                rows = attempt.short().get(cell, 0)
                per_call = self._log.settings.per_call
                # `generate`'s own chunks, one call each, so the ceiling is read before every call.
                for start in range(0, rows, per_call):
                    self._spendable()
                    generate(
                        {cell: min(per_call, rows - start)},
                        guidance,
                        send=self._send,
                        log=self._log,
                        on_batch=attempt.take,
                    )

    def _spendable(self) -> None:
        """Log the run's spend against the ceiling, and refuse the next call once it is reached."""
        spent = self._log.usd
        logger.info("spend usd=%.4f ceiling=%.2f", spent, self._ceiling)
        if spent >= self._ceiling:
            msg = f"the run's spend {spent:.4f} USD reached its ceiling of {self._ceiling} USD"
            raise EvaluationError(msg)
