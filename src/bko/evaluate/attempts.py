"""One BKO attempt's bookkeeping: the records it asked for, its candidate counts and its kept rows.

`bko.evaluate.guided` makes the calls and `bko.evaluate.direct` draws records with none; an
`Attempt` takes each call's or draw's candidates as they come,
validates them (`bko.validate`), drops each accepted one that `check` would refuse
(`bko.baselines.records.assess`), and for a prescribing call each one outside the call's cell or
whose concepts match no line of the call left unconsumed (`bko.generate.skeleton`), for a curating
attempt each one its curator drops (`bko.evaluate.curation`), keeps the rest
under the cell their own values name, and it
records each candidate's outcome for the per-holdout candidates file
(`bko.evaluate.holdout_candidates`). A ranking attempt asks each cell for `over_ask` times its
quota, drops each row outside its call's cell, and once its rounds end keeps each call cell's
`quota` rows its `rank` scores highest, ties in `bko.evaluate.ranking.tie` order, dropping the rest
for their likelihood (ADR `research/bkb-candidate-curation`). `Tally` counts candidates over some
calls, and `DROPPED` names the measure each drop reason is counted under.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.records import Drop, Prescribed, assess, no_rows, number
from bko.baselines.request import Request, Strata, Stratum, stratum_keys
from bko.evaluate.curation import Scores, discards
from bko.evaluate.holdout_candidates import (
    ALEATORIC,
    CONFIDENCE,
    KEPT,
    LIKELIHOOD,
    REJECTED,
    Seen,
)
from bko.evaluate.ranking import over_ask_error, tie
from bko.evaluate.scores import Measure
from bko.gaps.observe import Cell, cell_key
from bko.generate.calls import Batch
from bko.generate.decode import Undecoded
from bko.generate.skeleton import Skeleton
from bko.validate.checks import validate


DROPPED: Final[Mapping[Drop, Measure]] = MappingProxyType(
    {
        Drop.RANGE: Measure.DROPPED_RANGE,
        Drop.CELL: Measure.DROPPED_CELL,
        Drop.CONCEPTS: Measure.DROPPED_CONCEPTS,
        Drop.CURATION: Measure.DROPPED_CURATION,
        Drop.QUOTA: Measure.DROPPED_QUOTA,
        Drop.LIKELIHOOD: Measure.DROPPED_LIKELIHOOD,
    }
)
"""The measure each drop reason is counted under."""

COMMON: Final = (Drop.RANGE, Drop.CELL, Drop.QUOTA)
"""The drop reasons every arm counts; a prescribing arm also counts `Drop.CONCEPTS`, a curating
arm `Drop.CURATION`, and a ranking arm `Drop.LIKELIHOOD`."""

type Rank = Callable[[Sequence[Seen]], Sequence[float]]
"""A ranking: given one call cell's valid candidates, each one's score, higher kept first."""


@dataclass(frozen=True, slots=True)
class Ranked:
    """One call cell's ranking, as its `ranked` line logs it.

    Attributes:
        cell: The call cell.
        rows: Its valid rows ranked.
        kept: Those it kept, at most its quota.
        tied: Its rows whose score equals the last kept row's, when that score is finite and a
            dropped row shares it, else 0; rows tied at minus infinity count under
            `minus_infinity` alone.
        minus_infinity: Its rows the sampler cannot draw.

    A cell with no valid row is not ranked, so it has no `Ranked`.
    """

    cell: Stratum
    rows: int
    kept: int
    tied: int
    minus_infinity: int


@dataclass(slots=True)
class Tally:
    """Candidates counted over some calls.

    Attributes:
        generated: Candidates validated, undecoded placeholders included.
        passed: Those that passed every check.
        dropped: Those passed then dropped, by reason.
        reasons: The drop reasons the counts report, `COMMON` unless the arm prescribes.
    """

    generated: int = 0
    passed: int = 0
    dropped: Counter[Drop] = field(default_factory=Counter)
    reasons: tuple[Drop, ...] = COMMON

    def add(self, other: Tally) -> None:
        """Add `other`'s counts to these.

        Raises:
            ValueError: If `other` reports other drop reasons, so its counts would be misread.
        """
        if other.reasons != self.reasons:
            msg = f"a tally of {other.reasons} cannot add to one of {self.reasons}"
            raise ValueError(msg)
        self.generated += other.generated
        self.passed += other.passed
        self.dropped.update(other.dropped)

    def measures(self) -> dict[Measure, float]:
        """The counts as candidate measures, every one of `reasons` included."""
        return {
            Measure.GENERATED: float(self.generated),
            Measure.PASSED: float(self.passed),
            **{DROPPED[drop]: float(self.dropped[drop]) for drop in self.reasons},
        }

    def line(self) -> str:
        """The counts as `key=value` pairs for a log line."""
        dropped = " ".join(f"dropped_{drop}={self.dropped[drop]}" for drop in self.reasons)
        return f"generated={self.generated} passed={self.passed} {dropped}"


class Attempt:
    """One attempt's records asked, its counts, its rows kept per cell and every candidate seen.

    Attributes:
        seen: Every candidate the attempt took, in call order, with its outcome.
        ranking: Each ranked cell's counts, in the quota's order, once `rows` has ranked.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        request: Request,
        limit: int,
        *,
        prescribing: bool = False,
        curator: Scores | None = None,
        rank: Rank | None = None,
        over_ask: int = 1,
        holdout: int = -1,
    ) -> None:
        """Start an attempt at `request` over `train` that may ask for `limit` records.

        A `prescribing` attempt's calls carry each record's concepts, so its counts report
        `Drop.CONCEPTS` beside `COMMON`; an attempt given a `curator` drops each call's accepted
        rows it reads as `bko.evaluate.curation.discards` drops, keeps each one's scores on its
        candidate, and counts `Drop.CURATION`; an attempt given a `rank` holds `over_ask` times
        each cell's quota before `rows` ranks it, ties broken in `bko.evaluate.ranking.tie` order
        on `holdout`, and counts `Drop.LIKELIHOOD`. A `rank` needs `over_ask` above 1, so the
        attempt holds rows to drop; one without a `rank` needs `over_ask` 1.

        Raises:
            ValueError: If `over_ask` breaks `bko.evaluate.ranking.over_ask_error`'s rule for an
                attempt with or without a `rank`.
        """
        msg = over_ask_error(over_ask, ranks=rank is not None)
        if msg is not None:
            raise ValueError(msg)
        self.train = train
        self.request = request
        self.limit = limit
        self.asked = 0
        self.asked_in: Counter[Stratum] = Counter()
        self.curator = curator
        self.rank = rank
        self.holdout = holdout
        self.target = {cell: over_ask * rows for cell, rows in request.quota.items()}
        self.tally = self.tally_for(
            prescribing=prescribing, curating=curator is not None, ranking=rank is not None
        )
        self.kept: dict[Stratum, list[pd.DataFrame]] = defaultdict(list)
        self.seen: list[Seen] = []
        self._valid: dict[Stratum, list[tuple[int, pd.DataFrame]]] = defaultdict(list)
        self._ranked = False
        self.ranking: list[Ranked] = []

    @staticmethod
    def tally_for(*, prescribing: bool, curating: bool = False, ranking: bool = False) -> Tally:
        """An empty tally of `COMMON` and what a prescribing, curating or ranking attempt adds.

        `Drop.CONCEPTS` joins if `prescribing`, `Drop.CURATION` if `curating` and
        `Drop.LIKELIHOOD` if `ranking`, in `Drop`'s order.
        """
        extra = {Drop.CONCEPTS: prescribing, Drop.CURATION: curating, Drop.LIKELIHOOD: ranking}
        return Tally(reasons=tuple(d for d in Drop if d in COMMON or extra.get(d)))

    def held(self) -> dict[Stratum, int]:
        """Each cell's rows kept so far."""
        return {cell: sum(len(frame) for frame in frames) for cell, frames in self.kept.items()}

    def short(self) -> dict[Stratum, int]:
        """Each requested cell's rows still missing, in the quota's order.

        A ranking attempt's cell misses `over_ask` times its quota less its rows held.
        """
        held = self.held()
        return {
            cell: rows - held.get(cell, 0)
            for cell, rows in self.target.items()
            if rows > held.get(cell, 0)
        }

    def take(self, batch: Batch, *, skeleton: Skeleton | None = None) -> frozenset[int]:
        """Validate one call's candidates, drop what `add` drops, and keep the rest.

        Returns:
            As `add`.
        """
        return self.add(batch.index, batch.cell, batch.asked, batch.candidates, skeleton=skeleton)

    def add(
        self,
        call: int | None,
        cell: Cell,
        asked: int,
        candidates: Sequence[object],
        *,
        skeleton: Skeleton | None = None,
    ) -> frozenset[int]:
        """Validate candidates asked of `cell`, drop what `check` would refuse, keep the rest.

        Args:
            call: The call's index in the run, or None for records drawn with no call.
            cell: The cell they were asked for.
            asked: The records asked for.
            candidates: The decoded items, then one `Undecoded` per asked record short.
            skeleton: The call's prescriptions; a record outside `cell` is dropped for its
                cell, and one whose concepts match no line left unconsumed is dropped after its
                cell and before the quota (`bko.baselines.records.assess`). None prescribes
                nothing. An attempt given a `curator` drops each row its curator marks after the
                row's cell and concepts and before the quota. A ranking attempt drops a row outside
                `cell` for its cell, and counts the rest toward `over_ask` times `cell`'s quota.

        Returns:
            The positions among `skeleton`'s lines of those a kept record consumed; empty
            without a skeleton.
        """
        self.asked += asked
        self.asked_in[cell] += asked
        frame = self.request.frame
        checked = validate(candidates, frame=frame)
        self.tally.generated += checked.generated
        self.tally.passed += checked.passed
        rows = checked.accepted
        prescribed = (
            None
            if skeleton is None
            else Prescribed(
                cell,
                skeleton.lines,
                tuple(
                    skeleton.shows({str(name): float(value) for name, value in row.items()})
                    for row in rows.to_dict("records")
                ),
            )
        )
        curated, scores = self._curated(rows)
        assessed = assess(
            rows,
            self.train,
            self._asked_of(cell),
            held=self.held(),
            prescribed=prescribed,
            curated=curated,
        )
        reasons = assessed.reasons
        self.tally.dropped.update(reason for reason in reasons if reason is not None)
        outcomes = {
            int(position): KEPT if reason is None else DROPPED[reason].value
            for position, reason in zip(rows.index, reasons, strict=True)
        }
        failed = {rejection.position: rejection.failed for rejection in checked.rejected}
        key = cell_key(cell)
        first = len(self.seen)
        for position, candidate in enumerate(candidates):
            undecoded = candidate.reason.value if isinstance(candidate, Undecoded) else None
            self.seen.append(
                Seen(
                    call,
                    key,
                    position,
                    None if undecoded is not None else candidate,
                    undecoded,
                    outcomes.get(position, REJECTED),
                    failed.get(position, ()),
                    scores.get(position),
                )
            )
        # `.loc`: a plain `frame[[]]` would select no columns when nothing passed.
        kept = rows.loc[[reason is None for reason in reasons]]
        if self.rank is not None:
            self._valid[cell] += [(first + int(at), kept.loc[[at]]) for at in kept.index]
        keys = stratum_keys(kept, Strata.CELL, edges=frame.edges)
        for stratum in dict.fromkeys(keys):
            self.kept[stratum].append(kept.loc[[key == stratum for key in keys]])
        return frozenset(line for line in assessed.lines if line is not None)

    def _curated(
        self, rows: pd.DataFrame
    ) -> tuple[tuple[bool, ...] | None, dict[int, dict[str, float | None]]]:
        """Which accepted rows the curator drops, and each one's scores by position.

        None and no scores without a curator.
        """
        if self.curator is None:
            return None, {}
        read = self.curator(rows)
        confidence = read[CONFIDENCE].to_numpy()
        aleatoric = read[ALEATORIC].to_numpy()
        scores: dict[int, dict[str, float | None]] = {
            int(position): {ALEATORIC: float(a), CONFIDENCE: float(c)}
            for position, c, a in zip(rows.index, confidence, aleatoric, strict=True)
        }
        return tuple(discards(confidence, aleatoric).tolist()), scores

    def _asked_of(self, cell: Stratum) -> Request:
        """The request a call of `cell` is assessed against.

        A ranking attempt's asks `cell` alone, for `over_ask` times its quota.
        """
        if self.rank is None:
            return self.request
        return dataclasses.replace(self.request, quota={cell: self.target[cell]})

    def rows(self) -> pd.DataFrame:
        """The kept rows, cell by cell in the quota's order, in generation order within one.

        A ranking attempt first ranks, once: each call cell's valid rows are scored by `rank`,
        and the cell keeps its quota's highest, ties in `bko.evaluate.ranking.tie` order and
        minus infinity last; the rest take the outcome `dropped-likelihood`, and every ranked
        candidate keeps its score, None for minus infinity. A cell holding no more than its quota
        keeps every row. Each ranked cell's counts join `ranking`, in the quota's order; a cell
        with no valid row is not ranked. An added frame with no rows when none was kept.

        Raises:
            ValueError: If `rank` returns other than one score per candidate, or a score that is
                NaN or plus infinity; the attempt is then left unranked.
            EvaluationError: As `rank`, such as `bko.evaluate.ranking.record`'s refusal of a
                candidate that is not a record of every column.
        """
        if self.rank is not None and not self._ranked:
            # Every cell is scored before any candidate is marked, so a refusal changes nothing.
            scored = {
                cell: self._scores(self.rank, cell, self._valid[cell])
                for cell in self.request.quota
                if cell in self._valid
            }
            for cell, scores in scored.items():
                self.kept[cell] = self._ranked_rows(cell, self._valid[cell], scores)
            self._ranked = True
        frames = [frame for cell in self.request.quota for frame in self.kept[cell]]
        return number(pd.concat(frames)) if frames else no_rows(self.request.frame)

    def _scores(
        self, rank: Rank, cell: Stratum, valid: Sequence[tuple[int, pd.DataFrame]]
    ) -> list[float]:
        """`rank`'s score of each of `cell`'s `valid` rows.

        Raises:
            ValueError: If `rank` returns other than one score per row, or a NaN or plus infinity.
        """
        scores = [float(score) for score in rank([self.seen[at] for at, _ in valid])]
        if len(scores) != len(valid):
            msg = f"the rank scored {len(scores)} of cell {cell_key(cell)}'s {len(valid)} rows"
            raise ValueError(msg)
        if any(math.isnan(score) or score == math.inf for score in scores):
            msg = f"the rank scored a row of cell {cell_key(cell)} NaN or plus infinity: {scores}"
            raise ValueError(msg)
        return scores

    def _ranked_rows(
        self, cell: Stratum, valid: Sequence[tuple[int, pd.DataFrame]], scores: Sequence[float]
    ) -> list[pd.DataFrame]:
        """`cell`'s kept rows by `scores`, in generation order; marks every candidate."""
        seen = [self.seen[at] for at, _ in valid]
        order = sorted(
            range(len(seen)),
            key=lambda i: (-scores[i], tie(self.holdout, seen[i].call, seen[i].position)),
        )
        quota = self.request.quota[cell]
        best = set(order[:quota])
        cut = scores[order[quota - 1]] if 0 < quota < len(order) else None
        split = cut is not None and cut != -math.inf and scores[order[quota]] == cut
        self.ranking.append(
            Ranked(
                cell,
                rows=len(valid),
                kept=len(best),
                tied=sum(score == cut for score in scores) if split else 0,
                minus_infinity=sum(score == -math.inf for score in scores),
            )
        )
        dropped = DROPPED[Drop.LIKELIHOOD].value
        for i, (at, _) in enumerate(valid):
            score = None if scores[i] == -math.inf else scores[i]
            self.seen[at] = dataclasses.replace(
                self.seen[at],
                outcome=KEPT if i in best else dropped,
                score={LIKELIHOOD: score},
            )
        self.tally.dropped[Drop.LIKELIHOOD] += len(valid) - len(best)
        return [frame for i, (_, frame) in enumerate(valid) if i in best]

    def line(self) -> str:
        """The attempt's counts as `key=value` pairs for a log line."""
        return f"asked={self.asked} {self.tally.line()} kept={sum(self.held().values())}"
