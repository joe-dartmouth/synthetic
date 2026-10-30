"""One BKO attempt's bookkeeping: the records it asked for, its candidate counts and its kept rows.

`bko.evaluate.guided` makes the calls and `bko.evaluate.direct` draws records with none; an
`Attempt` takes each call's or draw's candidates as they come,
validates them (`bko.validate`), drops each accepted one that `check` would refuse
(`bko.baselines.records.assess`), and for a prescribing call each one outside the call's cell or
whose concepts match no line of the call left unconsumed (`bko.generate.skeleton`), for a curating
attempt each one its curator drops (`bko.evaluate.curation`), keeps the rest
under the cell their own values name, and it
records each candidate's outcome for the per-holdout candidates file
(`bko.evaluate.holdout_candidates`). `Tally` counts candidates over some calls, and `DROPPED`
names the measure each drop reason is counted under.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.records import Drop, Prescribed, assess, no_rows, number
from bko.baselines.request import Request, Strata, Stratum, stratum_keys
from bko.evaluate.curation import Discard
from bko.evaluate.holdout_candidates import KEPT, REJECTED, Seen
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
    }
)
"""The measure each drop reason is counted under."""

COMMON: Final = (Drop.RANGE, Drop.CELL, Drop.QUOTA)
"""The drop reasons every arm counts; a prescribing arm also counts `Drop.CONCEPTS`, and a curating
arm `Drop.CURATION`."""


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
    """

    def __init__(
        self,
        train: pd.DataFrame,
        request: Request,
        limit: int,
        *,
        prescribing: bool = False,
        discard: Discard | None = None,
    ) -> None:
        """Start an attempt at `request` over `train` that may ask for `limit` records.

        A `prescribing` attempt's calls carry each record's concepts, so its counts report
        `Drop.CONCEPTS` beside `COMMON`; an attempt given a `discard` curates each call's accepted
        rows with it, so its counts report `Drop.CURATION`.
        """
        self.train = train
        self.request = request
        self.limit = limit
        self.asked = 0
        self.asked_in: Counter[Stratum] = Counter()
        self.discard = discard
        self.tally = self.tally_for(prescribing=prescribing, curating=discard is not None)
        self.kept: dict[Stratum, list[pd.DataFrame]] = defaultdict(list)
        self.seen: list[Seen] = []

    @staticmethod
    def tally_for(*, prescribing: bool, curating: bool = False) -> Tally:
        """An empty tally of `COMMON` and the reasons a prescribing or curating attempt adds.

        `Drop.CONCEPTS` joins if `prescribing` and `Drop.CURATION` if `curating`, in `Drop`'s order.
        """
        extra = {Drop.CONCEPTS: prescribing, Drop.CURATION: curating}
        return Tally(reasons=tuple(d for d in Drop if d in COMMON or extra.get(d)))

    def held(self) -> dict[Stratum, int]:
        """Each cell's rows kept so far."""
        return {cell: sum(len(frame) for frame in frames) for cell, frames in self.kept.items()}

    def short(self) -> dict[Stratum, int]:
        """Each requested cell's rows still missing, in the quota's order."""
        held = self.held()
        return {
            cell: rows - held.get(cell, 0)
            for cell, rows in self.request.quota.items()
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
                nothing. An attempt given a `discard` drops each row its curator marks after the
                row's cell and concepts and before the quota.

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
        curated = None if self.discard is None else tuple(self.discard(rows).tolist())
        assessed = assess(
            rows,
            self.train,
            self.request,
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
                )
            )
        # `.loc`: a plain `frame[[]]` would select no columns when nothing passed.
        kept = rows.loc[[reason is None for reason in reasons]]
        keys = stratum_keys(kept, Strata.CELL, edges=frame.edges)
        for stratum in dict.fromkeys(keys):
            self.kept[stratum].append(kept.loc[[key == stratum for key in keys]])
        return frozenset(line for line in assessed.lines if line is not None)

    def rows(self) -> pd.DataFrame:
        """The kept rows, cell by cell in the quota's order, in generation order within one."""
        frames = [frame for cell in self.request.quota for frame in self.kept[cell]]
        return number(pd.concat(frames)) if frames else no_rows(self.request.frame)

    def line(self) -> str:
        """The attempt's counts as `key=value` pairs for a log line."""
        return f"asked={self.asked} {self.tally.line()} kept={sum(self.held().values())}"
