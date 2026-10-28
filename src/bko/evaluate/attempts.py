"""One BKO attempt's bookkeeping: the records it asked for, its candidate counts and its kept rows.

`bko.evaluate.guided` makes the calls and `bko.evaluate.direct` draws records with none; an
`Attempt` takes each call's or draw's candidates as they come,
validates them (`bko.validate`), drops each accepted one that `check` would refuse
(`bko.baselines.records.drops`), and for a prescribing arm each one whose concepts differ from its
line (`bko.generate.skeleton`), keeps the rest under the cell their own values name, and it
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

from bko.baselines.records import Drop, drops, number
from bko.baselines.request import Request, Strata, Stratum, stratum_keys
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
        Drop.QUOTA: Measure.DROPPED_QUOTA,
    }
)
"""The measure each drop reason is counted under."""

COMMON: Final = (Drop.RANGE, Drop.CELL, Drop.QUOTA)
"""The drop reasons every arm counts; a prescribing arm also counts `Drop.CONCEPTS`."""


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
        """Add `other`'s counts to these."""
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
        self, train: pd.DataFrame, request: Request, limit: int, *, prescribing: bool = False
    ) -> None:
        """Start an attempt at `request` over `train` that may ask for `limit` records.

        A `prescribing` attempt's calls carry each record's concepts, so its counts report every
        drop reason; any other's report `COMMON`.
        """
        self.train = train
        self.request = request
        self.limit = limit
        self.asked = 0
        self.tally = self.tally_for(prescribing=prescribing)
        self.kept: dict[Stratum, list[pd.DataFrame]] = defaultdict(list)
        self.seen: list[Seen] = []

    @staticmethod
    def tally_for(*, prescribing: bool) -> Tally:
        """An empty tally reporting a `prescribing` attempt's drop reasons, or `COMMON`."""
        return Tally(reasons=tuple(Drop) if prescribing else COMMON)

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

    def take(self, batch: Batch, *, skeleton: Skeleton | None = None) -> dict[int, Stratum]:
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
    ) -> dict[int, Stratum]:
        """Validate candidates asked of `cell`, drop what `check` would refuse, keep the rest.

        Args:
            call: The call's index in the run, or None for records drawn with no call.
            cell: The cell they were asked for.
            asked: The records asked for.
            candidates: The decoded items, then one `Undecoded` per asked record short.
            skeleton: The call's prescriptions; a record whose concepts differ from its line, or
                past the lines, is dropped after its cell and before the quota. None prescribes
                nothing.

        Returns:
            Each kept candidate's position among `candidates`, with the cell it counts toward.
        """
        self.asked += asked
        frame = self.request.frame
        checked = validate(candidates, frame=frame)
        self.tally.generated += checked.generated
        self.tally.passed += checked.passed
        rows = checked.accepted
        differs = (
            None
            if skeleton is None
            else [
                skeleton.differs(position, {str(name): float(value) for name, value in row.items()})
                for position, row in zip(rows.index.tolist(), rows.to_dict("records"), strict=True)
            ]
        )
        reasons = drops(rows, self.train, self.request, held=self.held(), differs=differs)
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
        return {int(position): key for position, key in zip(kept.index, keys, strict=True)}

    def rows(self) -> pd.DataFrame:
        """The kept rows, cell by cell in the quota's order, in generation order within one."""
        frames = [frame for cell in self.request.quota for frame in self.kept[cell]]
        return number(pd.concat(frames))

    def line(self) -> str:
        """The attempt's counts as `key=value` pairs for a log line."""
        return f"asked={self.asked} {self.tally.line()} kept={sum(self.held().values())}"
