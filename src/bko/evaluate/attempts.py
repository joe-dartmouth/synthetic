"""One BKO attempt's bookkeeping: the records it asked for, its candidate counts and its kept rows.

`bko.evaluate.guided` makes the calls; an `Attempt` takes each call's candidates as they return,
validates them (`bko.validate`), drops each accepted one that `check` would refuse
(`bko.baselines.records.drops`) and keeps the rest under the cell their own values name, and it
records each candidate's outcome for the per-holdout candidates file
(`bko.evaluate.holdout_candidates`). `Tally` counts candidates over some calls, and `DROPPED`
names the measure each drop reason is counted under.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.records import Drop, drops, number
from bko.baselines.request import Request, Strata, Stratum, stratum_keys
from bko.evaluate.holdout_candidates import KEPT, REJECTED, Seen
from bko.evaluate.scores import Measure
from bko.gaps.observe import cell_key
from bko.generate.calls import Batch
from bko.generate.decode import Undecoded
from bko.validate.checks import validate


DROPPED: Final[Mapping[Drop, Measure]] = MappingProxyType(
    {
        Drop.RANGE: Measure.DROPPED_RANGE,
        Drop.CELL: Measure.DROPPED_CELL,
        Drop.QUOTA: Measure.DROPPED_QUOTA,
    }
)
"""The measure each drop reason is counted under."""


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
        """The counts as candidate measures, every drop reason included."""
        return {
            Measure.GENERATED: float(self.generated),
            Measure.PASSED: float(self.passed),
            **{measure: float(self.dropped[drop]) for drop, measure in DROPPED.items()},
        }

    def line(self) -> str:
        """The counts as `key=value` pairs for a log line."""
        dropped = " ".join(f"dropped_{drop}={self.dropped[drop]}" for drop in Drop)
        return f"generated={self.generated} passed={self.passed} {dropped}"


class Attempt:
    """One attempt's records asked, its counts, its rows kept per cell and every candidate seen.

    Attributes:
        seen: Every candidate the attempt took, in call order, with its outcome.
    """

    def __init__(self, train: pd.DataFrame, request: Request, limit: int) -> None:
        """Start an attempt at `request` over `train` that may ask for `limit` records."""
        self.train = train
        self.request = request
        self.limit = limit
        self.asked = 0
        self.tally = Tally()
        self.kept: dict[Stratum, list[pd.DataFrame]] = defaultdict(list)
        self.seen: list[Seen] = []

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

    def take(self, batch: Batch) -> None:
        """Validate one call's candidates, drop what `check` would refuse, and keep the rest."""
        self.asked += batch.asked
        frame = self.request.frame
        checked = validate(batch.candidates, frame=frame)
        self.tally.generated += checked.generated
        self.tally.passed += checked.passed
        rows = checked.accepted
        reasons = drops(rows, self.train, self.request, held=self.held())
        self.tally.dropped.update(reason for reason in reasons if reason is not None)
        outcomes = {
            int(position): KEPT if reason is None else DROPPED[reason].value
            for position, reason in zip(rows.index, reasons, strict=True)
        }
        failed = {rejection.position: rejection.failed for rejection in checked.rejected}
        cell = cell_key(batch.cell)
        for position, candidate in enumerate(batch.candidates):
            undecoded = candidate.reason.value if isinstance(candidate, Undecoded) else None
            self.seen.append(
                Seen(
                    batch.index,
                    cell,
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

    def rows(self) -> pd.DataFrame:
        """The kept rows, cell by cell in the quota's order, in generation order within one."""
        frames = [frame for cell in self.request.quota for frame in self.kept[cell]]
        return number(pd.concat(frames))

    def line(self) -> str:
        """The attempt's counts as `key=value` pairs for a log line."""
        return f"asked={self.asked} {self.tally.line()} kept={sum(self.held().values())}"
