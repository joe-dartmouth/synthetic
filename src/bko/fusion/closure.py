"""What closes when validated records join: each gap definition's pairs before and after.

ADR `research/gap-closure-design` defines the reading: a (cell, statement) pair listed before the
join and not after is closed, one listed in both is open, and one listed only after is new.
Nothing here judges the share.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bko.bkb.structure import BKB
from bko.gaps.definitions import Definition, StatementGap, gaps
from bko.gaps.detect import Statement
from bko.gaps.observe import Cell, Cells


@dataclass(frozen=True, slots=True)
class Closure:
    """One definition's gaps before and after the join.

    Attributes:
        definition: The definition.
        before: Its gaps over the training records, in grid order.
        after: Its gaps over the training and joined records, in grid order.
    """

    definition: Definition
    before: tuple[StatementGap, ...]
    after: tuple[StatementGap, ...]

    @property
    def closed(self) -> tuple[StatementGap, ...]:
        """The pairs listed before and not after, in `before`'s order."""
        after = set(self.after)
        return tuple(gap for gap in self.before if gap not in after)

    @property
    def still_open(self) -> tuple[StatementGap, ...]:
        """The pairs listed before and after, in `before`'s order."""
        after = set(self.after)
        return tuple(gap for gap in self.before if gap in after)

    @property
    def opened(self) -> tuple[StatementGap, ...]:
        """The pairs listed after and not before, in `after`'s order."""
        before = set(self.before)
        return tuple(gap for gap in self.after if gap not in before)

    @property
    def share(self) -> float | None:
        """The closed pairs over the pairs listed before; `None` when none were."""
        return len(self.closed) / len(self.before) if self.before else None

    def listed(self, grid: Iterable[Cell]) -> tuple[StatementGap, ...]:
        """Every pair listed before or after, cells in `grid` order, statements sorted."""
        statements: dict[Cell, set[Statement]] = {}
        for gap in (*self.before, *self.after):
            statements.setdefault(gap.cell, set()).add(gap.statement)
        return tuple(
            StatementGap(cell, statement)
            for cell in grid
            for statement in sorted(statements.get(cell, ()))
        )


def closure(fragments: Sequence[BKB], before: Cells, after: Cells) -> tuple[Closure, ...]:
    """Each definition's gaps over `before` and over `after`, in `Definition` order.

    Raises:
        QueryError: As `bko.gaps.definitions.gaps`.
    """
    return tuple(
        Closure(definition, gaps(definition, fragments, before), gaps(definition, fragments, after))
        for definition in Definition
    )
