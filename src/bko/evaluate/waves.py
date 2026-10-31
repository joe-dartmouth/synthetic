"""The wave helpers the guided arms share: a round's next wave, its batches' commit, the spend gate.

A wave is the longest run of a round's short cells whose (sex, label) chains are distinct; its
batches are taken in call order, and a batch that keeps a row in a later group's cell refuses the
wave (ADR `research/concurrent-call-waves`). `spendable` is the exact spend check the flight runs
before each call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

from bko.evaluate.attempts import Attempt
from bko.evaluate.errors import EvaluationError, StopRuleError
from bko.evaluate.skeleton import Queues
from bko.gaps.observe import Cell, cell_key
from bko.generate.calls import Batch
from bko.generate.errors import WaveError
from bko.generate.flight import Planned
from bko.generate.guidance import Guidance
from bko.generate.log import CallLog
from bko.generate.skeleton import Skeleton


logger = logging.getLogger(__name__)


def lines_requeued(queues: Queues | None) -> int:
    """The lines `queues` has given back, 0 for none."""
    return 0 if queues is None else queues.requeued


def next_wave(cells: Sequence[Cell], guidance: Mapping[Cell, Guidance]) -> list[Cell]:
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


def committer(
    attempt: Attempt, planned: Sequence[Planned], queues: Queues | None
) -> Callable[[Batch], None]:
    """Take a wave's batches, given in call order, refusing a row kept in a later group's cell.

    Args:
        attempt: The attempt the wave fills.
        planned: Each planned call, in plan order; a group is one cell's calls.
        queues: The attempt's lines not yet consumed, None for an arm that prescribes nothing;
            each call's lines no kept record consumed go back to its cell's queue.

    Returns:
        The callable that takes each batch. It raises `EvaluationError` if a batch's cell is not
        its planned call's, and `WaveError` if a batch keeps a row in a cell a later group of the
        wave asked for, so a one-call-at-a-time run would have asked that group differently.
    """
    plan = [item.guidance.cell for item in planned]
    groups = list(dict.fromkeys(plan))
    taken: list[Batch] = []

    def commit(batch: Batch) -> None:
        first = batch.index - len(taken)
        call = planned[len(taken)]
        if batch.cell != call.guidance.cell:
            msg = (
                f"call {batch.index} answered cell {cell_key(batch.cell)}, but its planned call "
                f"asked cell {cell_key(call.guidance.cell)}"
            )
            raise EvaluationError(msg)
        prescription: Skeleton | None = call.skeleton
        taken.append(batch)
        before = attempt.held()
        consumed = attempt.take(batch, skeleton=prescription)
        if queues is not None and prescription is not None:
            lines = prescription.lines
            queues.give_back(
                batch.cell, [line for at, line in enumerate(lines) if at not in consumed]
            )
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


def give_up(
    attempt: Attempt, short: Mapping[Cell, int], *, name: str, budget: int
) -> dict[Cell, int]:
    """The short cells whose own budget, `budget` times the cell's quota, cannot cover them.

    A curating arm gives each cell its own budget, so a cell its curator keeps almost nothing of
    is given up without asking the rest of the attempt's budget one row at a time; each is logged
    as a `shortfall` line naming arm `name`.
    """
    found: dict[Cell, int] = {}
    for cell, rows in short.items():
        limit = budget * attempt.request.quota[cell]
        asked = attempt.asked_in[cell]
        if asked + rows > limit:
            found[cell] = rows
            logger.info(
                "shortfall arm=%s seed=%d cell=%s short=%d quota=%d asked=%d limit=%d",
                name,
                attempt.request.seed,
                cell_key(cell),
                rows,
                attempt.request.quota[cell],
                asked,
                limit,
            )
    return found
