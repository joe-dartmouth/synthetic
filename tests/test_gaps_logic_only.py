"""E2's reading of the logic-only ablation: the grounding definitions never read a prior's value.

The per-subgroup and population-level definitions ground a statement on which concepts a dataset
fragment holds, not on its priors, so setting every dataset prior to 0.5, as logic alone would,
leaves their listings unchanged. The frequency-aware definition reads counts, not the fragment's
priors either, but it is not this reading's subject.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Collection

import pytest

import bko.gaps.detect
import bko.gaps.experiment
import bko.gaps.scan
import bko.gaps.shares
from bko.bkb.structure import BKB
from bko.gaps.observe import Observed, dataset_fragment
from tests.test_gaps_cli import SYNTHEA_EXPECTED, make_context, run


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

GROUNDING = ("per-subgroup", "population-level")


def at_a_half(observed: Observed, *, held: Collection[str] | None = None) -> BKB:
    """`dataset_fragment` with every prior it gives set to 0.5."""
    shares = dataset_fragment(observed, held=held)
    fragment = BKB(shares.name)
    for snode in shares.snodes:
        fragment.add_inode(*snode.head)
        fragment.add_snode(*snode.head, 0.5)
    assert len(fragment.snodes) == len(shares.snodes)
    return fragment


def grounding_rows(text: str) -> list[dict[str, str]]:
    return [row for row in csv.DictReader(io.StringIO(text)) if row["detector"] in GROUNDING]


def test_logic_priors_leave_subgroup_and_population_gaps_unchanged(tmp_path, monkeypatch):
    halved: list[str] = []

    def counted(observed: Observed, *, held: Collection[str] | None = None) -> BKB:
        halved.append(observed.name)
        return at_a_half(observed, held=held)

    for module in (bko.gaps.detect, bko.gaps.experiment, bko.gaps.scan, bko.gaps.shares):
        monkeypatch.setattr(module, "dataset_fragment", counted)
    code, out = run("synthea", tmp_path, make_context())
    assert code == 0
    assert halved, "no dataset fragment was built at 0.5"
    rows = grounding_rows(out)
    assert rows, "the fixture lists no grounding gap"
    assert rows == grounding_rows(SYNTHEA_EXPECTED)
