"""No `RUNS` entry reads a frame whose field rows a unit edits under its own dataset name.

`tasks/direct-index.md`, What every unit relies on, lets a unit edit an existing dataset's field
rows only when no `RUNS` entry reads the dataset. ADRs `research/support2-frame`,
`research/nhanes-frame` and `research/physionet2012-frame` record SUPPORT2's, NHANES's and
PhysioNet 2012's frames changing under their own names by that exception.
"""

from __future__ import annotations

from bko.evaluate.data import Dataset
from bko.evaluate.named_runs import RUNS


UNRUN = frozenset({Dataset.SUPPORT2, Dataset.NHANES, Dataset.PHYSIONET2012})
"""The datasets whose field rows a unit edits under the exception."""


def test_no_named_run_reads_a_frame_edited_under_the_exception():
    reading = sorted(f"{name}: {run.dataset}" for name, run in RUNS.items() if run.dataset in UNRUN)
    assert not reading, (
        f"{reading} read a frame edited under its own name; tasks/direct-index.md, What every "
        "unit relies on, allows that edit only while no RUNS entry reads the dataset"
    )
