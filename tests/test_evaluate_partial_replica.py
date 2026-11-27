"""`Experiment.partial_replica`: a replica fill short on its last attempt fills partially.

ADR `research/e1-evaluation-protocol` records the rule; `bko.evaluate.filling.fill_arm` holds it.
"""

from __future__ import annotations

from typing import Final

import pytest

from bko.evaluate.experiment import E1
from bko.evaluate.provenance import experiment_line
from bko.evaluate.runs_published import P1


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

E1_LINE: Final = (
    "experiment run=e1 frame=uci holdouts=100 seeds=0-99 removal=none training_fraction=1"
    " decontaminate=true arms=none,class-weight,smote,adasyn,ctgan,tvae,bko,bko-population,"
    "bko-frequency replica=smote,adasyn,ctgan,tvae,bko,bko-population,bko-frequency threshold=20"
    " fill_to=20 budget=3 validity=19/20 spend_ceiling=25.00"
    " criteria=primary,secondary,privacy,validity models=lr,hgb"
)
"""E1's `experiment` line, captured at the commit before `partial_replica` existed."""

P1_LINE: Final = (
    "experiment run=p1-id frame=tableshift holdouts=10 seeds=0-9 removal=none training_fraction=1"
    " decontaminate=false arms=none,class-weight,smote,adasyn,ctgan,tvae,gaussian-copula,cart,"
    "bayes-net,tabddpm,label-bootstrap,bko-direct replica=smote,adasyn,ctgan,tvae,gaussian-copula,"
    "cart,bayes-net,tabddpm,label-bootstrap,bko-direct threshold=20 fill_to=20 budget=3"
    " validity=19/20 spend_ceiling=25.00 criteria=none models=lr,hgb"
)
"""P1's `experiment` line under `p1-id`, captured at the same commit."""


def test_the_default_experiment_line_is_unchanged():
    assert experiment_line("e1", E1, ["lr", "hgb"], frame="uci") == E1_LINE
    assert experiment_line("p1-id", P1, ["lr", "hgb"], frame="tableshift") == P1_LINE
