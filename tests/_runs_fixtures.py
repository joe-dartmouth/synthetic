"""Run folders under `tmp_path`, one per register entry and one demo fill, with tiny artifacts."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Final

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.experiment import E1
from bko.evaluate.scores import Measure, Scope, Score, Source, render
from bko.gaps.definitions import Definition
from bko.generate.calls import ask
from bko.generate.candidates import Header, Request, render_header, render_lines
from bko.generate.errors import GenerationError
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Level
from tests._fusion_fixtures import young
from tests._generate_fixtures import GUIDANCE, ScriptedSend, reply


SCORES: Final = [
    Score(holdout, arm, source, measure, Scope.ALL, value + (0.1 if holdout % 2 else -0.1))
    for holdout in E1.holdout_seeds
    for arm, value in (("none", 0.5), ("bko", 0.6))
    for source, measure in ((Source.AUGMENTED, Measure.RECALL), (Source.AUGMENTED, Measure.AUC_PR))
] + [
    Score(holdout, arm, source, measure, Scope.ALL, count)
    for holdout in E1.holdout_seeds
    for arm, passed in (("bko", 1.0), ("bko-population", 0.5))
    for source, made in ((Source.CANDIDATES, 20.0), (Source.REPLICA_CANDIDATES, 30.0))
    for measure, count in (
        (Measure.GENERATED, made),
        (Measure.PASSED, made * passed),
        (Measure.DROPPED_RANGE, 1.0),
        (Measure.DROPPED_CELL, 0.0),
        (Measure.DROPPED_QUOTA, 0.0),
    )
]
"""Every E1 holdout of `none` and `bko`, utility rows first, each 0.1 below its mean on even
holdouts and 0.1 above on odd ones; then the gap and replica fill counts of `bko`, whose every
candidate passes, and `bko-population`, whose half do. `validity` requires every one of E1's
holdouts and reads both fills of a replica arm."""

CRITERIA: Final = (
    "criterion,scope,verdict,arm,against,measure,mean,low,high,refuting_high,holdouts,differing\n"
    "primary,all,inconclusive,bko,none,recall,0.1,-0.1,0.3,0.4,2,2\n"
)
STATEMENT: Final = "source,antecedent,antecedent_label,consequent,consequent_label"
CLOSURE: Final = (
    f"definition,{STATEMENT},age_band,sex,target,before,after\n"
    "per-subgroup,hpo,HP:1,a,HP:2,b,<55,female,1,True,False\n"
    "per-subgroup,hpo,HP:3,c,HP:2,b,<55,female,1,True,True\n"
    "frequency-aware,hpo,HP:4,d,HP:2,b,<55,male,1,False,True\n"
)
GAPS_SYNTHEA: Final = (
    f"ablation,detector,{STATEMENT},age_band,sex,target,before,after,truth\n"
    "cell,per-subgroup,mondo,M:1,a,M:2,b,<40,female,,True,True,False\n"
    "cell,count,,,,,,<40,female,,False,True,True\n"
)
GAPS_UCI: Final = (
    f"detector,{STATEMENT},age_band,sex,target,count\n"
    "per-subgroup,hpo,HP:1,a,HP:2,b,<55,female,0,\n"
    "count,,,,,,<55,female,1,2\n"
)
REPRODUCTION: Final = (
    "kind,item,paper,measured,status\nterm,M:1,x,M:1 x,match\nterm,M:2,y,M:2 y,erratum\n"
)
DEMO: Final = "app-20261031T000000000000Z-smote"


def write(folder: Path, role: str, text: str) -> None:
    """Write `text` as `role` in `folder`, creating the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / role).write_text(text, encoding="utf-8")


def call_log(path: Path) -> Path:
    """A complete call log of one call for 2 records, as `CallLog` writes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with CallLog(path, SETTINGS, frame="uci") as log, contextlib.suppress(GenerationError):
        ask(GUIDANCE, 2, send=ScriptedSend(reply([young(1)] * 2)), log=log, frame=UCI_FRAME)
    return path


CELL1: Final = ("<55", "female", 1)
HEADER: Final = Header(
    request=Request.COMPLETENESS,
    definition=Definition.SUBGROUP,
    guidance=Level.ONTOLOGY,
    uci="a" * 64,
    fingerprint="b" * 64,
    releases=(("HP", "c" * 64), ("MONDO", "d" * 64)),
    calls=(("<55/female/1", 2),),
)
"""A candidates header for the run browser, which reads but never checks it."""
HEAD: Final = render_header(HEADER)


def every_kind(root: Path) -> Path:
    """`root` holding one folder per register entry and one demo fill."""
    call_log(root / "e1-3" / "calls.jsonl")
    write(root / "e1-3", "scores.csv", render(SCORES))
    write(root / "e1-3", "criteria.csv", CRITERIA)
    call_log(root / "c1-2" / "calls.jsonl")
    write(root / "c1-2", "candidates.jsonl", HEAD + render_lines(1, CELL1, [young(1)] * 2))
    write(root / "c1-2-closure", "closure.csv", CLOSURE)
    write(root / "e2-5", "gaps-synthea.csv", GAPS_SYNTHEA)
    write(root / "e2-5", "gaps-uci.csv", GAPS_UCI)
    write(root / "r1-2", "reproduction.csv", REPRODUCTION)
    write(root / "g6-2", "scores.csv", render(SCORES))
    write(root / "g6-2", "server.log", "llama-server: listening\n")
    write(root / DEMO, "scores.csv", render(SCORES[:4]))
    write(root / DEMO, "training.csv", "record_id,age\n-1,40\n")
    (root / "notes").mkdir()
    write(root / "e1-3", "stray.txt", "not a role")
    return root
