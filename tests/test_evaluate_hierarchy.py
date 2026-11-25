"""The `bko-hier` arms in a scored run: their arm lines, their trace and E16's runs.

The toy Diabetes cohort is scored with a MONDO release that puts `MONDO:0005016` under
`MONDO:0005300` and `MONDO:0005607` under `MONDO:0005002`, the two `is_a` pairs of the real
release among the frame's concepts, so each arm draws two broad concepts given their narrow ones.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset
from bko.evaluate.direct_arms import DIRECT_ARM, HIER_0_ARM, HIER_CLOSED_ARM, HIER_W_ARM
from bko.evaluate.experiment import E1, REAL_ARM, Experiment, Removal
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import RUNS, NamedRun
from bko.evaluate.reference import REFERENCE
from bko.evaluate.runs_diabetes import E9_EMPTY, E9_THIN
from bko.evaluate.runs_direct import E16_EMPTY, E16_THIN
from bko.gaps.observe import Cell, cell_key
from bko.generate import direct
from bko.generate.hierarchy import PairPart, log_pairs
from bko.generate.messages import SETTINGS, AnySend
from bko.ontology.icd9 import ICD9_MAPPINGS
from tests._diabetes_fixtures import diabetes_releases, pin, toy_archive
from tests._ontology_fixtures import RoutingOpener
from tests.test_evaluate_diabetes import PREVALENCE, REMOVED
from tests.test_evaluate_trace import _scored


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ARCHIVE: Final = toy_archive()
PIN: Final = pin(ARCHIVE, "uci-diabetes-130")

PAIRS: Final = {"MONDO:0005016": "MONDO:0005300", "MONDO:0005607": "MONDO:0005002"}
"""Each narrow concept with its broad one."""

MONDO: Final = "\n".join(
    [
        "format-version: 1.2",
        "data-version: releases/2026-01-01",
        "ontology: mondo",
        "",
        *("[Term]", "id: MONDO:9000001", "name: disease", ""),
        *(
            line
            for mapping in ICD9_MAPPINGS
            for line in (
                "[Term]",
                f"id: {mapping.concept}",
                f"name: {mapping.label}",
                f"is_a: {PAIRS.get(mapping.concept, 'MONDO:9000001')}",
                "",
            )
        ),
    ]
)

HIER_RUN: Final = dataclasses.replace(
    E1,
    holdout_seeds=(0, 1),
    arms=("none", REAL_ARM, DIRECT_ARM, HIER_0_ARM, HIER_W_ARM, HIER_CLOSED_ARM),
    candidate=HIER_W_ARM,
    comparators=(DIRECT_ARM, HIER_0_ARM, "none"),
    replica_arms=(),
    criteria=(),
    removal=Removal(REMOVED, 0.9),
    decontaminate=False,
    cutoff=PREVALENCE,
)
"""E16's arms on two holdouts of the toy cohort, 0.9 of `(6-7, female, 1)` removed."""

OWN: Final = {
    HIER_0_ARM: {"hierarchy": 0.0},
    HIER_W_ARM: {"hierarchy": 20.0},
    HIER_CLOSED_ARM: {"closed": True},
}
"""Each `bko-hier` arm's params beyond `bko-direct`'s."""

PART_FIELDS: Final = {field.name for field in dataclasses.fields(PairPart)}


def _never() -> AnySend:
    msg = "a conditional is_a run never reaches the API"
    raise AssertionError(msg)


def _context(experiment: Experiment = HIER_RUN) -> Context:
    pins, served = diabetes_releases()
    mondo = MONDO.encode()
    pins = {**pins, "MONDO": pin(mondo, name="mondo")}
    return Context(
        open_url=RoutingOpener({**served, pins["MONDO"].url: mondo, PIN.url: ARCHIVE}),
        runs={"toy": NamedRun(experiment, SETTINGS, Dataset.DIABETES)},
        releases=pins,
        send=_never,
        diabetes=PIN,
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `toy` folder of `HIER_RUN`, its data root beside it.

    Each cell's `PairPart`s as the draw logs them are kept in `parts.json` beside the folder, by
    holdout, pseudo-count and cell key.
    """
    base = tmp_path_factory.mktemp("hier")
    folder = base / "run"
    argv = ["score", "toy", "--candidates", str(folder / "holdout-candidates.jsonl")]
    drawn: dict[str, dict[str, dict[str, object]]] = {}

    def keep(holdout: int, fill: str, cell: Cell, parts: Mapping[str, PairPart]) -> None:
        (pseudo,) = {part.pseudo for part in parts.values()}
        found: dict[str, dict[str, object]] = {
            broad: dataclasses.asdict(part) for broad, part in parts.items()
        }
        assert drawn.setdefault(f"{holdout} {pseudo} {cell_key(cell)}", found) == found
        log_pairs(holdout, fill, cell, parts)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(direct, "log_pairs", keep)
        _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    (base / "parts.json").write_text(json.dumps(drawn), encoding="utf-8")
    return folder


def _trace(folder: Path, *args: str) -> tuple[int, list[dict]]:
    out = io.StringIO()
    data = str(folder.parent / "data")
    argv = ["trace", "toy", "--folder", str(folder), "--data-dir", data, "--holdout", "0"]
    code = main([*argv, *args], context=_context(), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def _params(log: str, arm: str) -> list[dict[str, object]]:
    pattern = rf"arm holdout=\d arm={arm} attempt=.* params=(\{{.*\}})$"
    return [json.loads(line) for line in re.findall(pattern, log, re.MULTILINE)]


def test_each_hier_arm_logs_only_its_nondefault_fields(scored):
    log = (scored / "scores.log").read_text(encoding="utf-8")
    direct = _params(log, DIRECT_ARM)
    assert len(direct) == 2
    for arm, own in OWN.items():
        params = _params(log, arm)
        extra = [
            {k: v for k, v in p.items() if k not in d} for p, d in zip(params, direct, strict=True)
        ]
        assert extra == [own, own]
        assert all({k: p[k] for k in d} == d for p, d in zip(params, direct, strict=True))
        token = " ".join(f"{name}={str(value).lower()}" for name, value in own.items())
        assert re.search(rf"direct holdout=0 fill=gap key=\S+ .* {token} presence=shares", log)
    assert re.search(r"^.*hier holdout=0 fill=gap key=\S+ pairs=2 clamped=\d+$", log, re.MULTILINE)


def test_a_hier_fill_traces_to_its_arm_sha(scored, caplog):
    """Each record's `PairPart`s are the ones its draw logged at its cell, value for value."""
    caplog.set_level(logging.INFO)
    drawn = json.loads((scored.parent / "parts.json").read_text(encoding="utf-8"))
    lines = read((scored / "holdout-candidates.jsonl").read_bytes()).candidates
    for arm, own in OWN.items():
        caplog.clear()
        code, found = _trace(scored, "--arm", arm)
        assert code == 0
        kept = [c for c in lines if (c.holdout, c.arm, c.outcome) == (0, arm, KEPT)]
        assert len(found) == len(kept) > 0
        (line,) = [m for m in caplog.messages if m.startswith("trace arm ")]
        assert f" arm={arm} " in line
        assert line.endswith(" kept_lines=true matches=true")
        pseudo = own.get("hierarchy")
        for record in found:
            broad = {f["concept"]: f["hierarchy"] for f in record["concepts"] if "hierarchy" in f}
            assert set(broad) == set(PAIRS.values())
            for concept, part in broad.items():
                assert set(part) == PART_FIELDS
                assert part["given"] == {b: n for n, b in PAIRS.items()}[concept]
                assert part["pseudo"] == pseudo
                if pseudo is None:
                    assert (part["conditional"],) == (1.0,)
            assert broad == drawn[f"0 {pseudo} {record['request']['cell']}"]


def test_a_bko_direct_trace_names_no_pair_parts(scored):
    code, found = _trace(scored, "--arm", DIRECT_ARM)
    assert code == 0
    assert found
    assert not any("hierarchy" in fact for line in found for fact in line["concepts"])


@pytest.mark.parametrize(
    ("arm", "old", "new", "says"),
    [
        (HIER_0_ARM, '"hierarchy": 0.0', '"hierarchy": 0', "draws hierarchy 0, not 0.0"),
        (HIER_W_ARM, '"hierarchy": 20.0', '"hierarchy": 20', "draws hierarchy 20, not 20.0"),
        (HIER_CLOSED_ARM, '"closed": true', '"closed": 1', "draws closed 1, not True"),
    ],
    ids=["hier-0", "hier-w", "closed"],
)
def test_a_hier_fill_whose_hierarchy_param_is_an_int_exits_3(
    scored, tmp_path, caplog, arm, old, new, says
):
    shutil.copytree(scored.parent, tmp_path / "copy")
    folder = tmp_path / "copy" / "run"
    log = folder / "scores.log"

    def edit(found: re.Match[str]) -> str:
        assert old in found[0]
        return found[0].replace(old, new)

    pattern = rf"arm holdout=0 arm={arm} attempt=.*"
    text, count = re.subn(pattern, edit, log.read_text(encoding="utf-8"))
    assert count == 1
    log.write_text(text, encoding="utf-8")
    assert _trace(folder, "--arm", arm) == (3, [])
    assert f"{arm}'s arm line {says}" in caplog.text


@pytest.mark.parametrize(("hier", "e9"), [(E16_THIN, E9_THIN), (E16_EMPTY, E9_EMPTY)])
def test_e16_is_e9_with_the_three_hier_arms_beside_bko_direct(hier, e9):
    assert hier.arms == ("none", REAL_ARM, DIRECT_ARM, HIER_0_ARM, HIER_W_ARM, HIER_CLOSED_ARM)
    assert (hier.candidate, hier.comparators) == (HIER_W_ARM, (DIRECT_ARM, HIER_0_ARM, "none"))
    assert (hier.replica_arms, hier.replica_rows) == ((), None)
    same = ("arms", "candidate", "comparators", "replica_arms", "replica_rows")
    assert dataclasses.replace(hier, **{name: getattr(e9, name) for name in same}) == e9


@pytest.mark.parametrize("fraction", ["thin", "empty"])
def test_each_e16_run_reads_e6s_frame_against_bko_direct(fraction):
    runs = [name for name in RUNS if re.fullmatch(rf"e16-{fraction}[a-j]?", name)]
    assert len(runs) == 11
    assert {RUNS[name].dataset for name in runs} == {Dataset.DIABETES}
    assert REFERENCE[f"e16-{fraction}"] == DIRECT_ARM
