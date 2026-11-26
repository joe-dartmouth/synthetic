"""An E19-shaped supported run end to end: `bko-kb` and `hb-marginals` on `diabetes-supported`.

No test reaches the API or reads `data/`: the archive is the toy cohort `toy_archive` builds and
the releases are invented. ADR `research/matched-knowledge-controls` fixes the arms, and ADR
`research/dataset-variants` the dataset.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
from pathlib import Path
from typing import Final

import pytest

from bko.datasets.split import held_out_count, split
from bko.evaluate.__main__ import main
from bko.evaluate.attribution import Attribution
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, frame, load
from bko.evaluate.direct_arms import KB_ARM
from bko.evaluate.donor_attacks import donor_counts, match_groups
from bko.evaluate.experiment import PRIOR_SHIFT_ARM, Removal, gap_request
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.prepare import prepared
from bko.evaluate.run import method_arms
from bko.evaluate.run_context import named as named_run
from bko.evaluate.run_folder import read_folder
from bko.evaluate.runs_direct import E19_SUPPORTED_ARMS, E19_SUPPORTED_THIN
from bko.evaluate.scores import Source, parse
from bko.evaluate.shared import rebuilt
from bko.evaluate.shift import prior_shift
from bko.evaluate.trace_lines import select
from bko.gaps.scope import load_scope
from bko.generate.knowledge_draw import kb_check
from bko.generate.messages import SETTINGS
from bko.ontology.frames import DIABETES_FIELD_ROWS
from tests._diabetes_fixtures import TOY_CELLS, diabetes_releases, pin, toy_archive
from tests._evaluate_fixtures import FAST_METHODS
from tests._ontology_fixtures import RoutingOpener
from tests.test_evaluate_trace import _scored, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ARCHIVE: Final = toy_archive()
PIN: Final = pin(ARCHIVE, "uci-diabetes-130")
REMOVED: Final = ("6-7", "female", 1)
TOY_FILL: Final = TOY_CELLS["6-7", "Female", 1] - held_out_count(TOY_CELLS["6-7", "Female", 1])

TOY_E19: Final = dataclasses.replace(
    E19_SUPPORTED_THIN,
    holdout_seeds=(0, 1),
    removal=Removal(REMOVED, 0.9),
    fill_to=TOY_FILL,
    cutoff=0.4,
)
"""`e19-supported-thin` on two holdouts of the toy cohort, 0.9 of `(6-7, female, 1)` removed."""


def _context() -> Context:
    pins, served = diabetes_releases()
    return Context(
        open_url=RoutingOpener({**served, PIN.url: ARCHIVE}),
        runs={"e19-toy": NamedRun(TOY_E19, SETTINGS, Dataset.DIABETES_SUPPORTED)},
        arms={**method_arms(FAST_METHODS), PRIOR_SHIFT_ARM: prior_shift},
        releases=pins,
        send=never,
        diabetes=PIN,
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `e19-toy` folder, its data root beside it."""
    base = tmp_path_factory.mktemp("e19")
    folder = base / "run"
    argv = ["score", "e19-toy", "--candidates", str(folder / "holdout-candidates.jsonl")]
    return _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, context=_context(), out=out)
    return code, out.getvalue()


def test_k_draws_every_supported_column():
    kb_check(DIABETES_FIELD_ROWS, frame(Dataset.DIABETES_SUPPORTED))


def test_every_supported_arm_scores_on_the_variant(scored):
    scores = parse((scored / "scores.csv").read_text(encoding="utf-8"))
    for holdout in (0, 1):
        found = {s.arm for s in scores if s.holdout == holdout and s.source is Source.AUGMENTED}
        assert found == set(E19_SUPPORTED_ARMS)
    log = (scored / "scores.log").read_text(encoding="utf-8")
    assert "data dataset=diabetes-supported frame=diabetes-supported records=" in log
    kb = re.findall(rf"arm holdout=\d+ arm={KB_ARM} attempt=.*", log)
    assert kb
    assert all('"donors": false' in line for line in kb)
    hb = re.findall(r"arm holdout=\d+ arm=hb-marginals attempt=.*", log)
    assert hb
    assert all('"fits": ' in line for line in hb)
    assert re.search(r"hb holdout=0 fill=gap label=1 column=\S+ tau2=", log)
    assert "pool holdout=0 fill=gap method=hb-marginals key=6-7/female/1 " in log


def test_a_kb_fill_traces_to_knowledge_and_seed_alone(scored, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    data = scored.parent / "data"
    argv = ["trace", "e19-toy", "--folder", str(scored), "--data-dir", str(data)]
    code, out = _run([*argv, "--holdout", "0", "--arm", KB_ARM])
    assert code == 0
    found = [json.loads(line) for line in out.splitlines()]
    lines = read((scored / "holdout-candidates.jsonl").read_bytes()).candidates
    kept = [c for c in lines if (c.holdout, c.arm, c.outcome) == (0, KB_ARM, KEPT)]
    assert len(found) == len(kept) > 0
    assert any(line["concepts"] for line in found)
    (traced,) = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert traced.endswith(" kept_lines=true matches=true")

    path = tmp_path / "trace.jsonl"
    path.write_text(out, encoding="utf-8")
    code, csv = _run(["attribution", "e19-toy", "--trace", str(path)])
    assert code == 0
    counts = {row.split(",")[3]: int(row.split(",")[5]) for row in csv.splitlines()[1:]}
    assert counts[Attribution.KNOWLEDGE] > 0
    assert counts[Attribution.DONOR] == 0


def test_privacys_redraw_of_a_kb_fill_names_no_donor(scored):
    data_root = scored.parent / "data"
    named = named_run("e19-toy", _context())
    data = load(named.dataset, data_root, named)
    concepts = {row.concept for row in data.field_rows}
    ontologies, _ = load_scope(
        concepts, data_root, releases=named.releases, open_url=named.open_url
    )
    folder = read_folder(scored)
    fill = select(folder, holdout=0, arm=KB_ARM)
    spec = data.frame
    parts = split(data.records, seed=0, edges=spec.edges)
    train = prepared(parts, named.experiment, edges=spec.edges).ablated.train
    request = gap_request(train, named.experiment, seed=0, frame=spec)
    lines = list(folder.lines[0, KB_ARM, "gap"])
    rows = rebuilt(lines, dataclasses.replace(request, seed=fill.seed), where="the fill")
    assert len(rows) > 0
    counts = donor_counts(
        fill,
        rows,
        train,
        request,
        lines=lines,
        data=data,
        ontologies=ontologies,
        budget=named.experiment.budget,
        groups=match_groups(data.field_rows, frame=spec),
    )
    assert not counts
