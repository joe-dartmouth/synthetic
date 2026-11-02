"""`python -m bko.evaluate`: score writes the scores, compare judges them, stdout holds only CSV."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
import re
import subprocess
import sys
import urllib.error
import warnings
from fractions import Fraction

import pandas as pd
import pytest

from bko.evaluate.__main__ import (
    Context,
    configure_logging,
    main,
)
from bko.evaluate.classifier import MODELS
from bko.evaluate.criteria import CONTRAST_COLUMNS
from bko.evaluate.experiment import CRITERION_NAMES, E1, MESSAGES_ARMS
from bko.evaluate.logs import log_copies, log_means
from bko.evaluate.named_runs import DEFAULT_RUN, E3, RUNS, NamedRun
from bko.evaluate.provenance import commit, log_experiment
from bko.evaluate.scores import SCORE_COLUMNS, Measure, Scope, Score, Source
from bko.generate.api import default_async_send
from bko.generate.messages import SETTINGS, Level
from tests._baselines_fixtures import cleveland_text, full_records
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import FakeOpener, archive, pin
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TEXT = cleveland_text(GRADED)
LIGHT = dataclasses.replace(
    SMALL, arms=("none", "smote"), candidate="smote", comparators=(), replica_arms=()
)
SCORED = dataclasses.replace(SMALL, replica_arms=("smote", "adasyn", "ctgan"))
"""`SMALL` without TVAE's replica: the archive renumbers `GRADED`'s records, and on the splits that
gives, the tiny TVAE never draws some cells. `tests/test_evaluate_run.py` covers TVAE's replica."""


def score(tmp_path, *, experiment=SCORED, data=None, opener=None):
    data = data or archive(TEXT)
    out = io.StringIO()
    context = Context(
        open_url=opener or FakeOpener(data), uci=pin(data), experiment=experiment, arms=FAST_ARMS
    )
    code = main(["score", "--data-dir", str(tmp_path)], context=context, out=out)
    return code, out.getvalue()


@pytest.fixture(scope="module")
def scores_file(tmp_path_factory):
    code, out = score(tmp_path_factory.mktemp("data"))
    assert code == 0
    path = tmp_path_factory.mktemp("scores") / "scores.csv"
    path.write_text(out, encoding="utf-8")
    return path


def test_score_writes_every_arm_on_every_holdout(scores_file):
    frame = pd.read_csv(scores_file)
    assert tuple(frame.columns) == SCORE_COLUMNS
    assert set(frame["holdout"]) == set(SMALL.holdout_seeds)
    assert set(frame["arm"]) == set(SMALL.arms)
    augmented = frame[frame["source"] == "augmented"]
    assert {"recall-hgb", "precision-hgb", "auc-pr-hgb"} <= set(augmented["measure"])


def test_score_fits_every_model_by_default():
    assert Context().models is MODELS


def test_context_send_is_the_async_send():
    assert Context().send is default_async_send


def test_compare_judges_a_candidate_from_the_scores_file(scores_file):
    out = io.StringIO()
    argv = ["compare", str(scores_file), "--candidate", "tvae"]
    assert main(argv, context=Context(experiment=SMALL), out=out) == 0
    frame = pd.read_csv(io.StringIO(out.getvalue()))
    assert tuple(frame.columns) == CONTRAST_COLUMNS
    assert list(dict.fromkeys(frame["criterion"])) == ["primary", "secondary", "privacy"]
    assert set(frame["verdict"]) <= {"met", "not met", "inconclusive"}
    assert len(frame) == 2 * 2 * (len(SMALL.comparators) - 1) + 1


def test_compare_without_the_candidates_augmented_rows_exits_two(scores_file, tmp_path):
    frame = pd.read_csv(scores_file)
    path = tmp_path / "s.csv"
    frame[frame["arm"] != "bko"].to_csv(path, index=False)
    out = io.StringIO()
    assert main(["compare", str(path)], context=Context(experiment=SMALL), out=out) == 2
    assert out.getvalue() == ""


def test_compare_refuses_scores_from_other_holdouts_than_e1s(scores_file):
    out = io.StringIO()
    assert main(["compare", str(scores_file), "--candidate", "tvae"], out=out) == 2
    assert out.getvalue() == ""


CONTRAST_LINE = re.compile(
    r"contrast model=hgb criterion=(primary|secondary) scope=(all|flagged-subgroups) arm=tvae "
    r"against=\S+ measure=(recall-hgb|auc-pr-hgb) mean=-?\d+\.\d{6} low=-?\d+\.\d{6} "
    r"high=-?\d+\.\d{6} refuting_high=-?\d+\.\d{6} holdouts=2 differing=\d+$"
)


def _compared(path, caplog):
    out = io.StringIO()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        argv = ["compare", str(path), "--candidate", "tvae"]
        assert main(argv, context=Context(experiment=SMALL), out=out) == 0
    return out.getvalue(), [record.getMessage() for record in caplog.records]


def test_compare_logs_the_boosting_contrasts_only_when_the_scores_hold_them(
    scores_file, tmp_path, caplog
):
    table, lines = _compared(scores_file, caplog)
    contrasts = [line for line in lines if line.startswith("contrast ")]
    assert len(contrasts) == 2 * 2 * (len(SMALL.comparators) - 1)
    assert all(CONTRAST_LINE.match(line) for line in contrasts)
    assert any(line.startswith("mean arm=tvae measure=auc-pr-hgb scope=all") for line in lines)
    stripped = tmp_path / "stripped.csv"
    text = scores_file.read_text(encoding="utf-8")
    kept = [line for line in text.splitlines(keepends=True) if "-hgb," not in line]
    stripped.write_text("".join(kept), encoding="utf-8")
    alone, lines = _compared(stripped, caplog)
    assert alone == table
    assert not [line for line in lines if line.startswith("contrast ") or "-hgb" in line]


def test_an_arm_without_boosting_rows_is_skipped_and_the_table_kept(scores_file, tmp_path, caplog):
    table, _ = _compared(scores_file, caplog)
    text = scores_file.read_text(encoding="utf-8")
    kept = [
        line
        for line in text.splitlines(keepends=True)
        if not (line.split(",")[1] == "smote" and "-hgb," in line)
    ]
    partial = tmp_path / "partial.csv"
    partial.write_text("".join(kept), encoding="utf-8")
    alone, lines = _compared(partial, caplog)
    assert alone == table
    assert "contrast skip model=hgb arm=tvae against=smote reason=absent" in lines
    contrasts = [line for line in lines if line.startswith("contrast model=hgb")]
    assert len(contrasts) == 2 * 2 * (len(SMALL.comparators) - 2)
    assert not [line for line in contrasts if "against=smote" in line]


def test_a_scores_file_that_is_not_utf8_exits_three(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_bytes(b"\xff\xfe")
    out = io.StringIO()
    assert main(["compare", str(path)], out=out) == 3
    assert out.getvalue() == ""


def test_compare_logs_the_candidate_and_the_scores_digest(scores_file, caplog):
    with caplog.at_level(logging.INFO):
        main(
            ["compare", str(scores_file), "--candidate", "tvae"], context=Context(experiment=SMALL)
        )
    digest = hashlib.sha256(scores_file.read_bytes()).hexdigest()
    assert f"compare candidate=tvae dry_run=True scores_sha256={digest}" in caplog.text
    assert " training_fraction=1\n" in caplog.text
    assert "criterion=primary verdict=" in caplog.text


def test_compare_header_names_the_run_and_its_level(scores_file, caplog):
    digest = hashlib.sha256(scores_file.read_bytes()).hexdigest()
    no_messages = dataclasses.replace(
        SMALL,
        arms=tuple(arm for arm in SMALL.arms if arm not in MESSAGES_ARMS),
        replica_arms=tuple(arm for arm in SMALL.replica_arms if arm not in MESSAGES_ARMS),
        candidate="tvae",
        comparators=("none", "class-weight", "smote", "adasyn", "ctgan"),
        criteria=(),
    )
    runs = {
        "g3-cell": NamedRun(SMALL, dataclasses.replace(SETTINGS, guidance=Level.CELL)),
        "x1": NamedRun(no_messages),
    }
    context = Context(experiment=SMALL, runs=runs)
    with caplog.at_level(logging.INFO):
        for run in ("e1", "g3-cell", "x1"):
            main(["compare", run, str(scores_file), "--candidate", "tvae"], context=context)
    for run, level in (("e1", "summary"), ("g3-cell", "cell"), ("x1", "none")):
        assert (
            f"compare candidate=tvae dry_run={run != 'x1'} scores_sha256={digest} "
            f"run={run} level={level}"
        ) in caplog.text


def test_a_malformed_scores_file_exits_three(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    out = io.StringIO()
    assert main(["compare", str(path)], out=out) == 3
    assert out.getvalue() == ""


def test_a_missing_scores_file_exits_two(tmp_path):
    assert main(["compare", str(tmp_path / "absent.csv")], out=io.StringIO()) == 2


def test_nothing_to_fill_exits_two_and_writes_nothing(tmp_path):
    data = archive(cleveland_text(full_records(26)))
    assert score(tmp_path, experiment=LIGHT, data=data) == (2, "")


def test_an_unreachable_archive_exits_two(tmp_path):
    opener = FakeOpener(error=urllib.error.URLError("down"))
    assert score(tmp_path, opener=opener) == (2, "")


def test_an_archive_that_contradicts_its_pin_exits_three(tmp_path):
    data = archive(TEXT)
    tampered = data[:-1] + bytes([data[-1] ^ 1])
    context = Context(
        open_url=FakeOpener(tampered), uci=pin(data), experiment=LIGHT, arms=FAST_ARMS
    )
    out = io.StringIO()
    assert main(["score", "--data-dir", str(tmp_path)], context=context, out=out) == 3
    assert out.getvalue() == ""


def test_stderr_carries_the_split_quota_versions_and_digest(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        code, out = score(tmp_path, experiment=LIGHT)
    assert code == 0
    assert "split seed=1 train=146 test=35" in caplog.text
    assert "quota holdout=0 stratum=<55/female/1 rows=18" in caplog.text
    assert "versions commit=" in caplog.text
    assert f"sha256={hashlib.sha256(out.encode()).hexdigest()}" in caplog.text


def test_configure_logging_captures_warnings_and_quiets_sdv():
    loggers = [logging.getLogger(name) for name in ("sdv", "SingleTableSynthesizer")]
    levels = [logger.level for logger in loggers]
    try:
        configure_logging()
        assert [logger.level for logger in loggers] == [logging.WARNING] * 2
        assert warnings.showwarning.__module__ == "logging"
    finally:
        logging.captureWarnings(False)
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


def test_the_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "bko.evaluate", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "score" in result.stdout
    assert "compare" in result.stdout


DRIVER = """
import dataclasses, sys
from bko.evaluate.__main__ import Context, configure_logging, main
from bko.evaluate.run import method_arms
from tests._baselines_fixtures import cleveland_text
from tests._datasets_fixtures import FakeOpener, archive, pin
from tests._evaluate_fixtures import FAST_METHODS, GRADED, SMALL
data = archive(cleveland_text(GRADED))
experiment = dataclasses.replace(
    SMALL, arms=("none", "smote"), candidate="smote", comparators=(), replica_arms=("smote",)
)
configure_logging()
arms = method_arms(FAST_METHODS)
context = Context(open_url=FakeOpener(data), uci=pin(data), experiment=experiment, arms=arms)
sys.exit(main(["score", "--data-dir", sys.argv[1]], context=context))
"""
"""The command as `__main__` runs it, with the opener, the pin and the experiment replaced."""


def test_stdout_carries_only_the_csv(tmp_path):
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", DRIVER, str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert tuple(pd.read_csv(io.StringIO(result.stdout)).columns) == SCORE_COLUMNS
    assert "INFO bko.evaluate: scores rows=" in result.stderr


def test_a_mean_with_no_values_is_skipped(caplog):
    scores = [Score(0, "smote", Source.AUGMENTED, Measure.RECALL, Scope.ALL, 0.5)]
    with caplog.at_level(logging.INFO):
        log_means(scores)
    means = [m for m in caplog.messages if m.startswith("mean ")]
    assert means == ["mean arm=smote measure=recall scope=all value=0.500000 holdouts=1"]


def test_copies_are_logged_as_whole_counts(caplog):
    scores = [
        Score(seed, "smote", source, Measure.TEST_COPIES, Scope.ALL, value)
        for seed, source, value in ((0, Source.ADDED, 2.0), (1, Source.ADDED, 1.0))
    ]
    with caplog.at_level(logging.INFO):
        log_copies(scores)
    assert "copies arm=smote added=3 replica=0" in caplog.messages


def test_the_versions_line_names_the_commit_and_whether_the_tree_is_dirty(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        code, _ = score(tmp_path, experiment=LIGHT)
    assert code == 0
    line = next(m for m in caplog.messages if m.startswith("versions "))
    assert re.match(r"versions commit=[0-9a-f]{40} dirty=(true|false) \{", line)
    assert '"threadpoolctl": "' in line
    experiment = next(m for m in caplog.messages if m.startswith("experiment "))
    assert experiment.endswith(" models=lr,hgb")


@pytest.mark.parametrize(
    ("experiment", "fields"),
    [
        (E1, "holdouts=100 seeds=0-99 removal=none training_fraction=1 decontaminate=true "),
        (
            E3,
            (
                "holdouts=100 seeds=0-99 removal=>=65/female/1@0.9 training_fraction=1 "
                "decontaminate=false "
            ),
        ),
        (
            dataclasses.replace(E3, holdout_seeds=tuple(range(50, 100))),
            (
                "holdouts=50 seeds=50-99 removal=>=65/female/1@0.9 training_fraction=1 "
                "decontaminate=false "
            ),
        ),
        (
            dataclasses.replace(E1, fill_to=30),
            "holdouts=100 seeds=0-99 removal=none training_fraction=1 decontaminate=true ",
        ),
        (
            dataclasses.replace(E1, training_fraction=Fraction(1, 4)),
            "holdouts=100 seeds=0-99 removal=none training_fraction=1/4 decontaminate=true ",
        ),
    ],
)
def test_the_experiment_line_names_the_seeds_removal_and_copy_rule(caplog, experiment, fields):
    with caplog.at_level(logging.INFO):
        log_experiment("x", experiment, ["lr"], frame="f")
    line = next(m for m in caplog.messages if m.startswith("experiment "))
    assert line.startswith(f"experiment run=x frame=f {fields}arms={','.join(experiment.arms)} ")
    assert f" threshold={experiment.threshold} fill_to={experiment.fill_to} " in line


def test_the_experiment_line_names_a_cutoff_only_when_it_is_not_0_5(caplog):
    def line(experiment):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            log_experiment("x", experiment, ["lr"], frame="f")
        return next(m for m in caplog.messages if m.startswith("experiment "))

    assert " cutoff=" not in line(E1)
    named = line(dataclasses.replace(E1, cutoff=0.25))
    assert f" fill_to={E1.fill_to} cutoff=0.25 budget={E1.budget} " in named


def test_the_commit_is_unknown_without_git(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", absent)
    assert commit() == ("unknown", "unknown")


def test_stderr_lines_carry_their_time():
    root = logging.getLogger()
    handlers = list(root.handlers)
    root.handlers.clear()
    try:
        configure_logging()
        added = [h for h in root.handlers if h not in handlers]
        assert added
        assert added[0].formatter is not None
        assert added[0].formatter._fmt is not None
        assert added[0].formatter._fmt.startswith("%(asctime)s ")
    finally:
        logging.captureWarnings(False)
        root.handlers[:] = handlers


def test_compare_logs_every_replica_arms_utility_and_fidelity_means(scores_file, caplog):
    with caplog.at_level(logging.INFO):
        argv = ["compare", str(scores_file), "--candidate", "tvae"]
        assert main(argv, context=Context(experiment=SMALL), out=io.StringIO()) == 0
    frame = pd.read_csv(scores_file)
    measures = ["recall", "auc-pr", "quality", "column-shapes", "column-pair-trends"]
    rows = frame[(frame["source"] == "replica") & frame["measure"].isin(measures)]
    expected = {
        f"replica arm={arm} measure={measure} scope={scope} "
        f"value={sum(list(v)) / len(v):.6f} holdouts={len(v)}"
        for (arm, measure, scope), v in rows.groupby(["arm", "measure", "scope"])["value"]
    }
    assert expected
    assert {m for m in caplog.messages if m.startswith("replica ")} == expected


def test_the_default_run_is_e1_at_the_generators_settings():
    assert DEFAULT_RUN == "e1"
    assert RUNS["e1"] == NamedRun(E1)


def test_score_e1_writes_the_bytes_score_writes(tmp_path):
    data = archive(TEXT)
    plain, named = io.StringIO(), io.StringIO()
    context = Context(open_url=FakeOpener(data), uci=pin(data), experiment=SCORED, arms=FAST_ARMS)
    assert main(["score", "--data-dir", str(tmp_path)], context=context, out=plain) == 0
    assert main(["score", "e1", "--data-dir", str(tmp_path)], context=context, out=named) == 0
    assert named.getvalue() == plain.getvalue()


def test_a_named_run_without_criteria_scores_and_compares_with_no_verdict(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    quiet = dataclasses.replace(SCORED, criteria=())
    data = archive(TEXT)
    context = Context(
        open_url=FakeOpener(data),
        uci=pin(data),
        runs={"quiet": NamedRun(quiet)},
        arms=FAST_ARMS,
    )
    out = io.StringIO()
    assert main(["score", "quiet", "--data-dir", str(tmp_path)], context=context, out=out) == 0
    scored = tmp_path / "quiet.csv"
    scored.write_text(out.getvalue(), encoding="utf-8")
    compared = io.StringIO()
    with caplog.at_level(logging.INFO):
        code = main(
            ["compare", "quiet", str(scored), "--candidate", "tvae"], context=context, out=compared
        )
    assert code == 0
    assert list(pd.read_csv(io.StringIO(compared.getvalue()))["criterion"]) == []
    assert "run=quiet" in caplog.text
    assert "criteria=none" in caplog.text
    assert "verdict=" not in caplog.text
    judged = [m for m in caplog.messages if "criterion=" in m]
    assert judged
    assert all(m.startswith("unjudged ") for m in judged)


def test_an_unknown_run_name_exits_two_naming_the_runs(tmp_path, caplog):
    out = io.StringIO()
    with caplog.at_level(logging.ERROR):
        assert main(["score", "nope", "--data-dir", str(tmp_path)], out=out) == 2
    assert "no run named 'nope'" in caplog.text
    assert "e1" in caplog.text
    assert out.getvalue() == ""


def test_criteria_are_distinct_known_names():
    assert CRITERION_NAMES == ("primary", "secondary", "privacy", "validity")
    for bad in (("nope",), ("primary", "primary")):
        with pytest.raises(ValueError, match="criteria"):
            dataclasses.replace(SMALL, criteria=bad)


def test_a_run_naming_only_privacy_judges_only_privacy(scores_file):
    only = dataclasses.replace(SMALL, criteria=("privacy",))
    out = io.StringIO()
    context = Context(experiment=only)
    assert main(["compare", str(scores_file), "--candidate", "tvae"], context=context, out=out) == 0
    assert list(dict.fromkeys(pd.read_csv(io.StringIO(out.getvalue()))["criterion"])) == ["privacy"]
