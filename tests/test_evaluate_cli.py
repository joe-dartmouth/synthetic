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

import pandas as pd
import pytest

from bko.evaluate.__main__ import (
    Context,
    _commit,
    configure_logging,
    main,
)
from bko.evaluate.criteria import CONTRAST_COLUMNS
from bko.evaluate.experiment import CRITERION_NAMES, DEFAULT_RUN, E1, RUNS, NamedRun
from bko.evaluate.logs import log_copies, log_means
from bko.evaluate.scores import SCORE_COLUMNS, Measure, Scope, Score, Source
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
    assert "criterion=primary verdict=" in caplog.text


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


def test_the_commit_is_unknown_without_git(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", absent)
    assert _commit() == ("unknown", "unknown")


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
    assert "criterion=" not in caplog.text


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
