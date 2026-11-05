"""`bko.app.steps`: each action over fakes; no test reaches the API or the network."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from bko.app import steps as steps_module
from bko.app.context import SPEND_CEILING, Context, configure_logging
from bko.app.steps import (
    Analysis,
    Outcome,
    Recorder,
    analyze,
    call_logs,
    experiment_for,
    fill_and_score,
    fillers,
    is_paid,
    run_folder,
    training_csv,
)
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.datasets.cells import cell_counts
from bko.datasets.fetch import default_data_dir
from bko.datasets.run_folders import RUN_ID, runs_root
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.evaluate.classifier import build
from bko.evaluate.criteria import BOOSTED
from bko.evaluate.data import digest
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1, METHOD_ARMS
from bko.evaluate.holdout_candidates import read
from bko.evaluate.named_runs import RUNS
from bko.evaluate.run import ARMS, method_arms
from bko.evaluate.scores import Measure, Source
from bko.gaps.definitions import Definition
from bko.generate.api import HTTP_LOGGERS
from bko.generate.errors import GenerationError, ReplayError
from bko.generate.flight import Limits
from bko.generate.messages import SETTINGS
from bko.generate.spend import cost
from tests._app_fixtures import app_context, never, one_fill
from tests._evaluate_fixtures import FAST_METHODS
from tests._fusion_fixtures import young
from tests._generate_fixtures import ScriptedSend, reply
from tests._guided_fixtures import YOUNG_SICK, YOUNG_WELL


def test_analyze_lists_the_flagged_cells_and_each_definitions_gaps(tmp_path):
    context, _ = app_context(tmp_path, send=never)
    analysis = analyze(context)
    assert {gap.cell for gap in analysis.counted} == {YOUNG_WELL, YOUNG_SICK}
    assert set(analysis.found) == set(Definition)
    assert analysis.counts.equals(cell_counts(analysis.parts.train, edges=UCI_FRAME.edges))
    assert analysis.parts.seed == 0


def test_analyze_reads_one_data_value_for_its_split_and_its_knowledge(tmp_path, monkeypatch):
    context, _ = app_context(tmp_path, send=never)
    loads = []

    def counting(*args, **kwargs):
        loads.append(args)
        return load_uci(*args, **kwargs)

    monkeypatch.setattr("bko.evaluate.data.load_uci", counting)
    monkeypatch.setattr("bko.datasets.split.load_uci", counting)
    analysis = analyze(context)
    assert len(loads) == 1
    records = analysis.data.records
    parts = analysis.parts
    assert pd.concat([parts.train, parts.test]).sort_index().equals(records.sort_index())
    assert analysis.knowledge.name == analysis.data.name


def test_analyze_fetches_nothing_once_the_data_is_present(tmp_path):
    context, opener = app_context(tmp_path, send=never)
    analyze(context)
    fetched = list(opener.calls)
    analyze(context)
    assert opener.calls == fetched


def test_the_default_roots_sit_at_the_checkout():
    context = Context()
    assert context.data_root() == default_data_dir()
    assert context.runs_root() == runs_root() == default_data_dir().parent / "runs"
    assert context.now().tzinfo is not None


def test_configure_logging_quiets_the_http_and_sdv_loggers(monkeypatch):
    """`captureWarnings` is stubbed: the real call is process-wide and outlives the test."""
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: recorded.append(kwargs))
    monkeypatch.setattr(logging, "captureWarnings", lambda _: None)
    loggers = [logging.getLogger(name) for name in (*QUIET_LOGGERS, *HTTP_LOGGERS)]
    for logger in loggers:
        monkeypatch.setattr(logger, "level", logger.level)
    configure_logging()
    [kwargs] = recorded
    handlers = kwargs["handlers"]
    assert isinstance(handlers, list)
    [handler] = handlers
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr
    assert kwargs["level"] == logging.INFO
    assert kwargs["force"] is True
    assert [logger.level for logger in loggers] == [logging.WARNING] * len(loggers)


def test_fillers_are_e1s_row_adding_arms_in_order():
    assert fillers(E1) == (
        "smote",
        "adasyn",
        "ctgan",
        "tvae",
        "bko",
        "bko-population",
        "bko-frequency",
    )
    assert fillers(RUNS["g5"].experiment)[-2:] == ("bko", "bko-skeleton")
    assert is_paid("bko-skeleton")
    assert not is_paid("bko-direct")


def test_experiment_for_puts_a_paid_fill_first_beside_every_method():
    paid = experiment_for("bko", Context())
    assert paid.arms == ("bko", *METHOD_ARMS)
    assert paid.candidate == "bko"
    assert paid.comparators == METHOD_ARMS
    assert paid.replica_arms == ()
    assert paid.spend_ceiling == SPEND_CEILING
    assert paid.holdout_seeds == E1.holdout_seeds
    free = experiment_for("smote", Context())
    assert free.arms == METHOD_ARMS
    assert free.candidate == "smote"
    assert "smote" not in free.comparators


@pytest.mark.parametrize("name", ["none", "class-weight"])
def test_experiment_for_refuses_an_arm_that_adds_no_rows(name):
    with pytest.raises(ValueError, match="adds rows"):
        experiment_for(name, Context())


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _edited(log: Path, to: Path, edit: Callable[[list[dict]], None]) -> Path:
    lines = _lines(log)
    edit(lines)
    to.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return to


def test_recorder_has_no_fill_before_its_runner_returns():
    with pytest.raises(EvaluationError, match="no fill"):
        _ = Recorder(ARMS["smote"]).fill


def test_a_method_fill_calls_no_api_and_keeps_its_run_folder(tmp_path):
    context, _ = app_context(tmp_path, send=never)
    analysis = analyze(context)
    outcome = fill_and_score(analysis, "smote", context)
    assert outcome.log is None
    assert len(outcome.baseline.added) == 28
    assert RUN_ID.fullmatch(outcome.folder.name)
    assert sorted(path.name for path in outcome.folder.iterdir()) == ["scores.csv", "training.csv"]
    training = (outcome.folder / "training.csv").read_text(encoding="utf-8")
    assert training == training_csv(analysis, outcome)
    assert {score.arm for score in outcome.scores} == {"none", "smote"}


def test_the_app_scores_with_the_contexts_regression_only(tmp_path):
    built = []

    def factory(frame):
        built.append(frame)
        return build(frame)

    context, _ = app_context(tmp_path, send=never, factory=factory)
    outcome = fill_and_score(analyze(context), "smote", context)
    assert built
    assert not {s.measure for s in outcome.scores} & set(BOOSTED)
    assert Measure.RECALL in {s.measure for s in outcome.scores}


def test_a_context_missing_a_method_runner_is_refused_before_any_call(tmp_path):
    context, _ = app_context(tmp_path, send=never, arms={"none": method_arms(FAST_METHODS)["none"]})
    analysis = analyze(context)
    with pytest.raises(EvaluationError, match="no runner"):
        fill_and_score(analysis, "bko", context)
    assert not (tmp_path / "runs").exists()


def test_a_context_missing_the_filling_method_is_refused(tmp_path):
    context, _ = app_context(tmp_path, send=never, arms={"none": method_arms(FAST_METHODS)["none"]})
    analysis = analyze(context)
    with pytest.raises(EvaluationError, match="no runner"):
        fill_and_score(analysis, "smote", context)
    assert not (tmp_path / "runs").exists()


def test_a_bko_fill_logs_its_calls_and_scores_its_candidates(tmp_path):
    send = one_fill()
    context, _ = app_context(tmp_path, send=lambda: send)
    analysis = analyze(context)
    outcome = fill_and_score(analysis, "bko", context)
    replies = (reply([young(0)] * 12), reply([young(1)] * 16))
    assert outcome.calls == 2
    assert outcome.usd == pytest.approx(sum(cost(r.usage, SETTINGS.model) for r in replies))
    assert outcome.log == outcome.folder / "calls.jsonl"
    assert outcome.log.stat().st_mode & 0o777 == 0o600
    assert outcome.folder.stat().st_mode & 0o777 == 0o700
    generated = [
        score.value
        for score in outcome.scores
        if score.arm == "bko"
        and score.source is Source.CANDIDATES
        and score.measure is Measure.GENERATED
    ]
    assert generated == [28.0]
    send.assert_consumed()
    assert [reading.definition for reading in outcome.closures] == list(Definition)
    assert outcome.replayed is False
    assert sorted(path.name for path in outcome.folder.iterdir()) == [
        "calls.jsonl",
        "holdout-candidates.jsonl",
        "scores.csv",
        "training.csv",
    ]
    header = read((outcome.folder / "holdout-candidates.jsonl").read_bytes()).header
    assert header["run"] == "app"
    assert analysis.data.frame is UCI_FRAME
    assert analysis.data.records.equals(load_uci(context.data_root(), source=context.uci))
    assert (header["frame"], header["data"]) == ("uci", analysis.data.digest)
    assert header["data"] == digest(analysis.data.records)


def test_a_bko_fill_sends_one_call_at_a_time_as_the_app_run(tmp_path, monkeypatch):
    seen: list[dict[str, object]] = []
    real = steps_module.guided_arms

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(steps_module, "guided_arms", recording)
    send = one_fill()
    context, _ = app_context(tmp_path, send=lambda: send)
    analysis = analyze(context)
    fill_and_score(analysis, "bko", context)
    [kwargs] = seen
    assert kwargs["run"] == "app"
    idle = experiment_for("bko", context).idle_limit
    assert kwargs["limits"] == Limits(in_flight=1, idle=idle)
    lines = (tmp_path / "runs").glob("*/calls.jsonl")
    assert [json.loads(path.read_text().splitlines()[0])["in_flight"] for path in lines] == [1]


def test_the_ceiling_stops_a_bko_fill_before_the_next_call(tmp_path):
    send = one_fill()
    ceiling = cost(reply([young(0)] * 12).usage, SETTINGS.model)
    context, _ = app_context(tmp_path, send=lambda: send, ceiling=ceiling)
    analysis = analyze(context)
    with pytest.raises(EvaluationError, match="ceiling") as caught:
        fill_and_score(analysis, "bko", context)
    assert len(send.calls) == 1
    (folder,) = (tmp_path / "runs").iterdir()
    assert sorted(path.name for path in folder.iterdir()) == [
        "calls.jsonl",
        "holdout-candidates.jsonl",
    ]
    assert caught.value.__notes__ == [
        f"The fill's run folder is {folder}; it spent {ceiling:.6f} USD over 1 calls."
    ]


def test_a_call_with_no_response_stops_the_fill(tmp_path):
    failure = GenerationError("no Anthropic credential: set ANTHROPIC_API_KEY", attempts=1)
    context, _ = app_context(tmp_path, send=lambda: ScriptedSend(failure))
    analysis = analyze(context)
    with pytest.raises(GenerationError, match="no Anthropic credential"):
        fill_and_score(analysis, "bko", context)
    (folder,) = (tmp_path / "runs").iterdir()
    assert _lines(folder / "calls.jsonl")[-1]["status"] == "failed"


def _recorded(tmp_path: Path) -> tuple[Context, Analysis, Outcome]:
    send = one_fill()
    context, _ = app_context(tmp_path, send=lambda: send)
    analysis = analyze(context)
    return context, analysis, fill_and_score(analysis, "bko", context)


def test_a_replayed_fill_equals_the_paid_one_and_pays_nothing(tmp_path):
    context, analysis, recorded = _recorded(tmp_path)
    assert recorded.log is not None
    replaying = dataclasses.replace(context, send=never)
    replayed = fill_and_score(analysis, "bko", replaying, replay=recorded.log)
    pd.testing.assert_frame_equal(replayed.baseline.added, recorded.baseline.added)
    assert replayed.replayed is True
    assert replayed.folder != recorded.folder
    assert replayed.log is not None
    digest = hashlib.sha256(recorded.log.read_bytes()).hexdigest()
    assert _lines(replayed.log)[0]["replay"] == digest


def test_a_replay_with_calls_left_over_is_refused(tmp_path):
    context, analysis, recorded = _recorded(tmp_path)
    assert recorded.log is not None

    def extend(lines: list[dict]) -> None:
        end = lines.pop()
        for kind in ("request", "reply"):
            last = next(line for line in reversed(lines) if line["kind"] == kind)
            lines.append({**last, "call": 3})
        lines.append(end)

    log = _edited(recorded.log, tmp_path / "more.jsonl", extend)
    with pytest.raises(ReplayError, match="1 logged calls were not replayed"):
        fill_and_score(analysis, "bko", dataclasses.replace(context, send=never), replay=log)


def test_a_replay_whose_request_moved_is_refused(tmp_path):
    context, analysis, recorded = _recorded(tmp_path)
    assert recorded.log is not None

    def move(lines: list[dict]) -> None:
        first = next(line for line in lines if line["kind"] == "request")
        first["params"]["max_tokens"] = 1

    log = _edited(recorded.log, tmp_path / "moved.jsonl", move)
    with pytest.raises(ReplayError, match="differs"):
        fill_and_score(analysis, "bko", dataclasses.replace(context, send=never), replay=log)


def test_each_fill_gets_its_own_run_folder(tmp_path):
    context, _ = app_context(tmp_path, send=never)
    first, second = run_folder(context, "bko"), run_folder(context, "bko")
    assert first != second
    assert RUN_ID.fullmatch(first.name)
    assert RUN_ID.fullmatch(second.name)
    assert (tmp_path / "runs").stat().st_mode & 0o777 == 0o700


def test_call_logs_list_only_the_arms_logs_newest_first(tmp_path):
    send = ScriptedSend(
        *[step for _ in range(2) for step in (reply([young(0)] * 12), reply([young(1)] * 16))]
    )
    context, _ = app_context(tmp_path, send=lambda: send)
    assert call_logs(context, "bko") == ()
    analysis = analyze(context)
    first = fill_and_score(analysis, "bko", context)
    second = fill_and_score(analysis, "bko", context)
    other = tmp_path / "runs" / "app-20261101T000009000000Z-bko-frequency"
    other.mkdir()
    (other / "calls.jsonl").write_text("{}\n", encoding="utf-8")
    assert call_logs(context, "bko") == (second.log, first.log)
