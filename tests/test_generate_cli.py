"""`python -m bko.generate` over an archive and releases built in tmp_path, through a fake send."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.gaps.definitions import Definition
from bko.gaps.observe import Cell, cell_key
from bko.generate.__main__ import Context, configure_logging, main
from bko.generate.calls import chunks
from bko.generate.errors import GenerationError
from bko.generate.guidance import Basis
from bko.generate.messages import SETTINGS, Call, Level, Reply, Send
from bko.generate.prompt import BASES
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import archive, cleveland_line, pin
from tests._fusion_fixtures import UCI_FULL, UCI_X, young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._generate_fixtures import SIZES, ScriptedSend, reply, uci
from tests._ontology_fixtures import RoutingOpener


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

UCI = uci({}, "4.0")

REQUEST: dict[Cell, int] = {("<55", "female", 0): 12, ("<55", "female", 1): 20}
"""The count-based request: each flagged cell asked for its shortfall below twenty."""

RECORD = {"age": 41, "sex": 0}
"""A decoded item; the command passes items through unchecked."""

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_context(send: Send, **change) -> Context:
    pins, served = ladder_releases()
    uci = pin(UCI)
    context = Context(
        open_url=RoutingOpener({**served, uci.url: UCI}),
        uci=uci,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        send=lambda: send,
        now=lambda: NOW,
    )
    return dataclasses.replace(context, **change)


def completeness_context(send: Send) -> Context:
    _, served = ladder_releases()
    uci = pin(UCI_X)
    return make_context(send, open_url=RoutingOpener({**served, uci.url: UCI_X}), uci=uci)


def two_replies() -> ScriptedSend:
    return ScriptedSend(reply([RECORD] * 12), reply([RECORD] * 19))


def run(tmp_path: Path, context: Context, *extra: str) -> tuple[int, str, Path]:
    log = tmp_path / "run.jsonl"
    out = io.StringIO()
    argv = ["per-subgroup", "--log", str(log), "--data-dir", str(tmp_path / "data"), *extra]
    code = main(argv, context=context, out=out)
    return code, out.getvalue(), log


def log_lines(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def asked_per_cell(log: Path) -> dict[str, int]:
    asked: dict[str, int] = {}
    for line in log_lines(log):
        if line["kind"] == "request":
            cell, rows = line["cell"], line["asked"]
            assert isinstance(cell, str)
            assert isinstance(rows, int)
            asked[cell] = asked.get(cell, 0) + rows
    return asked


def test_the_command_asks_each_flagged_cell_for_its_shortfall_and_prints_one_line_per_candidate(
    tmp_path,
):
    send = two_replies()
    code, out, log = run(tmp_path, make_context(send))
    assert code == 0
    assert asked_per_cell(log) == {"<55/female/0": 12, "<55/female/1": 20}
    assert [(call.sex, call.label) for call in send.calls] == [(0, 0), (0, 1)]
    assert len(out.splitlines()) == 1 + sum(REQUEST.values())
    send.assert_consumed()


def test_the_request_equals_e1s_gap_request(tmp_path):
    from bko.evaluate.experiment import E1, gap_request

    code, _, log = run(tmp_path, make_context(two_replies()))
    assert code == 0
    train = split(load_uci(tmp_path / "data", source=pin(UCI)), edges=UCI_FRAME.edges).train
    expected = gap_request(train, E1, seed=0, frame=UCI_FRAME).quota
    assert asked_per_cell(log) == {cell_key(cell): rows for cell, rows in expected.items()}
    assert dict(expected) == REQUEST


def test_generate_cli_prompts_with_the_training_split_bounds(tmp_path):
    from bko.generate.guidance import intervals

    code, _, log = run(tmp_path, make_context(two_replies()), "--guidance", "summary")
    assert code == 0
    parts = split(load_uci(tmp_path / "data", source=pin(UCI)), edges=UCI_FRAME.edges)
    youngest = parts.train["age"].min()
    users = {
        line["cell"]: line["params"]["messages"][0]["content"]
        for line in log_lines(log)
        if line["kind"] == "request"
    }
    assert set(users) == {cell_key(cell) for cell in REQUEST}
    for cell in REQUEST:
        user = users[cell_key(cell)]
        assert f"age {youngest:g} to 54 years." in user
        bounds = intervals(cell, parts.train, frame=UCI_FRAME)
        within = "; ".join(
            f"{name} {low:g} to {high:g} {unit}"
            for (name, (low, high)), unit in zip(
                bounds.items(), ("years", "mm Hg", "mg/dL", "beats per minute", "mm"), strict=True
            )
        )
        assert f"Keep each numeric field within: {within}." in user


def test_the_completeness_request_asks_twenty_of_each_per_subgroup_gap_cell(tmp_path):
    send = ScriptedSend(reply([young(0, cp=4)] * 20), reply([young(1, cp=1)] * 20))
    code, out, log = run(tmp_path, completeness_context(send), "--request", "completeness")
    assert code == 0
    assert asked_per_cell(log) == {"<55/female/0": 20, "<55/female/1": 20}
    assert [(call.sex, call.label) for call in send.calls] == [(0, 0), (0, 1)]
    assert len(out.splitlines()) == 41
    send.assert_consumed()


def test_the_default_request_is_e1s_not_completeness(tmp_path):
    send = two_replies()
    code, _, log = run(tmp_path, completeness_context(send))
    assert code == 0
    assert asked_per_cell(log) == {"<55/female/0": 12, "<55/female/1": 20}


def test_the_request_is_logged_with_each_cells_rows(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    send = ScriptedSend(reply([young(0, cp=4)] * 20), reply([young(1, cp=1)] * 20))
    code, _, _ = run(tmp_path, completeness_context(send), "--request", "completeness")
    assert code == 0
    messages = [record.getMessage() for record in caplog.records]
    assert "request name=completeness cells=2 rows=40" in messages
    assert "quota key=<55/female/0 rows=20" in messages
    assert "quota key=<55/female/1 rows=20" in messages


@pytest.mark.parametrize("definition", ["population-level", "frequency-aware"])
def test_the_completeness_request_refuses_another_guidance_definition(tmp_path, definition):
    send = ScriptedSend()
    log = tmp_path / "run.jsonl"
    argv = [definition, "--request", "completeness", "--log", str(log)]
    with pytest.raises(SystemExit) as raised:
        main(argv, context=completeness_context(send), out=io.StringIO())
    assert raised.value.code == 2
    assert not log.exists()
    assert send.calls == []


def test_a_completeness_request_with_nothing_to_ask_exits_two_before_any_call(tmp_path, caplog):
    _, served = ladder_releases()
    uci = pin(UCI_FULL)
    send = ScriptedSend()
    context = make_context(send, open_url=RoutingOpener({**served, uci.url: UCI_FULL}), uci=uci)
    code, out, log = run(tmp_path, context, "--request", "completeness")
    assert (code, out) == (2, "")
    assert any(
        record.levelno == logging.ERROR and "holds no per-subgroup gap" in record.getMessage()
        for record in caplog.records
    )
    assert not log.exists()
    assert send.calls == []


@pytest.mark.parametrize(
    ("extra", "level"),
    [
        pytest.param((), Level.SUMMARY, id="default"),
        pytest.param(("--guidance", "cell"), Level.CELL, id="cell"),
    ],
)
def test_the_guidance_flag_sets_the_level_and_defaults_to_summary(tmp_path, extra, level):
    send = two_replies()
    code, _, log = run(tmp_path, make_context(send), *extra)
    assert code == 0
    assert {call.settings.guidance for call in send.calls} == {level}
    assert log_lines(log)[0]["settings"]["guidance"] == level.value


OLDER_MEN_ANGINA = archive(
    "\n".join(
        cleveland_line(age=age, sex=sex, num=num, cp="1.0" if (age, sex) == (63, 1) else "4.0")
        for (age, sex, num), size in SIZES.items()
        for _ in range(size)
    )
    + "\n"
)
"""`UCI` with typical angina in every older man, so the training split shows it but no young
woman does: a per-subgroup gap of each young-women cell, no population-level gap, and a
frequency-aware gap of the young women without the condition."""


@pytest.mark.parametrize(
    ("definition", "basis"),
    [
        (Definition.SUBGROUP, [Basis.ABSENT]),
        (Definition.POPULATION, []),
        (Definition.FREQUENCY, [Basis.RARE]),
    ],
)
def test_the_definition_chooses_the_gaps_the_prompt_states(tmp_path, definition, basis):
    _, served = ladder_releases()
    uci = pin(OLDER_MEN_ANGINA)
    send = two_replies()
    context = make_context(
        send, open_url=RoutingOpener({**served, uci.url: OLDER_MEN_ANGINA}), uci=uci
    )
    argv = [definition.value, "--log", str(tmp_path / "run.jsonl")]
    argv += ["--data-dir", str(tmp_path / "data")]
    assert main(argv, context=context, out=io.StringIO()) == 0
    first = send.calls[0]
    assert (first.sex, first.label) == (0, 0)
    assert [key for key, sentence in BASES.items() if sentence in first.user] == basis
    send.assert_consumed()


def test_an_existing_log_path_exits_2_before_any_download_or_call(tmp_path, caplog):
    log = tmp_path / "run.jsonl"
    log.write_text("paid\n", encoding="utf-8")
    send = ScriptedSend()
    code, out, _ = run(tmp_path, make_context(send))
    assert (code, out) == (2, "")
    assert send.calls == []
    assert log.read_text(encoding="utf-8") == "paid\n"
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert not (tmp_path / "data").exists(), "the log was checked only after the downloads"


def test_an_api_failure_exits_2_keeping_earlier_batches_on_stdout_and_the_spend_on_stderr(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO)
    send = ScriptedSend(reply([RECORD] * 12), GenerationError("the API could not be reached"))
    code, out, log = run(tmp_path, make_context(send))
    assert code == 2
    assert [json.loads(line)["call"] for line in out.splitlines()[1:]] == [1] * 12
    send.assert_consumed()
    digest = hashlib.sha256(out.encode()).hexdigest()
    assert f"candidates rows=12 sha256={digest}" in [r.getMessage() for r in caplog.records]
    [summary] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("run ")]
    assert "status=failed run=generate level=summary in_flight=1 calls=2 failed=1" in summary
    assert "usd=0.000350" in summary
    assert log_lines(log)[-1]["status"] == "failed"
    assert any(
        r.levelno == logging.ERROR and r.getMessage() == "the API could not be reached"
        for r in caplog.records
    )


def test_a_dataset_contract_error_exits_3(tmp_path):
    tampered = bytes([UCI[0] ^ 1]) + UCI[1:]
    _, served = ladder_releases()
    uci = pin(UCI)
    send = ScriptedSend()
    context = make_context(send, open_url=RoutingOpener({**served, uci.url: tampered}), uci=uci)
    code, out, log = run(tmp_path, context)
    assert (code, out) == (3, "")
    assert send.calls == []
    assert not log.exists()


class FlushCounter(io.StringIO):
    """A stdout that counts its flushes."""

    flushes = 0

    def flush(self) -> None:
        """Count the flush."""
        self.flushes += 1
        super().flush()


def test_each_batch_is_written_and_flushed_as_its_call_returns(tmp_path):
    out = FlushCounter()
    seen: list[tuple[int, int]] = []
    scripted = two_replies()

    def send(call: Call) -> Reply:
        seen.append((len(out.getvalue().splitlines()), out.flushes))
        return scripted(call)

    argv = ["per-subgroup", "--log", str(tmp_path / "run.jsonl")]
    argv += ["--data-dir", str(tmp_path / "data")]
    assert main(argv, context=make_context(send), out=out) == 0
    assert seen[0] == (1, 1)
    assert seen[1][0] == 13
    assert seen[1][1] >= 1


def test_stdout_is_json_lines_of_call_cell_position_and_candidate_or_undecoded_reason(tmp_path):
    _, out, _ = run(tmp_path, make_context(two_replies()))
    lines = [json.loads(line) for line in out.splitlines()[1:]]
    assert lines[0] == {"call": 1, "cell": "<55/female/0", "position": 0, "candidate": RECORD}
    assert lines[12] == {"call": 2, "cell": "<55/female/1", "position": 0, "candidate": RECORD}
    assert lines[-1] == {"call": 2, "cell": "<55/female/1", "position": 19, "undecoded": "short"}
    assert all(line == json.dumps(json.loads(line), sort_keys=True) for line in out.splitlines())
    assert json.loads(out.splitlines()[0])["kind"] == "header"


def test_the_summary_line_names_calls_tokens_usd_log_and_stdout_sha256(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _, out, log = run(tmp_path, make_context(two_replies()))
    messages = [record.getMessage() for record in caplog.records]
    assert (
        "run status=complete run=generate level=summary in_flight=1 calls=2 failed=0 retried=0 "
        "asked=32 decoded=31 undecoded=1 "
        f"input_tokens=2000 output_tokens=1000 usd=0.000700 live_usd=0.000700 replay=none "
        f"resume=none rehearsal=false log={log}"
    ) in messages
    digest = hashlib.sha256(out.encode()).hexdigest()
    assert messages[-1] == f"candidates rows=32 sha256={digest}"


def test_configure_logging_sends_info_to_stderr_and_quiets_the_http_client(monkeypatch):
    """`captureWarnings` is stubbed: the real call is process-wide and outlives the test."""
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: recorded.append(kwargs))
    monkeypatch.setattr(logging, "captureWarnings", lambda _: None)
    quiet = logging.getLogger("httpx2")
    monkeypatch.setattr(quiet, "level", quiet.level)
    configure_logging()
    [kwargs] = recorded
    assert kwargs["stream"] is sys.stderr
    assert kwargs["level"] == logging.INFO
    assert quiet.level == logging.WARNING


def test_the_module_runs_as_a_program():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
    result = subprocess.run(
        [sys.executable, "-m", "bko.generate", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--guidance" in result.stdout
    assert "--request" in result.stdout
    assert "--log" in result.stdout


def test_the_header_is_the_first_line_and_names_the_run_and_its_plan(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _, out, _ = run(tmp_path, make_context(two_replies()))
    header = json.loads(out.splitlines()[0])
    train = split(load_uci(tmp_path / "data", source=pin(UCI)), edges=UCI_FRAME.edges)
    assert header["request"] == "e1"
    assert header["definition"] == "per-subgroup"
    assert header["guidance"] == "summary"
    assert header["uci"] == pin(UCI).sha256
    assert header["fingerprint"] == train.fingerprint
    assert f"fingerprint={train.fingerprint}" in caplog.text
    pins, _ = ladder_releases()
    assert header["releases"] == {prefix: pins[prefix].sha256 for prefix in ("HP", "MONDO")}
    plan = chunks(REQUEST, SETTINGS.per_call)
    assert header["calls"] == [[cell_key(cell), rows] for cell, rows in plan]


def test_a_run_that_fails_at_its_first_call_leaves_only_the_header(tmp_path):
    send = ScriptedSend(GenerationError("the API could not be reached"))
    code, out, _ = run(tmp_path, make_context(send))
    assert code == 2
    assert [json.loads(line)["kind"] for line in out.splitlines()] == ["header"]


def test_the_header_names_the_completeness_request_and_the_chosen_level(tmp_path):
    send = ScriptedSend(reply([young(0, cp=4)] * 20), reply([young(1, cp=1)] * 20))
    argv = ("--request", "completeness", "--guidance", "cell")
    code, out, _ = run(tmp_path, completeness_context(send), *argv)
    assert code == 0
    header = json.loads(out.splitlines()[0])
    assert (header["request"], header["guidance"]) == ("completeness", "cell")
    assert header["calls"] == [["<55/female/0", 20], ["<55/female/1", 20]]


@pytest.mark.parametrize(
    ("argv", "parsed"),
    [
        (
            ["per-subgroup", "--guidance", "ontology", "--log", "runs/x/calls.jsonl"],
            {"command": "per-subgroup", "guidance": "ontology", "request": "e1"},
        ),
        (
            ["per-subgroup", "--guidance", "ontology", "--request", "completeness", "--log", "a"],
            {"command": "per-subgroup", "guidance": "ontology", "request": "completeness"},
        ),
        (
            ["frequency-aware", "--log", "a", "--data-dir", "d"],
            {"command": "frequency-aware", "guidance": "summary", "data_dir": Path("d")},
        ),
        (
            ["population-level", "--request", "e1", "--guidance", "cell", "--log", "a"],
            {"command": "population-level", "guidance": "cell", "request": "e1"},
        ),
    ],
    ids=["documented", "completeness", "data-dir", "options-reordered"],
)
def test_todays_command_lines_parse_as_they_always_did(argv, parsed):
    from bko.generate.__main__ import _parser

    args = vars(_parser(SETTINGS).parse_args(argv))
    assert args["log"] == Path(argv[argv.index("--log") + 1])
    assert {key: args[key] for key in parsed} == parsed


def direct(
    tmp_path: Path, *extra: str, seed: str = "0", log: str = "draws.jsonl"
) -> tuple[int, str, Path]:
    send = ScriptedSend()
    out = io.StringIO()
    argv = ["direct", "--request", "completeness", "--seed", seed, "--log", str(tmp_path / log)]
    argv += ["--data-dir", str(tmp_path / "data"), *extra]
    code = main(argv, context=completeness_context(send), out=out)
    assert send.calls == []
    return code, out.getvalue(), tmp_path / log


def test_direct_writes_a_schema_2_file_naming_the_sampler_and_its_seed(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    code, out, log = direct(tmp_path)
    assert code == 0
    header, *lines = [json.loads(line) for line in out.splitlines()]
    assert header["schema"] == 2
    assert (header["generator"], header["seed"]) == ("direct", 0)
    assert (header["definition"], header["guidance"]) == (None, None)
    assert header["calls"] == [["<55/female/0", 20], ["<55/female/1", 20]]
    assert [(line["call"], line["cell"]) for line in lines] == [(1, "<55/female/0")] * 20 + [
        (2, "<55/female/1")
    ] * 20
    assert all(line["candidate"]["sex"] == 0 for line in lines)
    draws = log_lines(log)
    assert [(d["kind"], d["call"], d["position"]) for d in draws[:2]] == [
        ("draw", 1, 0),
        ("draw", 1, 1),
    ]
    assert len(draws) == 40
    digest = hashlib.sha256(out.encode()).hexdigest()
    assert caplog.messages[-1] == f"candidates rows=40 sha256={digest}"


def test_direct_writes_the_same_bytes_on_a_second_run_with_the_same_seed(tmp_path):
    first = direct(tmp_path, log="a.jsonl")
    second = direct(tmp_path, log="b.jsonl")
    assert first[1] == second[1]
    assert first[2].read_bytes() == second[2].read_bytes()


def test_direct_refuses_an_existing_draw_log_before_any_download(tmp_path):
    (tmp_path / "draws.jsonl").write_text("kept\n", encoding="utf-8")
    code, out, log = direct(tmp_path)
    assert (code, out) == (2, "")
    assert log.read_text(encoding="utf-8") == "kept\n"
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["direct", "--log", "a"],
        ["direct", "--seed", "-1", "--log", "a"],
        ["direct", "--seed", "0", "--guidance", "cell", "--log", "a"],
        ["per-subgroup", "--seed", "0", "--log", "a"],
        ["direct", "--seed", "0"],
    ],
    ids=["no-seed", "negative-seed", "direct-guidance", "messages-seed", "no-log"],
)
def test_a_malformed_direct_command_exits_2(tmp_path, argv):
    with pytest.raises(SystemExit) as raised:
        main(argv, context=completeness_context(ScriptedSend()), out=io.StringIO())
    assert raised.value.code == 2


def test_direct_creates_its_draw_log_readable_by_its_owner_only(tmp_path):
    code, _, log = direct(tmp_path)
    assert code == 0
    assert log.stat().st_mode & 0o777 == 0o600


def test_direct_exits_2_writing_nothing_when_no_training_record_can_supply_a_draw(tmp_path, caplog):
    no_young_sick = uci({}, "4.0", {**SIZES, (40, 1, 1): 0})
    _, served = ladder_releases()
    source = pin(no_young_sick)
    send = ScriptedSend()
    context = make_context(
        send, open_url=RoutingOpener({**served, source.url: no_young_sick}), uci=source
    )
    log = tmp_path / "draws.jsonl"
    argv = ["direct", "--request", "completeness", "--seed", "0", "--log", str(log)]
    argv += ["--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=context, out=out) == 2
    assert out.getvalue() == ""
    assert not log.exists()
    assert send.calls == []
    assert any(
        r.levelno == logging.ERROR and "<55/female/1 cannot draw age" in r.getMessage()
        for r in caplog.records
    )


def test_a_split_without_a_cells_ages_exits_3_with_one_line(tmp_path, caplog):
    young_only = uci({}, "4.0", sizes={(40, 0, 0): 25, (40, 0, 1): 25, (40, 1, 0): 25})
    _, served = ladder_releases()
    pinned = pin(young_only)
    send = ScriptedSend()
    context = make_context(
        send, open_url=RoutingOpener({**served, pinned.url: young_only}), uci=pinned
    )
    code, out, log = run(tmp_path, context)
    assert (code, out) == (3, "")
    assert send.calls == []
    assert not log.exists()
    assert [record.getMessage() for record in caplog.records if record.levelno >= 40] == [
        "age band '>=55' does not overlap the split's ages 40 to 40"
    ]
