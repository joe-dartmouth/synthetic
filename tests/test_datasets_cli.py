"""The `python -m bko.datasets` command: its report, its streams, its exit codes."""

from __future__ import annotations

import dataclasses
import io
import logging
import subprocess
import sys
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from bko.datasets.__main__ import Context, main
from bko.datasets.ablation import DEFAULT_ABLATION, AblationSpec
from bko.datasets.cells import SEXES, age_band_labels
from bko.datasets.errors import UnavailableError
from bko.datasets.fetch import default_data_dir
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE
from bko.datasets.synthea_run import DEFAULT_RUN, SyntheaRun
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import FakeOpener, FakeSynthea, archive, cleveland_line, pin
from tools._git import child_environment


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CELLS = [(40, 0, 0), (40, 1, 2), (60, 0, 1), (60, 1, 0)]
"""(age, sex, num) of four populated cells; five records each, so one of each five is held out."""

TEXT = (
    "\n".join(
        [
            *(cleveland_line(age=a, sex=s, num=n) for a, s, n in CELLS for _ in range(5)),
            cleveland_line(age=60, ca="?"),
        ]
    )
    + "\n"
)

EXPECTED = (
    "age_band,sex,target,count,underrepresented\n"
    "<55,female,0,4,True\n"
    "<55,female,1,0,True\n"
    "<55,male,0,0,True\n"
    "<55,male,1,4,True\n"
    ">=55,female,0,0,True\n"
    ">=55,female,1,4,True\n"
    ">=55,male,0,4,True\n"
    ">=55,male,1,0,True\n"
)


def run_cells(data_dir, data, opener=None):
    out = io.StringIO()
    context = Context(open_url=opener or FakeOpener(data), uci=pin(archive(TEXT)))
    code = main(["cells", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue()


def test_cells_writes_the_full_training_grid_as_csv(tmp_path):
    assert run_cells(tmp_path, archive(TEXT)) == (0, EXPECTED)


def test_a_second_run_fetches_nothing_and_prints_the_same_bytes(tmp_path):
    data = archive(TEXT)
    opener = FakeOpener(data)
    first = run_cells(tmp_path, data, opener)
    second = run_cells(tmp_path, data, opener)
    assert first == second == (0, EXPECTED)
    assert len(opener.calls) == 1


def test_stderr_names_the_underrepresented_cells_and_the_split(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        run_cells(tmp_path, archive(TEXT))
    assert "8 of 8 cells hold fewer than 20 training records" in caplog.text
    assert "split seed=0 train=16 test=4" in caplog.text
    assert "dropped 1 records with a missing value" in caplog.text


def test_a_checksum_mismatch_exits_three_and_writes_nothing(tmp_path, caplog):
    data = archive(TEXT)
    tampered = data[:-1] + bytes([data[-1] ^ 1])
    with caplog.at_level(logging.ERROR):
        code, out = run_cells(tmp_path, data, FakeOpener(tampered))
    assert (code, out) == (3, "")
    assert pin(data).sha256 in caplog.text


def test_an_unreachable_host_exits_two_and_writes_nothing(tmp_path):
    opener = FakeOpener(error=urllib.error.URLError("down"))
    assert run_cells(tmp_path, archive(TEXT), opener) == (2, "")


def test_a_data_dir_that_is_a_file_exits_two(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    assert run_cells(blocker, archive(TEXT))[0] == 2


def test_an_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code == 2


def test_the_default_data_root_is_data_at_the_repository_root():
    assert default_data_dir() == REPO_ROOT / "data"


def test_outside_a_checkout_there_is_no_default_data_root(tmp_path):
    with pytest.raises(UnavailableError, match="pass --data-dir"):
        default_data_dir(tmp_path / "site-packages" / "bko" / "datasets" / "__main__.py")


def test_a_shallow_anchor_has_no_default_data_root():
    with pytest.raises(UnavailableError, match="pass --data-dir"):
        default_data_dir(Path("/__main__.py"))


FULL = "\n".join(
    cleveland_line(age=age, sex=sex, num=num)
    for age in (40, 60)
    for sex in (0, 1)
    for num in (0, 1)
    for _ in range(25)
)
"""Twenty-five records in each of the 8 cells, so every cell keeps 20 training records."""


def test_with_no_underrepresented_cell_stderr_says_none(tmp_path, caplog):
    data = archive(FULL)
    context = Context(open_url=FakeOpener(data), uci=pin(data))
    with caplog.at_level(logging.INFO):
        main(["cells", "--data-dir", str(tmp_path)], context=context, out=io.StringIO())
    assert "0 of 8 cells hold fewer than 20 training records: none" in caplog.text


DRIVER = """
import sys
from bko.datasets.__main__ import Context, configure_logging, main
from tests._datasets_fixtures import FakeOpener, archive, pin
from tests.test_datasets_cli import TEXT
data = archive(TEXT)
configure_logging()
context = Context(open_url=FakeOpener(data), uci=pin(data))
sys.exit(main(["cells", "--data-dir", sys.argv[1]], context=context))
"""
"""The command as `__main__` runs it, with the opener and the pin replaced."""


def test_stdout_carries_only_the_csv_and_stderr_the_logs(tmp_path):
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", DRIVER, str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert (result.returncode, result.stdout) == (0, EXPECTED)
    assert "INFO bko.datasets: split seed=0 train=16 test=4" in result.stderr


def test_the_default_data_root_is_gitignored():
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "--no-index",
            "data/uci-heart-disease/0123456789abcdef/heart-disease.zip",
        ],
        cwd=REPO_ROOT,
        env=child_environment(),
        check=False,
        timeout=60,
    )
    assert result.returncode == 0


def test_the_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "bko.datasets", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0
    assert "cells" in result.stdout


@pytest.mark.parametrize("command", ["cells", "synthea"])
def test_each_command_has_a_phony_make_target_in_the_project_environment(command):
    """`make <command>` runs the CLI through `uv run --no-sync`, inside `.venv`, and is phony.

    The `@` keeps make's echo of the command out of stdout, which carries the CSV report.
    """
    lines = (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    recipe = lines[lines.index(f"{command}:") + 1]
    assert recipe == f"\t@uv run --no-sync python -m bko.datasets {command}"
    phony = next(line for line in lines if line.startswith(".PHONY:"))
    assert command in phony.split()


JAR = b"jar" * 100


def synthea_context(runner):
    return Context(
        open_url=FakeOpener(JAR),
        synthea_jar=pin(JAR, "synthea-jar"),
        resolve_java=lambda: (Path("/opt/jdk/bin/java"), 'openjdk version "21.0.4"'),
        runner=runner,
        run=SyntheaRun(seed=1, clinician_seed=1, population=10, snapshot_date=date(2026, 1, 1)),
        ablation=AblationSpec(codes=frozenset({ISCHEMIC_HEART_DISEASE}), age_band="60-69", sex=1),
    )


def test_synthea_prints_counts_before_and_after_and_reuses_the_population(tmp_path):
    runner = FakeSynthea()
    context = synthea_context(runner)
    outs = []
    for _ in range(2):
        out = io.StringIO()
        assert main(["synthea", "--data-dir", str(tmp_path)], context=context, out=out) == 0
        outs.append(out.getvalue())
    assert outs[0] == outs[1]
    assert outs[0].splitlines()[0] == "age_band,sex,target,before,after"
    assert "60-69,male,1,2,0" in outs[0].splitlines()
    assert len(runner.calls) == 1


def test_synthea_reports_every_cell_and_only_the_target_cell_changes(tmp_path):
    out = io.StringIO()
    main(["synthea", "--data-dir", str(tmp_path)], context=synthea_context(FakeSynthea()), out=out)
    header, *rows = out.getvalue().splitlines()
    assert header == "age_band,sex,target,before,after"
    assert len(rows) == 6 * 2 * 2
    changed = [row for row in rows if row.split(",")[3] != row.split(",")[4]]
    assert changed == ["60-69,male,1,2,0"]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"runner": FakeSynthea(status=4)}, 2),
        ({"open_url": FakeOpener(b"x" * len(JAR))}, 3),
        (
            {
                "ablation": AblationSpec(
                    codes=frozenset({ISCHEMIC_HEART_DISEASE}), age_band="40-49", sex=0
                )
            },
            2,
        ),
    ],
    ids=["synthea-fails", "jar-fails-its-pin", "spec-selects-nothing"],
)
def test_a_failed_synthea_command_exits_with_its_code_and_writes_nothing(tmp_path, change, code):
    context = dataclasses.replace(synthea_context(FakeSynthea()), **change)
    out = io.StringIO()
    assert main(["synthea", "--data-dir", str(tmp_path)], context=context, out=out) == code
    assert out.getvalue() == ""


def test_the_default_context_ablates_a_cell_of_the_default_grid():
    context = Context()
    assert context.run == DEFAULT_RUN
    assert context.ablation == DEFAULT_ABLATION
    assert DEFAULT_ABLATION.age_band in age_band_labels(DEFAULT_ABLATION.edges)
    assert DEFAULT_ABLATION.sex in SEXES
