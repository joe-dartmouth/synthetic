"""The stamped entry points' logging setup: offset-bearing stamps, and a forced root handler.

Each case configures logging in a fresh interpreter, so no case leaves a root handler behind.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Final

import pytest

from bko.evaluate.run_folder import LINE
from tests._corpus import REPO_ROOT


STAMPED: Final = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}[+-]\d{4} [A-Z]+ ")
RECORDED: Final = (
    "2026-10-28 04:08:09,250 INFO bko.evaluate.run: split seed=0 train=1 test=1 fingerprint=x"
)


def configuring_modules() -> list[str]:
    """Every tracked module under `src/bko` that defines `configure_logging`, as a dotted name."""
    listed = subprocess.run(
        ["git", "ls-files", "src/bko/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted(
        path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        for path in listed
        if "\ndef configure_logging(" in (REPO_ROOT / path).read_text(encoding="utf-8")
    )


STAMPED_SETUPS: Final = {
    module: f"from {module} import configure_logging" for module in configuring_modules()
}
"""Each configuring module's import line, keyed by its dotted name."""


def basic_config_calls() -> dict[str, list[str]]:
    """Each tracked `src/bko` module's `basicConfig(` calls, as the text up to the closing paren."""
    listed = subprocess.run(
        ["git", "ls-files", "src/bko/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    found: dict[str, list[str]] = {}
    for path in listed:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        calls = [
            text[start : text.index(")\n", start)]
            for start in (m.start() for m in re.finditer(r"\bbasicConfig\(", text))
        ]
        if calls:
            found[path] = calls
    return found


def run_python(script: str, **env: str) -> str:
    """Run `script` in a fresh interpreter at the repository root; return its stderr."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def stamped_line(setup: str, **env: str) -> str:
    """The line `setup`'s `configure_logging` writes for one bko INFO record."""
    marker = "stamp-marker"
    script = (
        f"import logging\n{setup}\nconfigure_logging()\nlogging.getLogger('bko').info({marker!r})\n"
    )
    stderr = run_python(script, **env)
    lines = [line for line in stderr.splitlines() if marker in line]
    assert len(lines) == 1, stderr
    return lines[0]


@pytest.mark.parametrize("setup", STAMPED_SETUPS.values(), ids=STAMPED_SETUPS.keys())
def test_stamped_lines_carry_an_offset(setup):
    line = stamped_line(setup)
    assert STAMPED.match(line), line


def test_line_parses_an_offset_stamped_line_and_a_recorded_one():
    line = stamped_line(STAMPED_SETUPS["bko.evaluate.__main__"])
    assert LINE.fullmatch(line), line
    found = LINE.fullmatch(RECORDED)
    assert found is not None
    assert found["message"] == "split seed=0 train=1 test=1 fingerprint=x"


def stamp_at(zone: str, created: float) -> str:
    """`Stamp.formatTime` of a record created at `created` seconds, run with `TZ=zone`."""
    script = (
        "import logging\nfrom bko.core.logs import Stamp\n"
        "record = logging.LogRecord('x', logging.INFO, 'f', 1, 'm', None, None)\n"
        f"record.created = {created!r}\nrecord.msecs = 250.0\n"
        "import sys\nsys.stderr.write(Stamp().formatTime(record))\n"
    )
    return run_python(script, TZ=zone)


def test_utc_stamps_end_in_a_zero_offset():
    assert (
        stamped_line(STAMPED_SETUPS["bko.evaluate.__main__"], TZ="UTC").split()[1].endswith("+0000")
    )


JANUARY: Final = 1767225600.0
"""2026-01-01 00:00:00 UTC."""
JULY: Final = 1782864000.0
"""2026-07-01 00:00:00 UTC."""


@pytest.mark.parametrize(
    ("zone", "created", "expected"),
    [
        ("America/St_Johns", JANUARY, "2025-12-31 20:30:00,250-0330"),
        ("America/St_Johns", JULY, "2026-06-30 21:30:00,250-0230"),
        ("America/New_York", JANUARY, "2025-12-31 19:00:00,250-0500"),
        ("America/New_York", JULY, "2026-06-30 20:00:00,250-0400"),
        ("UTC", JANUARY, "2026-01-01 00:00:00,250+0000"),
    ],
)
def test_stamp_writes_the_zone_offset_in_force_at_the_instant(zone, created, expected):
    assert stamp_at(zone, created) == expected


def test_every_configuring_module_is_found():
    assert {"bko.evaluate.__main__", "bko.app.context"} <= set(configuring_modules())


def test_the_configuring_modules_are_exactly_these():
    assert configuring_modules() == [
        "bko.app.context",
        "bko.baselines.__main__",
        "bko.datasets.__main__",
        "bko.evaluate.__main__",
        "bko.fusion.__main__",
        "bko.gaps.__main__",
        "bko.generate.__main__",
        "bko.generate.local",
        "bko.ontology.__main__",
    ]


def test_every_basic_config_call_forces():
    calls = basic_config_calls()
    assert calls
    unforced = [path for path, found in calls.items() if any("force=True" not in c for c in found)]
    assert unforced == []


def test_the_sdk_installs_a_root_handler_under_anthropic_log():
    script = "import logging\nimport anthropic\nprint(len(logging.getLogger().handlers))\n"
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "ANTHROPIC_LOG": "info"},
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) >= 1, result.stderr


SDK_FIRST: Final = "import logging\nimport anthropic\n"
SETUPS: Final = {
    **{
        module: f"from {module} import configure_logging\nconfigure_logging()\n"
        for module in configuring_modules()
    },
}


@pytest.mark.parametrize("setup", SETUPS.values(), ids=SETUPS.keys())
def test_anthropic_log_does_not_silence_info(setup):
    script = f"{SDK_FIRST}{setup}logging.getLogger('bko').info('reached')\n"
    stderr = run_python(script, ANTHROPIC_LOG="info")
    assert re.search(r"\bINFO bko: reached$", stderr, re.MULTILINE), stderr
