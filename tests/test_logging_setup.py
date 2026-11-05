"""Every entry point's logging setup: offset-bearing stamps, and a setup the SDK cannot pre-empt.

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
from bko.generate.local import LOCAL_MODELS
from tests._corpus import REPO_ROOT


STAMPED: Final = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}[+-]\d{4} [A-Z]+ ")
RECORDED: Final = (
    "2026-10-28 04:08:09,250 INFO bko.evaluate.run: split seed=0 train=1 test=1 fingerprint=x"
)
STAMPED_SETUPS: Final = {
    "evaluate": "from bko.evaluate.__main__ import configure_logging",
    "app": "from bko.app.context import configure_logging",
}


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


def run_python(script: str, **env: str) -> str:
    """Run `script` in a fresh interpreter at the repository root; return its stderr."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return result.stderr


def stamped_line(setup: str) -> str:
    """The line `setup`'s `configure_logging` writes for one bko INFO record."""
    script = f"import logging\n{setup}\nconfigure_logging()\nlogging.getLogger('bko').info('x')\n"
    return run_python(script).splitlines()[-1]


@pytest.mark.parametrize("setup", STAMPED_SETUPS.values(), ids=STAMPED_SETUPS.keys())
def test_stamped_lines_carry_an_offset(setup):
    line = stamped_line(setup)
    assert STAMPED.match(line), line


def test_line_parses_an_offset_stamped_line_and_a_recorded_one():
    line = stamped_line(STAMPED_SETUPS["evaluate"])
    assert LINE.fullmatch(line), line
    found = LINE.fullmatch(RECORDED)
    assert found is not None
    assert found["message"] == "split seed=0 train=1 test=1 fingerprint=x"


def test_every_configuring_module_is_found():
    assert {"bko.evaluate.__main__", "bko.app.context"} <= set(configuring_modules())


SDK_FIRST: Final = "import logging\nimport anthropic\n"
SETUPS: Final = {
    **{
        module: f"from {module} import configure_logging\nconfigure_logging()\n"
        for module in configuring_modules()
    },
    "bko.generate.local.main": (
        "import tempfile\nfrom bko.generate.local import main\n"
        f"main(['serve-command', {min(LOCAL_MODELS)!r}, '--data-dir', tempfile.mkdtemp()])\n"
    ),
}


@pytest.mark.parametrize("setup", SETUPS.values(), ids=SETUPS.keys())
def test_anthropic_log_does_not_silence_info(setup):
    script = f"{SDK_FIRST}{setup}logging.getLogger('bko').info('reached')\n"
    stderr = run_python(script, ANTHROPIC_LOG="info")
    assert re.search(r"\bINFO bko: reached$", stderr, re.MULTILINE), stderr
