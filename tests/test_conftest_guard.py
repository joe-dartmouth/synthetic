"""The root conftest errors a test that leaves the root logger's level or handlers changed.

Each case plants one test beside a copy of the conftest and judges a child pytest run, since this
run's own conftest is already active.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._scratch_fixtures import scratch_checkout, scratch_pytest


pytestmark = pytest.mark.integration

_PLANTED = "tests/test_planted.py"


def _run(tmp_path: Path, body: str) -> str:
    """Run one planted test and return the child's output.

    Args:
        tmp_path: The case's scratch directory.
        body: The planted test's source.

    Returns:
        The child run's stdout and stderr.
    """
    root = scratch_checkout(tmp_path, "guard", {_PLANTED: body})
    run = scratch_pytest(root, _PLANTED)
    return run.stdout + run.stderr


def test_a_test_that_leaves_the_root_level_raised_errors(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        "import logging\n\n\ndef test_it():\n    logging.getLogger().setLevel(logging.DEBUG)\n",
    )
    assert "root logger changed: level" in output, output
    assert "1 error" in output, output


def test_caplog_set_level_alone_passes(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        "import logging\n\n\ndef test_it(caplog):\n    caplog.set_level(logging.INFO)\n",
    )
    assert "1 passed" in output, output


def test_a_test_that_restores_the_root_logger_passes(tmp_path: Path) -> None:
    body = (
        "import logging\n\n\n"
        "def test_it():\n"
        "    root = logging.getLogger()\n"
        "    level, handlers = root.level, list(root.handlers)\n"
        "    try:\n"
        "        root.setLevel(logging.DEBUG)\n"
        "        root.addHandler(logging.NullHandler())\n"
        "    finally:\n"
        "        root.setLevel(level)\n"
        "        root.handlers[:] = handlers\n"
    )
    output = _run(tmp_path, body)
    assert "1 passed" in output, output
