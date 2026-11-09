"""The root conftest errors a test that leaves the logging state changed, then restores it.

The state is the root logger's level and handlers, warning capture and `warnings.showwarning`,
and every other logger's level. Each case plants tests beside a copy of the conftest and judges a
child pytest run, since this run's own conftest is already active.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._scratch_fixtures import scratch_checkout, scratch_pytest
from tests.conftest import LOGGING_CHANGED


pytestmark = pytest.mark.integration

_PLANTED = "tests/test_planted.py"


def _run(tmp_path: Path, body: str, *, serial: bool = False, conftest: str = "") -> str:
    """Run the planted tests and return the child's output.

    Args:
        tmp_path: The case's scratch directory.
        body: The planted module's source.
        serial: Whether to run the module's tests in order in one process.
        conftest: A `conftest.py` planted at the checkout's root, beside the copied one.

    Returns:
        The child run's stdout and stderr.
    """
    planted = {_PLANTED: body} | ({"conftest.py": conftest} if conftest else {})
    root = scratch_checkout(tmp_path, "guard", planted)
    run = scratch_pytest(root, *(["-p", "no:xdist"] if serial else []), _PLANTED)
    return run.stdout + run.stderr


def test_a_test_that_leaves_the_root_level_raised_errors(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        "import logging\n\n\ndef test_it():\n    logging.getLogger().setLevel(logging.DEBUG)\n",
    )
    assert f"{LOGGING_CHANGED} root level WARNING -> DEBUG" in output, output
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


def test_a_test_that_adds_a_root_handler_raised_errors(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        "import logging\n\n\n"
        "def test_it():\n"
        "    logging.getLogger().addHandler(logging.NullHandler())\n",
    )
    assert f"{LOGGING_CHANGED} root handlers" in output, output
    assert "1 error" in output, output
    match = re.search(r"root handlers (\d+) -> (\d+)", output)
    assert match is not None, output
    assert int(match[2]) == int(match[1]) + 1, output


def test_the_guard_restores_the_root_logger_for_the_next_test(tmp_path: Path) -> None:
    body = (
        "import logging\n\n\n"
        "class Leak(logging.Handler):\n"
        "    pass\n\n\n"
        "def test_a_leaks():\n"
        "    logging.getLogger().setLevel(logging.DEBUG)\n"
        "    logging.getLogger().addHandler(Leak())\n\n\n"
        "def test_b_sees_a_clean_root():\n"
        "    root = logging.getLogger()\n"
        "    assert root.level != logging.DEBUG\n"
        "    assert not any(type(h).__name__ == 'Leak' for h in root.handlers)\n"
    )
    root = scratch_checkout(tmp_path, "guard", {_PLANTED: body})
    run = scratch_pytest(root, "-p", "no:xdist", _PLANTED)
    output = run.stdout + run.stderr
    assert "1 error" in output, output
    assert "2 passed" in output, output


def test_a_teardown_failure_is_not_masked_and_still_restores(tmp_path: Path) -> None:
    body = (
        "import logging\n\nimport pytest\n\n\n"
        "@pytest.fixture\n"
        "def boom():\n"
        "    yield\n"
        "    logging.getLogger().setLevel(logging.DEBUG)\n"
        "    raise RuntimeError('teardown-boom')\n\n\n"
        "def test_a(boom):\n"
        "    pass\n\n\n"
        "def test_b():\n"
        "    assert logging.getLogger().level != logging.DEBUG\n"
    )
    root = scratch_checkout(tmp_path, "guard", {_PLANTED: body})
    run = scratch_pytest(root, "-p", "no:xdist", _PLANTED)
    output = run.stdout + run.stderr
    assert "teardown-boom" in output, output
    assert "1 error" in output, output
    assert "2 passed" in output, output


def test_a_test_that_leaves_warning_capture_on_errors(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        "import logging\n\n\ndef test_it():\n    logging.captureWarnings(True)\n",
    )
    assert f"{LOGGING_CHANGED} warning capture off -> on" in output, output
    assert "1 error" in output, output


def test_a_test_that_restores_warning_capture_passes(tmp_path: Path) -> None:
    body = (
        "import logging\nimport warnings\n\n\n"
        "def test_it():\n"
        "    shown = warnings.showwarning\n"
        "    try:\n"
        "        logging.captureWarnings(True)\n"
        "    finally:\n"
        "        logging.captureWarnings(False)\n"
        "        warnings.showwarning = shown\n"
    )
    assert "1 passed" in _run(tmp_path, body)


def test_a_replaced_showwarning_errors_and_is_restored(tmp_path: Path) -> None:
    body = (
        "import warnings\n\n\n"
        "def fake(*args, **kwargs):\n"
        "    pass\n\n\n"
        "def test_a_leaks():\n"
        "    warnings.showwarning = fake\n\n\n"
        "def test_b_sees_clean_state():\n"
        "    assert warnings.showwarning is not fake\n"
    )
    output = _run(tmp_path, body, serial=True)
    assert f"{LOGGING_CHANGED} warnings.showwarning replaced" in output, output
    assert "1 error" in output, output
    assert "2 passed" in output, output


def test_after_a_capture_leak_the_next_capture_swaps_again(tmp_path: Path) -> None:
    body = (
        "import logging\nimport warnings\n\n\n"
        "def test_a_leaks():\n"
        "    logging.captureWarnings(True)\n\n\n"
        "def test_b_captures():\n"
        "    shown = warnings.showwarning\n"
        "    try:\n"
        "        logging.captureWarnings(True)\n"
        "        assert warnings.showwarning.__module__ == 'logging'\n"
        "    finally:\n"
        "        logging.captureWarnings(False)\n"
        "        warnings.showwarning = shown\n"
    )
    output = _run(tmp_path, body, serial=True)
    assert "1 error" in output, output
    assert "2 passed" in output, output


def test_capture_turned_off_is_turned_back_on_without_recursion(tmp_path: Path) -> None:
    """Capture on at setup and off after the test comes back on, saving the real `showwarning`.

    A restore that only put `warnings.showwarning` back would leave logging recording no capture,
    so the next `captureWarnings(True)` would save logging's own function and recurse on a warning.
    """
    conftest = (
        "import logging\n\n\ndef pytest_sessionstart(session):\n    logging.captureWarnings(True)\n"
    )
    body = (
        "import io\nimport logging\nimport warnings\n\n\n"
        "def test_a_turns_capture_off():\n"
        "    logging.captureWarnings(False)\n\n\n"
        "def test_b_sees_capture_on(caplog):\n"
        "    saved = logging._warnings_showwarning\n"
        "    assert saved is not None\n"
        "    assert saved.__module__ != 'logging'\n"
        "    logging.captureWarnings(True)\n"
        "    logging._showwarning('to-file', UserWarning, 'f.py', 1, file=io.StringIO())\n"
        "    logging._showwarning('to-log', UserWarning, 'f.py', 1)\n"
        "    assert 'to-log' in caplog.text\n"
    )
    output = _run(tmp_path, body, serial=True, conftest=conftest)
    assert f"{LOGGING_CHANGED} warning capture on -> off" in output, output
    assert "1 error" in output, output
    assert "2 passed" in output, output


def test_a_test_that_leaves_logger_levels_changed_errors(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        "import logging\n\n\n"
        "def test_it():\n"
        "    logging.getLogger('httpx2').setLevel(logging.WARNING)\n"
        "    logging.getLogger('sdv').setLevel(logging.WARNING)\n",
    )
    assert f"{LOGGING_CHANGED} httpx2 NOTSET -> WARNING; sdv NOTSET -> WARNING" in output, output
    assert "1 error" in output, output


def test_the_guard_restores_every_logger_level(tmp_path: Path) -> None:
    """A raised level returns to its prior level, and a logger made during the test to `NOTSET`."""
    body = (
        "import logging\n\n"
        "logging.getLogger('held').setLevel(logging.ERROR)\n"
        "logging.getLogger('plain')\n\n\n"
        "def test_a_leaks():\n"
        "    logging.getLogger('held').setLevel(logging.DEBUG)\n"
        "    logging.getLogger('plain').setLevel(logging.INFO)\n"
        "    logging.getLogger('made.here').setLevel(logging.CRITICAL)\n\n\n"
        "def test_b_sees_clean_state():\n"
        "    assert logging.getLogger('held').level == logging.ERROR\n"
        "    assert logging.getLogger('plain').level == logging.NOTSET\n"
        "    assert logging.getLogger('made.here').level == logging.NOTSET\n"
    )
    output = _run(tmp_path, body, serial=True)
    assert (
        f"{LOGGING_CHANGED} held ERROR -> DEBUG; made.here NOTSET -> CRITICAL; plain NOTSET -> INFO"
    ) in output, output
    assert "1 error" in output, output
    assert "2 passed" in output, output


def test_the_message_puts_the_short_changes_first(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        "import logging\n\n\n"
        "def test_it():\n"
        "    logging.getLogger('sdv').setLevel(logging.WARNING)\n"
        "    logging.getLogger().setLevel(logging.DEBUG)\n"
        "    logging.captureWarnings(True)\n",
    )
    assert (
        f"{LOGGING_CHANGED} warning capture off -> on; root level WARNING -> DEBUG; "
        "sdv NOTSET -> WARNING"
    ) in output, output
    assert "1 error" in output, output


def test_the_guard_restores_warning_capture(tmp_path: Path) -> None:
    body = (
        "import logging\nimport warnings\n\n\n"
        "def test_a_leaks():\n"
        "    logging.captureWarnings(True)\n\n\n"
        "def test_b_sees_clean_state():\n"
        "    assert warnings.showwarning.__module__ != 'logging'\n"
        "    assert logging._warnings_showwarning is None\n"
    )
    output = _run(tmp_path, body, serial=True)
    assert "1 error" in output, output
    assert "2 passed" in output, output
