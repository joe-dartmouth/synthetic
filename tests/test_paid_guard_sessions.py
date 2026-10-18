"""The paid-API guard holds for a whole session: at collection, under xdist and after a live test.

`tests/test_paid_guard.py` drives the guard inside this run. What it cannot observe is how the
conftest behaves in a session of its own: whether a `live` test's body is admitted under `--live`
while an unmarked test beside it is refused, in one process and in xdist workers; whether the sink
closes again once a live test ends; and whether a request made while a module is imported is
refused. These cases copy the conftest into a scratch checkout and run a child pytest there.

Every planted request goes to the loopback discard port with retries off, so a request the guard
admits meets a refused connection (`APIConnectionError`) on this machine, never a server.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._scratch_fixtures import scratch_checkout, scratch_pytest


pytestmark = pytest.mark.integration

_REFUSAL = "reached the Anthropic API"
"""The part of the guard's refusal these cases look for."""

_WIDE = {"COLUMNS": "1000"}
"""A terminal wide enough that pytest's short summary does not cut a refusal short."""

_CALL = (
    "import anthropic\n"
    "\n"
    "\n"
    "def _create():\n"
    "    client = anthropic.Anthropic(\n"
    '        api_key="not-a-credential", base_url="http://127.0.0.1:9", max_retries=0\n'
    "    )\n"
    "    return client.messages.create(\n"
    '        model="claude-opus-5-5", max_tokens=1, messages=[{"role": "user", "content": "hi"}]\n'
    "    )\n"
)
"""A discard-port request, shared by every planted module."""

_LIVE = "tests/test_a_live.py"
_UNMARKED = "tests/test_b_unmarked.py"
_AT_IMPORT = "tests/test_at_import.py"
_IN_SETUP = "tests/test_c_in_setup.py"

_PLANTED: dict[str, str] = {
    _LIVE: "import pytest\n\n" + _CALL + "\n\n@pytest.mark.live\ndef test_it():\n    _create()\n",
    _UNMARKED: _CALL + "\n\ndef test_it():\n    _create()\n",
    _AT_IMPORT: _CALL + "\n\n_create()\n\n\ndef test_it():\n    pass\n",
    _IN_SETUP: "import pytest\n\n"
    + _CALL
    + "\n\n@pytest.fixture\ndef created():\n    return _create()\n"
    + "\n\ndef test_it(created):\n    pass\n",
}
"""Each planted module, repo-relative: a live test, an unmarked one, a request at import, and a
request in a fixture's setup."""


def _failure_lines(stdout: str) -> dict[str, str]:
    """Each failed planted test's short-summary line, keyed by its repo-relative file.

    Args:
        stdout: A scratch run's output.

    Returns:
        The rest of each `FAILED <file>::test_it` line.
    """
    return dict(re.findall(r"^FAILED (tests/\S+\.py)::test_it(.*)$", stdout, re.MULTILINE))


@pytest.mark.parametrize("workers", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_a_live_test_is_admitted_and_an_unmarked_one_refused(
    tmp_path: Path, workers: list[str]
) -> None:
    """Under `--live` the live test reaches the connection; the unmarked one meets the guard."""
    root = scratch_checkout(tmp_path, "guard", _PLANTED)

    run = scratch_pytest(root, "-rfE", *workers, "--live", _LIVE, _UNMARKED, env=_WIDE)

    output = run.stdout + run.stderr
    assert "2 failed" in run.stdout, output
    failures = _failure_lines(run.stdout)
    assert "APIConnectionError" in failures.get(_LIVE, ""), output
    assert _REFUSAL not in failures.get(_LIVE, ""), output
    assert _REFUSAL in failures.get(_UNMARKED, ""), output


def test_the_sink_closes_again_after_a_live_test(tmp_path: Path) -> None:
    """A serial run of the live test, then a request in the next test's setup, refuses the second.

    Each test's call phase sets the sink's state afresh, so an unmarked test body would be refused
    even if the live test left the sink open. The setup phase is what runs between the two, so a
    request made there meets an open sink unless the live test's reset ran.
    """
    root = scratch_checkout(tmp_path, "guard", _PLANTED)

    run = scratch_pytest(root, "-rfE", "--live", _LIVE, _IN_SETUP, env=_WIDE)

    output = run.stdout + run.stderr
    assert "1 failed, 1 error" in run.stdout, output
    assert run.stdout.index(_LIVE) < run.stdout.index(_IN_SETUP), output
    errors = dict(re.findall(r"^ERROR (tests/\S+\.py)::test_it(.*)$", run.stdout, re.MULTILINE))
    assert _REFUSAL in errors.get(_IN_SETUP, ""), output


def test_a_request_made_at_collection_is_refused(tmp_path: Path) -> None:
    """A module that reaches the API while it is imported fails to collect, with the refusal."""
    root = scratch_checkout(tmp_path, "guard", _PLANTED)

    run = scratch_pytest(root, _AT_IMPORT)

    output = run.stdout + run.stderr
    assert run.returncode != 0, output
    assert f"ERROR collecting {_AT_IMPORT}" in run.stdout, output
    assert _REFUSAL in run.stdout, output
    assert "APIConnectionError" not in run.stdout, output


def test_without_live_a_live_test_is_skipped_not_refused(tmp_path: Path) -> None:
    """Without `--live` the collection hook skips the live test before the guard could see it."""
    root = scratch_checkout(tmp_path, "guard", _PLANTED)

    run = scratch_pytest(root, "--allow-all-skipped", _LIVE)

    output = run.stdout + run.stderr
    assert run.returncode == 0, output
    assert "1 skipped" in run.stdout, output
    assert _REFUSAL not in output, output
