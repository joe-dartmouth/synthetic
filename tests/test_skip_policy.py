"""The root conftest's opt-in skips reach their own cases and nothing else, and its skip floor.

Without `--live` the conftest skips a paid case, and that skip is the only gate: `addopts` carries
no `-m`, so `--live` alone runs paid tests and no marker expression does. It keys on the `live`
marker and never on `"live" in item.keywords`, and a test's keywords include the name of every
collection node above it: a checkout whose own directory is named `live` would skip every test in
the suite, with exit 0 and a reason naming the flag. These cases copy the conftest into such a
checkout and run a scratch pytest there. A `live` marker applied to a whole module rather than to
the function is planted too. `--releases` does the same for a case marked `releases`, in a
checkout named `releases`, and each flag leaves the other's cases skipped.

**And the skip floor.** A skip that matched every case would skip this module too, so its own
cases could not report it; the conftest's `_SkipFloor` fails such a run instead, and the last
cases here drive it, serially and under xdist, where the controller counts the reports its workers
forward.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

from tests._scratch_fixtures import scratch_checkout, scratch_pytest


pytestmark = pytest.mark.integration

_PASSING = "def test_it():\n    assert True\n"

_PAID = "import pytest\n\n\n@pytest.mark.live\ndef test_it():\n    assert True\n"

_PAID_MODULE = (
    "import pytest\n\npytestmark = pytest.mark.live\n\n\ndef test_it():\n    assert True\n"
)

_EXPECTED_FAILURE = (
    "import pytest\n\n\n@pytest.mark.xfail(strict=True)\ndef test_it():\n    assert False\n"
)

_PAID_REASON = "paid-API tests need `--live`"

_RELEASES = (
    "import pytest\n\npytestmark = pytest.mark.releases\n\n\ndef test_it():\n    assert True\n"
)

_RELEASES_REASON = "pinned-release tests need `--releases`"


class _Planted(NamedTuple):
    """One planted case and what each scratch run must report for it.

    Attributes:
        source: The test module's text.
        default: Its verbose outcome with no flag.
        opted_in: Its verbose outcome with `--live`.
        released: Its verbose outcome with `--releases`.
        reason: The skip reason it carries whenever it is skipped, or ``None`` when it never is.
    """

    source: str
    default: str
    opted_in: str
    released: str
    reason: str | None = None


#: Every planted case, repo-relative.
_PLANTED: dict[str, _Planted] = {
    "tests/test_outside.py": _Planted(_PASSING, "PASSED", "PASSED", "PASSED"),
    "tests/test_paid.py": _Planted(_PAID, "SKIPPED", "PASSED", "SKIPPED", _PAID_REASON),
    "tests/test_paid_module.py": _Planted(
        _PAID_MODULE, "SKIPPED", "PASSED", "SKIPPED", _PAID_REASON
    ),
    "tests/test_releases.py": _Planted(_RELEASES, "SKIPPED", "SKIPPED", "PASSED", _RELEASES_REASON),
    "tests/test_expected_failure.py": _Planted(_EXPECTED_FAILURE, "XFAIL", "XFAIL", "XFAIL"),
}


def _checkout_named(tmp_path: Path, name: str) -> Path:
    """A minimal checkout in a directory called `name`, holding every case in :data:`_PLANTED`.

    Args:
        tmp_path: The case's scratch directory.
        name: The checkout directory's name.

    Returns:
        The checkout's root.
    """
    return scratch_checkout(tmp_path, name, {path: p.source for path, p in _PLANTED.items()})


def _outcomes(stdout: str) -> dict[str, str]:
    """Each planted case's verbose outcome, keyed by its repo-relative file.

    Args:
        stdout: A serial `-v` scratch run's output.

    Returns:
        The outcome word per file.
    """
    return dict(
        re.findall(r"^\S*?(tests/\S+\.py)::test_it (PASSED|SKIPPED|XFAIL)", stdout, re.MULTILINE)
    )


def test_a_live_marked_test_is_skipped_without_live_and_runs_with_it(tmp_path: Path) -> None:
    """`--live` alone runs a `live` case, and without it the case is skipped by its marker.

    The checkout's directory is `live`, so a skip keyed on keywords would skip every case. Without
    the flag both the function-level and the module-level marker are skipped with the reason naming
    `--live`, and the unmarked cases run beside them. **The control is the second run**: with
    `--live` and nothing else the same paid cases pass, so the skip is the flag's doing and not a
    collection that never reached them. Each case's expected outcome in both runs is its row in
    :data:`_PLANTED`.
    """
    root = _checkout_named(tmp_path, "live")
    files = tuple(_PLANTED)

    default = scratch_pytest(root, "-v", *files)
    assert default.returncode == 0, default.stdout + default.stderr
    opted_in = scratch_pytest(root, "-v", "--live", *files)
    assert opted_in.returncode == 0, opted_in.stdout + opted_in.stderr

    for run, column in ((default, "default"), (opted_in, "opted_in")):
        outcomes = _outcomes(run.stdout)
        for relative, planted in _PLANTED.items():
            expected = getattr(planted, column)
            assert outcomes.get(relative) == expected, (
                f"{relative} should be {expected} in the {column} run; it said:\n{run.stdout}"
            )
    for run, column in ((default, "default"), (opted_in, "opted_in")):
        for relative, planted in _PLANTED.items():
            if planted.reason and getattr(planted, column) == "SKIPPED":
                pattern = rf"{re.escape(relative)}(:\d+)?: {re.escape(planted.reason)}"
                assert re.search(pattern, run.stdout), (
                    f"{relative} was skipped for the wrong reason in the {column} run; it "
                    f"said:\n{run.stdout}"
                )


def test_a_marker_expression_cannot_enable_a_live_test(tmp_path: Path) -> None:
    """`-m live` selects the paid cases, and without `--live` they are still skipped.

    So no `-m` spelling is a way round the flag. Every selected case skipping, the floor below
    fails the run too, which is the outcome an operator who tried `-m live` should see.
    """
    root = _checkout_named(tmp_path, "marker")

    result = scratch_pytest(root, "-m", "live", *_PLANTED)

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert "every selected test skipped (2 skipped, none ran)" in result.stdout, result.stdout
    assert "3 deselected" in result.stdout, "`-m live` selected the paid cases and no others"
    assert result.stdout.count(_PAID_REASON) == 2, result.stdout


def test_a_releases_marked_test_runs_with_releases_and_nothing_else_does(tmp_path: Path) -> None:
    """`--releases` alone runs a `releases` case, and leaves the paid cases skipped.

    The checkout's directory is `releases`, so a skip keyed on keywords would skip every case.
    Each case's expected outcome without a flag and with `--releases` is its row in
    :data:`_PLANTED`; `--live` leaving the `releases` case skipped is the first test's.
    """
    root = _checkout_named(tmp_path, "releases")
    files = tuple(_PLANTED)

    default = scratch_pytest(root, "-v", *files)
    assert default.returncode == 0, default.stdout + default.stderr
    released = scratch_pytest(root, "-v", "--releases", *files)
    assert released.returncode == 0, released.stdout + released.stderr

    for run, column in ((default, "default"), (released, "released")):
        outcomes = _outcomes(run.stdout)
        for relative, planted in _PLANTED.items():
            expected = getattr(planted, column)
            assert outcomes.get(relative) == expected, (
                f"{relative} should be {expected} in the {column} run; it said:\n{run.stdout}"
            )
            if planted.reason and expected == "SKIPPED":
                pattern = rf"{re.escape(relative)}(:\d+)?: {re.escape(planted.reason)}"
                assert re.search(pattern, run.stdout), (
                    f"{relative} was skipped for the wrong reason in the {column} run; it "
                    f"said:\n{run.stdout}"
                )


@pytest.mark.parametrize("flags", [(), ("--live",)], ids=["no-flag", "live"])
def test_a_marker_expression_cannot_enable_a_releases_test(
    tmp_path: Path, flags: tuple[str, ...]
) -> None:
    """`-m releases` selects the release case, and without `--releases` it is still skipped."""
    root = _checkout_named(tmp_path, "marker")

    result = scratch_pytest(root, *flags, "-m", "releases", *_PLANTED)

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert "every selected test skipped (1 skipped, none ran)" in result.stdout, result.stdout
    assert "4 deselected" in result.stdout, "`-m releases` selected the release case alone"
    assert result.stdout.count(_RELEASES_REASON) == 1, result.stdout


def test_a_run_in_which_every_selected_test_skipped_fails(tmp_path: Path) -> None:
    """The skip floor: a run that executed nothing exits non-zero and says why.

    This is the regression no other case can see, because a skip matching every case skips them
    too: the paid skip reduced to `if True:` would leave this module `skipped` with exit 0.
    """
    result = scratch_pytest(_checkout_named(tmp_path, "floor"), "tests/test_paid.py")

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert "every selected test skipped (1 skipped, none ran)" in result.stdout, result.stdout


def test_the_skip_floor_holds_under_xdist(tmp_path: Path) -> None:
    """Under `-n`, the floor counts on the controller from the reports its workers forward.

    `make test` runs under xdist, so a floor that held only serially would not hold where the suite
    runs. An all-skipped selection must fail with the floor's message, and the control is a mixed
    selection, which must pass: a floor that refused every xdist run would satisfy the first half.
    """
    root = _checkout_named(tmp_path, "floor")

    skipped = scratch_pytest(root, "-n", "2", "tests/test_paid.py", "tests/test_paid_module.py")
    assert skipped.returncode == pytest.ExitCode.TESTS_FAILED, skipped.stdout + skipped.stderr
    assert "every selected test skipped (2 skipped, none ran)" in skipped.stdout, skipped.stdout

    mixed = scratch_pytest(root, "-n", "2", "tests/test_outside.py", "tests/test_paid.py")
    assert mixed.returncode == 0, mixed.stdout + mixed.stderr
    assert "1 passed, 1 skipped" in mixed.stdout, mixed.stdout
    assert "every selected test skipped" not in mixed.stdout, mixed.stdout


def test_the_skip_floor_has_an_opt_out_and_counts_an_expected_failure_as_run(
    tmp_path: Path,
) -> None:
    """`--allow-all-skipped` stands the floor down, and an `xfail` is a case that ran.

    The second half is the boundary of "nothing ran": an expected failure's body executed, so a run
    holding only skips and expected failures has tested something.
    """
    root = _checkout_named(tmp_path, "floor")

    allowed = scratch_pytest(root, "--allow-all-skipped", "tests/test_paid.py")
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert "1 skipped" in allowed.stdout, allowed.stdout

    expected = scratch_pytest(root, "tests/test_paid.py", "tests/test_expected_failure.py")
    assert expected.returncode == 0, expected.stdout + expected.stderr
    assert "1 skipped, 1 xfailed" in expected.stdout, expected.stdout
