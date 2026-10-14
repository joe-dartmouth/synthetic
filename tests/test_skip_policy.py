"""The root conftest's `live` skip reaches its own cases and nothing else, and its skip floor.

Without `--live` the conftest skips a paid case. It keys on the `live` marker and never on
`"live" in item.keywords`, and a test's keywords include the name of every collection node above
it: a checkout whose own directory is named `live` would skip every test in the suite, with exit 0
and a reason naming the flag. These cases copy the conftest into such a checkout and run a scratch
pytest there. A `live` marker applied to a whole module rather than to the function is planted too.

**And the skip floor.** A skip that matched every case would skip this module too, so its own
cases could not report it; the conftest's `_SkipFloor` fails such a run instead, and the last
cases here drive it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._corpus import REPO_ROOT


pytestmark = pytest.mark.integration

#: The files the root conftest needs to import and run one test, repo-relative. `pyproject.toml`
#: carries the marker registration and `addopts` the conftest's skip is written against.
_COPIED = (
    "pyproject.toml",
    "tests/__init__.py",
    "tests/conftest.py",
)

#: One scratch pytest run over a handful of one-line tests. Generous because the child pays
#: interpreter start-up and the conftest's imports, and a timeout would read as a guard failure.
_SCRATCH_RUN_TIMEOUT_S = 300

_PASSING = "def test_it():\n    assert True\n"

_PAID = "import pytest\n\n\n@pytest.mark.live\ndef test_it():\n    assert True\n"

_PAID_MODULE = (
    "import pytest\n\npytestmark = pytest.mark.live\n\n\ndef test_it():\n    assert True\n"
)

_EXPECTED_FAILURE = (
    "import pytest\n\n\n@pytest.mark.xfail(strict=True)\ndef test_it():\n    assert False\n"
)

_PAID_REASON = "paid-API tests need `--live`"

#: A marker expression selecting every case, `live` or not. It replaces `addopts`' `-m 'not live'`,
#: which is the case the conftest's skip exists for.
_SELECT_ALL = ("-m", "live or not live")

#: Every planted case, repo-relative, and the skip reason it must carry or ``None`` to run.
_PLANTED: dict[str, tuple[str, str | None]] = {
    "tests/test_outside.py": (_PASSING, None),
    "tests/test_paid.py": (_PAID, _PAID_REASON),
    "tests/test_paid_module.py": (_PAID_MODULE, _PAID_REASON),
    "tests/test_expected_failure.py": (_EXPECTED_FAILURE, None),
}


def _checkout_named(tmp_path: Path, name: str) -> Path:
    """A minimal checkout in a directory called `name`, holding every case in :data:`_PLANTED`.

    Args:
        tmp_path: The case's scratch directory.
        name: The checkout directory's name.

    Returns:
        The checkout's root.
    """
    root = tmp_path / name
    for relative in _COPIED:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, root / relative)
    for relative, (source, _) in _PLANTED.items():
        (root / relative).write_text(source, encoding="utf-8")
    return root


def _scratch_pytest(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run pytest in the copied checkout over the given options and files.

    Args:
        root: The checkout.
        *args: The test files, relative to it or absolute, and any options before them.

    Returns:
        The finished run, its output captured.
    """
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - a fixed argv
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--no-cov", "-rs", *args],
        cwd=root,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=_SCRATCH_RUN_TIMEOUT_S,
        check=False,
    )


def _outcomes(stdout: str) -> dict[str, str]:
    """Each planted case's verbose outcome, keyed by its repo-relative file.

    Args:
        stdout: A `-v` scratch run's output.

    Returns:
        The outcome word per file.
    """
    return dict(
        re.findall(r"^\S*?(tests/\S+\.py)::test_it (PASSED|SKIPPED|XFAIL)", stdout, re.MULTILINE)
    )


def test_a_live_marked_test_is_skipped_without_live_and_runs_with_it(tmp_path: Path) -> None:
    """A `live` case is skipped by name of its marker, in a checkout whose directory is `live`.

    Every planted case is selected with `-m 'live or not live'`, so the marker expression is not
    what keeps a paid case from running. Both the function-level and the module-level marker are
    skipped with the reason naming `--live`, and the unmarked cases run beside them. **The control
    is the second run**: with `--live` the same paid cases pass, so the skip is the flag's doing
    and not a collection that never reached them.
    """
    root = _checkout_named(tmp_path, "live")
    files = tuple(_PLANTED)

    default = _scratch_pytest(root, "-v", *_SELECT_ALL, *files)
    assert default.returncode == 0, default.stdout + default.stderr
    outcomes = _outcomes(default.stdout)
    for relative, (_, reason) in _PLANTED.items():
        expected = "SKIPPED" if reason else "PASSED"
        expected = "XFAIL" if relative == "tests/test_expected_failure.py" else expected
        assert outcomes.get(relative) == expected, (
            f"{relative} should be {expected}; the scratch run said:\n{default.stdout}"
        )
        if reason:
            pattern = rf"{re.escape(relative)}(:\d+)?: {re.escape(reason)}"
            assert re.search(pattern, default.stdout), (
                f"{relative} was skipped for the wrong reason; the scratch run said:\n"
                f"{default.stdout}"
            )

    opted_in = _scratch_pytest(root, "-v", "--live", *_SELECT_ALL, *files)
    assert opted_in.returncode == 0, opted_in.stdout + opted_in.stderr
    assert "SKIPPED" not in opted_in.stdout, opted_in.stdout
    assert _outcomes(opted_in.stdout).get("tests/test_paid.py") == "PASSED", opted_in.stdout


def test_a_run_in_which_every_selected_test_skipped_fails(tmp_path: Path) -> None:
    """The skip floor: a run that executed nothing exits non-zero and says why.

    This is the regression no other case can see, because a skip matching every case skips them
    too: the paid skip reduced to `if True:` would leave this module `skipped` with exit 0.
    """
    result = _scratch_pytest(_checkout_named(tmp_path, "floor"), *_SELECT_ALL, "tests/test_paid.py")

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert "every selected test skipped (1 skipped, none ran)" in result.stdout, result.stdout


def test_the_skip_floor_has_an_opt_out_and_counts_an_expected_failure_as_run(
    tmp_path: Path,
) -> None:
    """`--allow-all-skipped` stands the floor down, and an `xfail` is a case that ran.

    The second half is the boundary of "nothing ran": an expected failure's body executed, so a run
    holding only skips and expected failures has tested something.
    """
    root = _checkout_named(tmp_path, "floor")

    allowed = _scratch_pytest(root, "--allow-all-skipped", *_SELECT_ALL, "tests/test_paid.py")
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert "1 skipped" in allowed.stdout, allowed.stdout

    expected = _scratch_pytest(
        root, *_SELECT_ALL, "tests/test_paid.py", "tests/test_expected_failure.py"
    )
    assert expected.returncode == 0, expected.stdout + expected.stderr
    assert "1 skipped, 1 xfailed" in expected.stdout, expected.stdout
