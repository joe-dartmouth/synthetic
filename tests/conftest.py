"""Pytest configuration: the `--live` opt-in and the all-skipped floor."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--live` and `--allow-all-skipped`.

    `--live` is a dedicated flag rather than a reading of `-m`, because no inspection of a marker
    expression is sound: a hand-written `-m` replaces the one in `addopts`, and an expression such
    as `not live or e2e` selects a `live` test whenever its other marker matches. A flag has one
    meaning.

    Args:
        parser: The pytest argument parser.
    """
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests that call a paid API (they are skipped otherwise)",
    )
    # The opt-out from `_SkipFloor`, for a run whose every case skipping is what the operator
    # meant -- `-k` over the paid tests without `--live`, say. A flag rather than a default,
    # because the run it refuses is the one that looks green.
    parser.addoption(
        "--allow-all-skipped",
        action="store_true",
        default=False,
        help="exit 0 even when every selected test skipped and none ran",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the skip floor on the controller.

    Args:
        config: The active pytest config.
    """
    # The controller only: an xdist worker forwards each report to it, so counting in a worker
    # would count one share of the run and judge it alone.
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(_SkipFloor(config), "bko-skip-floor")


class _SkipFloor:
    """Fail a run in which every selected test skipped and none ran.

    A skip condition that matches every item skips the test that exists to catch it, so the run
    would report `skipped` and exit 0. A run that executed nothing passes nothing, so it exits
    non-zero, naming the counts.

    Scoped to "nothing ran", not to "something skipped": a run with paid tests skipped beside tests
    that ran is unaffected. A run that collected nothing is pytest's own exit 5 and is left to it.
    An expected failure counts as having run, since its body executed.
    """

    def __init__(self, config: pytest.Config) -> None:
        """Count from zero for one session.

        Args:
            config: The session's config, read for `--allow-all-skipped`.
        """
        self._allowed = bool(config.getoption("--allow-all-skipped"))
        self._ran = 0
        self._skipped = 0
        self._refused = False

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Count one phase's report: a skip at any phase, or a call phase that executed.

        Args:
            report: The report, forwarded from a worker under xdist.
        """
        if hasattr(report, "wasxfail") or (report.when == "call" and not report.skipped):
            self._ran += 1
        elif report.skipped:
            self._skipped += 1

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Turn an all-skipped pass into a failure, unless `--allow-all-skipped` was passed.

        Args:
            session: The finishing session, whose exit status is replaced.
        """
        if session.exitstatus != pytest.ExitCode.OK or self._allowed:
            return
        if self._ran == 0 and self._skipped > 0:
            self._refused = True
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        """Say why the run failed, since no test did.

        Args:
            terminalreporter: pytest's terminal reporter.
        """
        if self._refused:
            terminalreporter.write_sep(
                "=",
                f"every selected test skipped ({self._skipped} skipped, none ran), so nothing "
                "passed; pass --allow-all-skipped if that is what this run was for",
                red=True,
            )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `live` tests unless `--live` was passed, whatever `-m` says.

    `addopts`' `-m 'not live'` is not a floor: pytest's `-m` replaces the one in `addopts` instead
    of composing with it, so any hand-written selector re-enables every `live` test. `-m live`
    still selects them, and this hook then skips them, naming the flag.

    The skip keys on the `live` marker itself, never on `item.keywords`: a test's keywords hold the
    name of every collection node above it, so a checkout in a directory named `live` would skip
    the whole suite.

    Args:
        config: The active pytest config, read for `--live`.
        items: The collected items, marked in place.
    """
    if config.getoption("--live"):
        return
    refuse = pytest.mark.skip(
        reason="paid-API tests need `--live`; no `-m` expression enables them"
    )
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(refuse)
