"""Pytest configuration: the `--live` and `--releases` opt-ins and the all-skipped floor."""

from __future__ import annotations

from typing import Final, NamedTuple

import pytest


class _OptIn(NamedTuple):
    """One marker whose tests run only when its flag is passed.

    Attributes:
        marker: The marker's name.
        flag: The command-line flag that runs the marked tests.
        help_text: The flag's `--help` text.
        reason: The reason a marked test is skipped without the flag.
    """

    marker: str
    flag: str
    help_text: str
    reason: str


_OPT_INS: Final = (
    _OptIn(
        marker="live",
        flag="--live",
        help_text="run tests marked `live`, which call a paid API; without it they are skipped, "
        "whatever `-m` selects",
        reason="paid-API tests need `--live`; no `-m` expression enables them",
    ),
    _OptIn(
        marker="releases",
        flag="--releases",
        help_text="run tests marked `releases`, which read the pinned ontology releases under "
        "`data/`; without it they are skipped, whatever `-m` selects",
        reason="pinned-release tests need `--releases` (docs/DEVELOPMENT.md, Ontologies); no `-m` "
        "expression enables them",
    ),
)
"""Every opt-in marker, in the order a test carrying several reports its skip reason."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register each opt-in flag and `--allow-all-skipped`.

    An opt-in is a dedicated flag rather than a reading of `-m`, because no inspection of a marker
    expression is sound: an expression such as `not live or e2e` selects a `live` test whenever its
    other marker matches. A flag has one meaning: `--live` alone runs paid tests, `--releases`
    alone runs the release tests, and neither runs the other's.

    Args:
        parser: The pytest argument parser.
    """
    for opt in _OPT_INS:
        parser.addoption(opt.flag, action="store_true", default=False, help=opt.help_text)
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
    """Skip each opt-in marker's tests unless its flag was passed, whatever `-m` says.

    **This hook is the only gate on opt-in tests.** `addopts` carries no `-m`, so every `live` and
    `releases` test is collected and then skipped here, with a reason naming its flag. `-m` chooses
    among collected tests and cannot lift the skip: `-m live` without `--live` reports each paid
    test skipped, and `--live` alone runs them. Each flag lifts its own marker's skip and no other.

    The skip keys on the marker itself, never on `item.keywords`: a test's keywords hold the name
    of every collection node above it, so a checkout in a directory named `live` would skip the
    whole suite.

    Args:
        config: The active pytest config, read for each opt-in flag.
        items: The collected items, marked in place.
    """
    refusals = {
        opt.marker: pytest.mark.skip(reason=opt.reason)
        for opt in _OPT_INS
        if not config.getoption(opt.flag)
    }
    for item in items:
        for marker, refuse in refusals.items():
            if item.get_closest_marker(marker) is not None:
                item.add_marker(refuse)
                break
