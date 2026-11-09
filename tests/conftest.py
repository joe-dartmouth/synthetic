"""Pytest configuration: the `--live` and `--releases` opt-ins, the skip floor, the paid guard."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from typing import Any, Final, NamedTuple, NoReturn

import pytest
from anthropic._base_client import AsyncAPIClient, SyncAPIClient

from bko.core.logs import SDV_LOGGERS
from bko.generate.api import HTTP_LOGGERS


PAID_REQUEST_REFUSAL: Final = "a request outside a `live` test's body reached the Anthropic API"
"""The start of the guard's refusal."""

DISCARD_URL: Final = "http://127.0.0.1:9"
"""Where a request goes in a run without `--live`: the loopback discard port."""


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
    """Register the skip floor on the controller, and install the paid-request guard everywhere.

    Args:
        config: The active pytest config.
    """
    # The controller only: an xdist worker forwards each report to it, so counting in a worker
    # would count one share of the run and judge it alone.
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(_SkipFloor(config), "bko-skip-floor")
    # A nested in-process session (`pytest.main` inside a test) configures again; it keeps the
    # outer session's patch rather than replacing it, so its unconfigure cannot disarm the outer.
    _PaidGuard.depth += 1
    if _PaidGuard.depth == 1:
        refuse_unmarked_api_requests(config)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Undo the guard's patch when the outermost session ends.

    Args:
        config: The active pytest config.
    """
    _PaidGuard.depth -= 1
    if _PaidGuard.depth == 0 and _PaidGuard.patch is not None:
        _PaidGuard.patch.undo()
        _PaidGuard.patch = None


class _PaidGuard:
    """Whether the request sink may be reached now, and the patch that guards it.

    Attributes:
        allowed: Whether a request may reach the sink: true only during a `live` test's body.
        patch: The guard's own patcher, undone at the end of the outermost session.
        depth: How many sessions in this process are configured and not yet unconfigured.
    """

    allowed: bool = False
    patch: pytest.MonkeyPatch | None = None
    depth: int = 0


def refuse_unmarked_api_requests(config: pytest.Config) -> None:
    """Refuse every Anthropic API request made outside the body of a test marked `live`.

    **The sink rather than the label or the constructor.** The collection hook keys on the `live`
    marker, which only covers the tests a human remembered to mark. Building a client is free; only
    a request costs. So the guard patches `SyncAPIClient.request` and `AsyncAPIClient.request`,
    which every resource method reaches, rather than a list of methods a later caller walks past.

    **Installed at configure time, through a patcher of its own.** Collection, module- and
    session-scoped fixtures all run after this hook, so none of them can reach the network first,
    and a test's own `monkeypatch.undo()` cannot restore the real sink. `pytest_runtest_call` opens
    the sink only while a `live` test's body runs, and a `live` test runs only under `--live`.

    **`pytest.fail` rather than an exception class**, because production code catches broad
    `Exception` around a call; `pytest.fail` raises a `BaseException`.

    **Subprocesses.** A patch does not cross a process boundary, so without `--live` the guard also
    points `ANTHROPIC_BASE_URL` at the loopback discard port, which every child inherits and the SDK
    reads before any profile. Under `--live` a child is unguarded.

    Args:
        config: The active pytest config, read for `--live`.
    """
    # A second install in one process must wrap the real sink, not the first guard, and must not
    # orphan the first patch.
    if _PaidGuard.patch is not None:
        _PaidGuard.patch.undo()
    real_sync = SyncAPIClient.request
    real_async = AsyncAPIClient.request

    def refuse() -> NoReturn:
        pytest.fail(
            f"{PAID_REQUEST_REFUSAL}: make it in the body of a test marked `live` if it is meant "
            "to spend money, or give the test a fake client if it is not"
        )

    # `tests/test_paid_guard.py` reads these two names, so neither takes `functools.wraps`.
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if not _PaidGuard.allowed:
            refuse()
        return real_sync(*args, **kwargs)

    async def guarded_async(*args: Any, **kwargs: Any) -> Any:
        if not _PaidGuard.allowed:
            refuse()
        return await real_async(*args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(SyncAPIClient, "request", guarded)
    patch.setattr(AsyncAPIClient, "request", guarded_async)
    if not config.getoption("--live"):
        patch.setenv("ANTHROPIC_BASE_URL", DISCARD_URL)
    _PaidGuard.patch = patch


class _LoggingState(NamedTuple):
    """The process-wide logging state a test must leave as it found it.

    Attributes:
        level: The root logger's level.
        handlers: The root logger's handlers.
        showwarning: `warnings.showwarning`, which `logging.captureWarnings(True)` swaps.
        quiet: The level of each logger a stage's `configure(quiet=...)` sets.
    """

    level: int
    handlers: list[logging.Handler]
    showwarning: Any
    quiet: dict[str, int]


QUIET_LOGGERS: Final = (*SDV_LOGGERS, *HTTP_LOGGERS)
"""Every logger a stage passes to `bko.core.logs.configure(quiet=...)`."""

_LOGGING_STATE = pytest.StashKey[_LoggingState]()


def _logging_state() -> _LoggingState:
    """Read the logging state the guard compares."""
    root = logging.getLogger()
    return _LoggingState(
        root.level,
        list(root.handlers),
        warnings.showwarning,
        {name: logging.getLogger(name).level for name in QUIET_LOGGERS},
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> Iterator[None]:
    """Note the logging state before anything of the test runs.

    Args:
        item: The test about to be set up.

    Yields:
        Control to the setup.
    """
    item.stash[_LOGGING_STATE] = _logging_state()
    return (yield)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """Error a test whose run left the logging state changed, then restore it.

    The state is the root logger's level and handlers, `warnings.showwarning` and the levels of the
    loggers a stage quiets. Fixtures such as `caplog` restore at their teardown, which runs inside
    this wrapper, so only a change that outlives the whole test is reported. The restore runs even
    when the teardown raised, and the error is raised only after a teardown that returned normally.

    Args:
        item: The test being torn down.
        nextitem: The next test, unused.

    Yields:
        Control to the teardown.
    """
    del nextitem
    try:
        result = yield
    finally:
        before = item.stash[_LOGGING_STATE]
        after = _logging_state()
        changed = []
        root = logging.getLogger()
        if (after.level, after.handlers) != (before.level, before.handlers):
            changed.append(
                f"level {before.level} -> {after.level} "
                f"handlers {len(before.handlers)} -> {len(after.handlers)}"
            )
            root.setLevel(before.level)
            root.handlers[:] = before.handlers
        if after.showwarning is not before.showwarning:
            changed.append("warnings.showwarning")
            # `captureWarnings(False)` also clears logging's own record of the swap, so a later
            # `captureWarnings(True)` swaps again.
            logging.captureWarnings(False)
            warnings.showwarning = before.showwarning
        moved = [name for name in QUIET_LOGGERS if after.quiet[name] != before.quiet[name]]
        if moved:
            changed.append(f"logger levels {', '.join(moved)}")
            for name in moved:
                logging.getLogger(name).setLevel(before.quiet[name])
    if changed:
        pytest.fail(f"root logger changed: {'; '.join(changed)}", pytrace=False)
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    """Open the request sink for the body of a `live` test, and for nothing else.

    Args:
        item: The test about to run.

    Yields:
        Control to the test body.
    """
    _PaidGuard.allowed = item.get_closest_marker("live") is not None
    try:
        return (yield)
    finally:
        _PaidGuard.allowed = False


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
