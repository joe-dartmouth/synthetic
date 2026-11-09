"""`bko.core.logs.configure`: one stamped stderr handler, quieted loggers, warnings in the log."""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager

from bko.core.logs import Stamp, configure, stamped_handler


@contextmanager
def restored(*names: str) -> Generator[None]:
    """Restore the root's level and handlers, the named loggers' levels and warning capture."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    named = {name: logging.getLogger(name).level for name in names}
    try:
        yield
    finally:
        logging.captureWarnings(False)
        root.setLevel(level)
        root.handlers[:] = handlers
        for name, saved in named.items():
            logging.getLogger(name).setLevel(saved)


def test_configure_replaces_the_root_handlers_with_one_stamped_stderr_handler() -> None:
    with restored():
        root = logging.getLogger()
        root.handlers[:] = [logging.StreamHandler(sys.stdout)]
        configure()
        [handler] = root.handlers
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr
        assert isinstance(handler.formatter, Stamp)
        assert root.level == logging.INFO


def test_configure_keeps_a_stamped_stderr_handler_already_on_the_root() -> None:
    with restored():
        root = logging.getLogger()
        ours = stamped_handler(sys.stderr)
        root.handlers[:] = [ours]
        configure()
        configure()
        assert root.handlers == [ours]


def test_configure_quiets_the_named_loggers() -> None:
    with restored("bko-core-a", "bko-core-b"):
        configure(quiet=("bko-core-a", "bko-core-b"))
        assert logging.getLogger("bko-core-a").level == logging.WARNING
        assert logging.getLogger("bko-core-b").level == logging.WARNING


def test_configure_routes_warnings_into_the_log() -> None:
    with restored():
        configure()
        assert logging._warnings_showwarning is not None  # ty: ignore[unresolved-attribute]
