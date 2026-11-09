"""The stamped log line every stage writes, and the setup that installs it.

`Stamp` writes each line's local time with its UTC offset. `configure` is the one logging setup
each entry point calls, so every stage's stderr lines carry the same stamp.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable
from typing import Final, TextIO, override


FORMAT: Final = "%(asctime)s %(levelname)s %(name)s: %(message)s"
"""A stamped line: its time, level and logger name, then the message."""


class Stamp(logging.Formatter):
    """Format `%(asctime)s` as `YYYY-MM-DD HH:MM:SS,mmm` local time, then its UTC offset `+HHMM`.

    The stamp holds no space between the milliseconds and the offset, so it stays two tokens.
    """

    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """The record's local time, its milliseconds and its UTC offset; `datefmt` is ignored."""
        ct = self.converter(record.created)
        return (
            time.strftime("%Y-%m-%d %H:%M:%S", ct)
            + f",{int(record.msecs):03d}"
            + time.strftime("%z", ct)
        )


def stamped_handler(stream: TextIO) -> logging.Handler:
    """A handler that writes stamped lines to `stream`."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(Stamp(FORMAT))
    return handler


def configure(*, quiet: Iterable[str] = ()) -> None:
    """Log INFO and above to stderr as stamped lines, and route warnings into the log.

    A root handler that already writes stamped lines to stderr is kept, so a process that
    configures twice (Streamlit reruns its script) keeps one; otherwise the root's handlers are
    replaced (`force=True`). Each logger in `quiet` logs WARNING and above only.

    Args:
        quiet: Loggers that log WARNING and above only, such as an HTTP client's.
    """
    root = logging.getLogger()
    kept = any(
        isinstance(handler, logging.StreamHandler)
        and handler.stream is sys.stderr
        and isinstance(handler.formatter, Stamp)
        for handler in root.handlers
    )
    if not kept:
        logging.basicConfig(handlers=[stamped_handler(sys.stderr)], level=logging.INFO, force=True)
    logging.captureWarnings(True)
    for name in quiet:
        logging.getLogger(name).setLevel(logging.WARNING)
