"""What the app reaches outside the process: tests replace each `Context` field.

`SPEND_CEILING` bounds one paid fill; ADR `research/streamlit-app-design` records why it is not
E1's ceiling.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from bko.baselines.methods import Runner
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.datasets.run_folders import runs_root
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.evaluate.classifier import Factory, build
from bko.evaluate.experiment import E1, Experiment
from bko.evaluate.run import ARMS
from bko.generate.api import HTTP_LOGGERS, default_send
from bko.generate.messages import Send
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


SPEND_CEILING: Final = 0.25
"""The call log's spend in USD at which one paid fill makes no further call."""


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr; a second call changes nothing."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.captureWarnings(True)
    for name in (*QUIET_LOGGERS, *HTTP_LOGGERS):
        logging.getLogger(name).setLevel(logging.WARNING)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Context:
    """What the app reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        releases: The ontology releases, by concept prefix.
        field_rows: The field mappings.
        arms: Each comparison method's runner, by name.
        factory: Builds the unfitted classifier.
        experiment: The configuration every fill narrows (`bko.app.steps.experiment_for`).
        send: Builds the send a live BKO fill's calls go through; called only inside that fill.
        ceiling: The spend in USD at which one paid fill stops.
        data_dir: The data root; `default_data_dir()` when None.
        runs_dir: The run folders' root; `bko.datasets.run_folders.runs_root()` when None.
        now: The clock that names each fill's run folder.
        setup_logging: Configures logging; every rerun calls it.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    arms: Mapping[str, Runner] = ARMS
    factory: Factory = build
    experiment: Experiment = E1
    send: Callable[[], Send] = default_send
    ceiling: float = SPEND_CEILING
    data_dir: Path | None = None
    runs_dir: Path | None = None
    now: Callable[[], datetime] = _utc_now
    setup_logging: Callable[[], None] = configure_logging

    def data_root(self) -> Path:
        """The data root.

        Raises:
            UnavailableError: As `default_data_dir`, outside a checkout with no `data_dir`.
        """
        return self.data_dir if self.data_dir is not None else default_data_dir()

    def runs_root(self) -> Path:
        """The run folders' root.

        Raises:
            UnavailableError: As `default_data_dir`, outside a checkout with no `runs_dir`.
        """
        return self.runs_dir if self.runs_dir is not None else runs_root()
