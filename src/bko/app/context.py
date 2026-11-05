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
from bko.datasets.sources import DIABETES_130, SYNTHEA_JAR, UCI_HEART_DISEASE, Source
from bko.datasets.synthea_run import DEFAULT_RUN as SYNTHEA_RUN
from bko.datasets.synthea_run import SyntheaRun
from bko.evaluate.classifier import Factory, build
from bko.evaluate.experiment import E1, Experiment
from bko.evaluate.logs import Stamp, stamped_handler
from bko.evaluate.run import ARMS
from bko.generate.api import HTTP_LOGGERS, default_async_send
from bko.generate.messages import AnySend
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


SPEND_CEILING: Final = 0.25
"""The call log's spend in USD at which one paid fill makes no further call."""


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr, replacing any root handler but ours.

    Streamlit reruns the script on every interaction, so a root that already holds a stamped
    handler on stderr is left alone; any other root handler, such as an SDK's, is replaced.
    """
    if not any(
        isinstance(h, logging.StreamHandler)
        and h.stream is sys.stderr
        and isinstance(h.formatter, Stamp)
        for h in logging.getLogger().handlers
    ):
        logging.basicConfig(handlers=[stamped_handler(sys.stderr)], level=logging.INFO, force=True)
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
        synthea: The pinned Synthea population's run; the app scores UCI only, so it is
            never read.
        synthea_jar: The jar that population was generated with; never read.
        diabetes: The pinned Diabetes 130 archive; never read.
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
    synthea: SyntheaRun = SYNTHEA_RUN
    synthea_jar: Source = SYNTHEA_JAR
    diabetes: Source = DIABETES_130
    arms: Mapping[str, Runner] = ARMS
    factory: Factory = build
    experiment: Experiment = E1
    send: Callable[[], AnySend] = default_async_send
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
