"""What `python -m bko.evaluate` reaches outside the process: the `Context` tests replace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bko.baselines.methods import Runner
from bko.datasets.fetch import UrlOpener, open_https
from bko.datasets.run_folders import runs_root
from bko.datasets.sources import DIABETES_130, SYNTHEA_JAR, UCI_HEART_DISEASE, Source
from bko.datasets.synthea_run import DEFAULT_RUN as SYNTHEA_RUN
from bko.datasets.synthea_run import SyntheaRun
from bko.evaluate.classifier import MODELS, PANEL, Factory, Model
from bko.evaluate.data import Dataset
from bko.evaluate.experiment import E1, Experiment
from bko.evaluate.launch import Power, Spawn, on_battery, spawn
from bko.evaluate.named_runs import RUNS, NamedRun
from bko.evaluate.rehearse import CannedSend, Rehearse
from bko.evaluate.run import ARMS
from bko.evaluate.supervise import Segment, run_segment
from bko.generate.api import default_async_send
from bko.generate.local import local_send
from bko.generate.messages import SETTINGS, AnySend, Settings
from bko.generate.weights import LOCAL_MODELS
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        experiment: The configuration the run named `e1` is; every other name reads `runs`.
        settings: The generator settings of the run named `e1`.
        dataset: The data the run named `e1` scores; every other name reads `runs`.
        runs: Every other named run, by name.
        arms: Each arm's runner; a BKO name present here runs as given, so tests can stand in for
            the API.
        models: Builds each unfitted classifier, by model; it holds `Model.LR`.
        panel: The classifiers `panel` scores, by model, in order; `models`' come first.
        releases: The ontology releases the BKO arms' gap detection reads, by concept prefix.
        field_rows: The field mappings the BKO arms observe cells by.
        synthea: The pinned Synthea population's run, which a Synthea run's records are
            read from (`bko.evaluate.data.load`).
        synthea_jar: The jar that population was generated with.
        diabetes: The pinned Diabetes 130 archive, which a Diabetes 130 run's records are read
            from (`bko.evaluate.data.load`).
        send: Builds the send every BKO arm's call goes through, for an API model.
        local_send: Builds the send for an on-device model, given its id
            (`bko.generate.weights.LOCAL_MODELS`).
        rehearse: Builds the send a rehearsal answers every BKO call with.
        power: Whether the machine runs on battery, for `launch` and each `supervise` resume.
        start: Starts `launch`'s detached run.
        runs_dir: The run folders' root for `launch` and `privacy`; `runs/` at the repository root
            when None.
        segment: Runs one of `supervise`'s `score` segments to its exit.
        sleep: Waits `supervise`'s backoff before a resume; None waits until it ends or the
            operator stops the run.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    experiment: Experiment = E1
    settings: Settings = SETTINGS
    dataset: Dataset = Dataset.UCI
    runs: Mapping[str, NamedRun] = RUNS
    arms: Mapping[str, Runner] = ARMS
    models: Mapping[Model, Factory] = MODELS
    panel: Mapping[Model, Factory] = PANEL
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    synthea: SyntheaRun = SYNTHEA_RUN
    synthea_jar: Source = SYNTHEA_JAR
    diabetes: Source = DIABETES_130
    send: Callable[[], AnySend] = default_async_send
    local_send: Callable[[str], AnySend] = local_send
    rehearse: Rehearse = CannedSend
    power: Power = on_battery
    start: Spawn = spawn
    runs_dir: Path | None = None
    segment: Segment = run_segment
    sleep: Callable[[float], object] | None = None

    def runs_root(self) -> Path:
        """The run folders' root: `runs_dir`, or `runs/` at the repository root when None."""
        return self.runs_dir if self.runs_dir is not None else runs_root()

    def sender(self, model: str) -> Callable[[], AnySend]:
        """What builds the send for `model`'s calls.

        `local_send` for an on-device model (`bko.generate.weights.LOCAL_MODELS`), else `send`.
        """
        if model in LOCAL_MODELS:
            return lambda: self.local_send(model)
        return self.send
