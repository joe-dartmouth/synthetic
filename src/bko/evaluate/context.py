"""What `python -m bko.evaluate` reaches outside the process: the `Context` tests replace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bko.baselines.methods import Runner
from bko.datasets.fetch import UrlOpener, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.evaluate.classifier import MODELS, Factory, Model
from bko.evaluate.experiment import E1, RUNS, Experiment, NamedRun
from bko.evaluate.launch import Power, Spawn, on_battery, spawn
from bko.evaluate.rehearse import CannedSend, Rehearse
from bko.evaluate.run import ARMS
from bko.evaluate.supervise import Segment, run_segment
from bko.generate.api import default_async_send
from bko.generate.messages import SETTINGS, AnySend, Settings
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
        runs: Every other named run, by name.
        arms: Each arm's runner; a BKO name present here runs as given, so tests can stand in for
            the API.
        models: Builds each unfitted classifier, by model; it holds `Model.LR`.
        releases: The ontology releases the BKO arms' gap detection reads, by concept prefix.
        field_rows: The field mappings the BKO arms observe cells by.
        send: Builds the send every BKO arm's call goes through.
        rehearse: Builds the send a rehearsal answers every BKO call with.
        power: Whether the machine runs on battery, for `launch` and each `supervise` resume.
        start: Starts `launch`'s detached run.
        runs_dir: The run folders' root for `launch`; `runs/` at the repository root when None.
        segment: Runs one of `supervise`'s `score` segments to its exit.
        sleep: Waits `supervise`'s backoff before a resume; None waits until it ends or the
            operator stops the run.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    experiment: Experiment = E1
    settings: Settings = SETTINGS
    runs: Mapping[str, NamedRun] = RUNS
    arms: Mapping[str, Runner] = ARMS
    models: Mapping[Model, Factory] = MODELS
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    send: Callable[[], AnySend] = default_async_send
    rehearse: Rehearse = CannedSend
    power: Power = on_battery
    start: Spawn = spawn
    runs_dir: Path | None = None
    segment: Segment = run_segment
    sleep: Callable[[float], object] | None = None
