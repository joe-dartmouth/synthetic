"""The app's two actions over the stages: analyze the default UCI split, then fill and score it.

Nothing here imports streamlit; `bko.app.page` renders what these return. Each action calls the
stages' public functions and adds no pipeline logic (ADR `research/streamlit-app-design`).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from bko.app.context import Context
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, training_set
from bko.baselines.request import Method, Request
from bko.datasets.cells import cell_counts
from bko.datasets.split import Split, split
from bko.evaluate.classifier import Model
from bko.evaluate.data import Data, Dataset, load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import BKO_ARMS, METHOD_ARMS, Experiment
from bko.evaluate.guided import GuidedRun, Knowledge, guided_arms
from bko.evaluate.holdout_candidates import Origin
from bko.evaluate.run import Drain, score_holdout
from bko.evaluate.scores import Score, render
from bko.fusion.closure import Closure, closure
from bko.gaps.definitions import Definition, StatementGap, gaps
from bko.gaps.detect import CellGap, count_gaps
from bko.gaps.observe import Cells, record_cells
from bko.gaps.scope import load_scope
from bko.generate.flight import Limits
from bko.generate.log import CallLog
from bko.generate.replay import Replay


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class Analysis:
    """The default split of UCI Heart Disease and its gaps.

    Attributes:
        parts: The default split, E1's first holdout; only `bko.evaluate` reads its test rows.
        data: The records it splits, as a fill's candidates file names them.
        knowledge: The releases and field mappings every BKO arm reads.
        cells: The training records observed whole and per cell.
        counts: `cell_counts` of the training split.
        counted: Its count-based gaps.
        found: Each definition's gaps over the training split.
    """

    parts: Split
    data: Data
    knowledge: Knowledge
    cells: Cells
    counts: pd.DataFrame
    counted: tuple[CellGap, ...]
    found: Mapping[Definition, tuple[StatementGap, ...]]


def analyze(context: Context) -> Analysis:
    """Load UCI Heart Disease and the releases, fetching what is absent, and list every gap.

    The records are loaded once (`bko.evaluate.data.load`); the default split, the observed cells
    and the knowledge's name all come from that one `Data` value.

    Raises:
        DatasetError: As `bko.evaluate.data.load` and `record_cells`.
        GapError: As `load_scope`.
        BKBError: As `gaps`.
        OSError: If the data root cannot be written.
    """
    data_dir = context.data_root()
    data = load(Dataset.UCI, data_dir, context)
    rows = data.field_rows
    parts = split(data.records, seed=context.experiment.holdout_seeds[0], edges=data.frame.edges)
    ontologies, fragments = load_scope(
        {row.concept for row in rows},
        data_dir,
        releases=context.releases,
        open_url=context.open_url,
    )
    cells = record_cells(parts.train, rows, name=data.name)
    counts = cell_counts(parts.train, edges=data.frame.edges)
    return Analysis(
        parts=parts,
        data=data,
        knowledge=Knowledge(data.name, ontologies, fragments, rows),
        cells=cells,
        counts=counts,
        counted=count_gaps(counts),
        found={definition: gaps(definition, fragments, cells) for definition in Definition},
    )


def fillers(experiment: Experiment) -> tuple[str, ...]:
    """The arms of `experiment` that add rows, in its order: row-adding methods and BKO arms."""
    return tuple(name for name in experiment.arms if name in BKO_ARMS or Method(name).adds_rows)


def is_paid(name: str) -> bool:
    """Whether filling with `name` calls the Messages API."""
    return name in BKO_ARMS


def experiment_for(name: str, context: Context) -> Experiment:
    """`context.experiment` narrowed to its comparison methods and `name`, with no replica.

    A BKO arm fills first, so a call that fails costs no comparator's fit.

    Raises:
        ValueError: If `name` adds no rows or is not an arm of `context.experiment`.
    """
    base = context.experiment
    if name not in fillers(base):
        msg = f"{name} is not an arm of the experiment that adds rows"
        raise ValueError(msg)
    methods = tuple(arm for arm in base.arms if arm in METHOD_ARMS)
    return dataclasses.replace(
        base,
        arms=methods if name in methods else (name, *methods),
        candidate=name,
        comparators=tuple(arm for arm in methods if arm != name),
        replica_arms=(),
        spend_ceiling=context.ceiling,
    )


class Recorder:
    """A `Runner` that hands on its runner's fill and keeps the last one it returned."""

    def __init__(self, runner: Runner) -> None:
        """Wrap `runner`."""
        self._runner = runner
        self._fill: tuple[Request, Baseline] | None = None

    def __call__(self, train: pd.DataFrame, request: Request) -> Baseline:
        """Fill `request` with the wrapped runner and keep what it returns."""
        baseline = self._runner(train, request)
        self._fill = (request, baseline)
        return baseline

    @property
    def fill(self) -> tuple[Request, Baseline]:
        """The request and fill the runner last returned.

        Raises:
            EvaluationError: If it has returned none.
        """
        if self._fill is None:
            msg = "the arm has returned no fill"
            raise EvaluationError(msg)
        return self._fill


@dataclass(frozen=True, slots=True, eq=False)
class Outcome:
    """One fill of E1's gap request on the default split, scored beside every comparison method.

    Attributes:
        name: The arm that filled.
        request: The request its fill answered, with the seed of the attempt that returned.
        baseline: The fill.
        scores: Every arm's scores on the split, in `score_holdout`'s order.
        closures: Each definition's gaps over the training rows, and over them with the added rows.
        folder: The fill's run folder.
        log: A BKO fill's call log in that folder; None for a method.
        usd: The spend its call log recorded.
        calls: The calls it recorded.
        replayed: Whether every call was answered from a call log, so nothing was paid.
    """

    name: str
    request: Request
    baseline: Baseline
    scores: tuple[Score, ...]
    closures: tuple[Closure, ...]
    folder: Path
    log: Path | None = None
    usd: float = 0.0
    calls: int = 0
    replayed: bool = False


def run_folder(context: Context, name: str) -> Path:
    """A new run folder for a fill by `name`, named by the clock and created owner-only."""
    root = context.runs_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    folder = root / f"app-{context.now():%Y%m%dT%H%M%S%fZ}-{name}"
    folder.mkdir(mode=0o700)
    return folder


def call_logs(context: Context, name: str) -> tuple[Path, ...]:
    """The call logs of earlier fills by `name`, newest first; none before the first."""
    root = context.runs_root()
    return (
        tuple(sorted(root.glob(f"app-*-{name}/calls.jsonl"), reverse=True)) if root.is_dir() else ()
    )


def training_csv(analysis: Analysis, outcome: Outcome) -> str:
    """The augmented training set as `python -m bko.baselines` prints it: real rows, then added."""
    return training_set(analysis.parts.train, outcome.baseline).to_csv(lineterminator="\n")


def scores_csv(outcome: Outcome) -> str:
    """The split's scores, in the format `python -m bko.evaluate score` writes."""
    return render(outcome.scores)


def fill_and_score(
    analysis: Analysis, name: str, context: Context, *, replay: Path | None = None
) -> Outcome:
    """Fill E1's gap request on the default split with `name`, then score every method beside it.

    The fill's run folder keeps its scores and training set, and a BKO arm's call log and
    candidates file. A BKO arm is built by `bko.evaluate.guided.guided_arms`; its calls go through
    `context.send()`, built only here, or are answered from the call log at `replay`.

    Raises:
        BaselineError: If the arm refuses every attempt, as `score_holdout`.
        EvaluationError: If a method has no runner, the spend reaches `context.ceiling`, or as
            `score_holdout`.
        GenerationError: If a call gets no response.
        ReplayError: If `replay` is not a call log that answers the fill's calls exactly.
        OSError: If a file cannot be read or created.
    """
    experiment = experiment_for(name, context)
    built = {name} if is_paid(name) else set()
    missing = sorted(set(experiment.arms) - {*context.arms, *built})
    if missing:
        msg = f"no runner for arms {missing}"
        raise EvaluationError(msg)
    if not is_paid(name):
        recorder = Recorder(context.arms[name])
        scores = _score(analysis, experiment, context, recorder)
        return _kept(
            analysis, _outcome(analysis, name, recorder, scores, run_folder(context, name))
        )
    folder = run_folder(context, name)
    paid: GuidedRun | None = None
    try:
        answers = Replay(replay) if replay is not None else None
        with guided_arms(
            (name,),
            analysis.knowledge,
            experiment,
            log=folder / "calls.jsonl",
            candidates=folder / "holdout-candidates.jsonl",
            origin=Origin("app", analysis.data, context.releases),
            send=context.send,
            run="app",
            limits=Limits(in_flight=1, idle=experiment.idle_limit),
            answers=answers,
            now=context.now,
        ) as paid:
            arm = paid.arms[name]
            recorder = Recorder(arm)
            scores = _score(analysis, experiment, context, recorder, drains={name: arm.drain})
    except Exception as exc:
        log = None if paid is None else paid.log
        spent = "" if log is None else f"; it spent {log.live_usd:.6f} USD over {log.calls} calls"
        exc.add_note(f"The fill's run folder is {folder}{spent}.")
        raise
    return _kept(analysis, _outcome(analysis, name, recorder, scores, folder, log=paid.log))


def _score(
    analysis: Analysis,
    experiment: Experiment,
    context: Context,
    recorder: Recorder,
    *,
    drains: Mapping[str, Drain] = MappingProxyType({}),
) -> list[Score]:
    arms = {**context.arms, experiment.candidate: recorder}
    models = {Model.LR: context.factory}
    return score_holdout(analysis.parts, experiment, arms, models=models, drains=drains)


def _outcome(
    analysis: Analysis,
    name: str,
    recorder: Recorder,
    scores: list[Score],
    folder: Path,
    *,
    log: CallLog | None = None,
) -> Outcome:
    request, baseline = recorder.fill
    known = analysis.knowledge
    after = record_cells(
        training_set(analysis.parts.train, baseline), known.field_rows, name=f"{known.name}-{name}"
    )
    return Outcome(
        name=name,
        request=request,
        baseline=baseline,
        scores=tuple(scores),
        closures=closure(known.fragments, analysis.cells, after),
        folder=folder,
        log=None if log is None else folder / "calls.jsonl",
        usd=0.0 if log is None else log.usd,
        calls=0 if log is None else log.calls,
        replayed=log is not None and log.replay is not None,
    )


def _kept(analysis: Analysis, outcome: Outcome) -> Outcome:
    """`outcome`, after writing its scores and training set into its run folder."""
    files = {"scores.csv": scores_csv(outcome), "training.csv": training_csv(analysis, outcome)}
    for role, text in files.items():
        with (outcome.folder / role).open("x", encoding="utf-8") as handle:
            handle.write(text)
    return outcome
