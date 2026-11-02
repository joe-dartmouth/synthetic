"""A run's BKO arms over one call log and one candidates file.

`guided_arms` builds a `bko.evaluate.guided.GuidedArm` for each named Messages arm over one call
log, one candidates file and one `Flight`, for the evaluate command and the app alike.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import MappingProxyType

from bko.datasets.sources import permits_sending
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import MESSAGES_ARMS, Experiment
from bko.evaluate.guided import GuidedArm, Knowledge
from bko.evaluate.holdout_candidates import CandidateFile, Origin
from bko.evaluate.skeleton import prescriber
from bko.evaluate.watch import Watch
from bko.evaluate.waves import spendable
from bko.generate.errors import ReplayError
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog, utc_now
from bko.generate.messages import SETTINGS, AnySend, Settings
from bko.generate.replay import Answers, Replay


@dataclass(frozen=True, slots=True)
class GuidedRun:
    """A run's BKO arms over one call log and one candidates file.

    Attributes:
        log: The call log every arm writes.
        arms: Each BKO arm, by name.
        watch: The run's stop rules, which the harness calls after each holdout.
        candidates: The candidates file every arm writes; a direct arm in the run shares it.
    """

    log: CallLog
    arms: Mapping[str, GuidedArm]
    watch: Watch
    candidates: CandidateFile


@contextmanager
def guided_arms(  # ruff: ignore[too-many-arguments]
    names: Sequence[str],
    knowledge: Knowledge,
    experiment: Experiment,
    *,
    settings: Settings = SETTINGS,
    log: Path,
    candidates: Path,
    origin: Origin,
    send: Callable[[], AnySend],
    run: str,
    limits: Limits,
    answers: Answers | None = None,
    rehearsal: bool = False,
    now: Callable[[], datetime] = utc_now,
) -> Generator[GuidedRun]:
    """Build a `GuidedArm` for each of `names` over one new call log, for the body to run.

    The candidates file at `candidates` is created before the call log, with a header naming
    `origin` and `settings`, and every arm writes its lines there. The call log names the run `run`
    and `origin`'s frame and records `limits.in_flight`, and every arm's calls go through one
    `Flight` at `limits`. Every call goes through `answers` when given, else through `send()`;
    `rehearsal` marks the call log as a rehearsal's, whose `send()` answers with canned replies, and
    a replay of a rehearsal's log keeps the mark. Once the body returns, answers left over are
    refused. Each arm logs its validity sums on the way out, a stopped run included.

    Raises:
        EvaluationError: If the data is held under an agreement that grants no permission to send
            its records, before either file exists.
        ReplayError: If the body returns while `answers` holds calls it did not answer.
        GenerationError: As `CallLog`, if the settings model has no price, before `log` exists;
            the candidates file is then removed.
        importlib.metadata.PackageNotFoundError: As `CallLog`, before `log` exists; the
            candidates file is then removed.
        FileExistsError: If `candidates` exists, or `log` does; the candidates file made is then
            removed.
        Exception: Anything building `send()` or the arms raises, before the first call; both
            files are then removed.
    """
    agreement = origin.data.agreement
    if agreement is not None and not permits_sending(agreement):
        msg = (
            f"{origin.data.name} is held under the agreement {agreement.name!r}, which grants no "
            "permission to send its records to the Messages API"
        )
        raise EvaluationError(msg)
    answered = None if answers is None else answers.answered
    rehearsal = rehearsal or (answers is not None and answers.rehearsal)
    kept = CandidateFile(candidates, origin, settings)
    try:
        call_log = CallLog(
            log,
            settings,
            answered=answered,
            rehearsal=rehearsal,
            run=run,
            in_flight=limits.in_flight,
            frame=origin.data.frame.name,
            now=now,
        )
    except BaseException:
        kept.discard()
        raise
    try:
        sender = answers if answers is not None else send()
        watch = Watch(experiment, lambda: call_log.usd, projecting=not isinstance(answers, Replay))
        ceiling = experiment.spend_ceiling
        flight = Flight(
            sender,
            call_log,
            limits,
            spendable=partial(spendable, call_log, ceiling),
            ceiling=ceiling,
        )
        arms = {
            name: GuidedArm(
                name,
                spec.definition,
                knowledge,
                log=call_log,
                budget=experiment.budget if spec.budget is None else spec.budget,
                flight=flight,
                candidates=kept,
                prescribe=prescriber(knowledge) if spec.prescribes else None,
                shows=spec.shows,
                curation=spec.curation,
            )
            for name in names
            for spec in (MESSAGES_ARMS[name],)
        }
    except BaseException:
        call_log.discard()
        kept.discard()
        raise
    with kept, call_log, flight:
        try:
            yield GuidedRun(call_log, MappingProxyType(arms), watch, kept)
            if answers is not None and answers.remaining:
                msg = f"{answers.remaining} logged calls were not replayed"
                raise ReplayError(msg)
        finally:
            for arm in arms.values():
                arm.summary()
