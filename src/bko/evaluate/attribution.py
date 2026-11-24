"""`python -m bko.evaluate attribution`: what each arm's record fields can be attributed to.

`attribution` reads `trace.jsonl` files (`bko.evaluate.trace`) and counts, per fill, each record
field's class: a knowledge fact, a training donor id, a prompt section only, or the method and seed
only (ADR `research/record-explainability`, decision 7). `arm_kind` reads an arm's kind from one
trace line and `classify` gives each of the line's fields its class. A baseline's trace holds no
record, so its fill counts once, in the unit `fill`. It reads no data and makes no API call.
"""

from __future__ import annotations

import argparse
import csv
import enum
import hashlib
import io
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.data import Dataset, field_rows
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARMS, PRAM_ARM
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.provenance import log_versions
from bko.evaluate.run_folder import base
from bko.generate.draw_spec import DEFAULT_SPEC
from bko.ontology.frames import DIABETES_FRAME, SYNTHEA_FRAME


if TYPE_CHECKING:
    from bko.evaluate.context import Context


logger = logging.getLogger("bko.evaluate")


class ArmKind(enum.StrEnum):
    """How an arm determines its records, read from one trace line."""

    DIRECT = "direct"
    PERTURBATION = "perturbation"
    MESSAGES = "messages"
    BASELINE = "baseline"


class Attribution(enum.StrEnum):
    """What a record field can be attributed to, in precedence order."""

    KNOWLEDGE = "knowledge"
    DONOR = "donor"
    PROMPT = "prompt"
    SEED = "method-seed"


ATTRIBUTION_COLUMNS: Final = ("holdout", "arm", "kind", "class", "unit", "count")


def arm_kind(line: Mapping[str, object]) -> ArmKind:
    """The kind of the arm whose trace `line` is.

    A null `record` is a baseline's; a `perturb` key a perturbation arm's; a `request` holding
    `quota` a direct arm's; any other record a Messages arm's.

    Raises:
        EvaluationError: If `line` lacks `record`, or its `record`, `perturb` or `request` is not
            the shape `trace` writes.
    """
    if "record" not in line:
        msg = "a trace line holds no record key"
        raise EvaluationError(msg)
    record, request = line["record"], line.get("request")
    if record is None:
        return ArmKind.BASELINE
    if not isinstance(record, Mapping):
        msg = f"a trace line's record is not a JSON object: {record!r}"
        raise EvaluationError(msg)
    if request is not None and not isinstance(request, Mapping):
        msg = f"a trace line's request is not a JSON object: {request!r}"
        raise EvaluationError(msg)
    if "perturb" in line:
        perturb = line["perturb"]
        if not (isinstance(perturb, Mapping) and isinstance(perturb.get("fields"), list)):
            msg = f"a trace line's perturb is not an object with fields: {perturb!r}"
            raise EvaluationError(msg)
        return ArmKind.PERTURBATION
    if request is not None and "quota" in request:
        return ArmKind.DIRECT
    return ArmKind.MESSAGES


def classify(line: Mapping[str, object], *, mapped: frozenset[str]) -> dict[str, Attribution]:
    """Each field of `line`'s record with its class; empty for a baseline's line.

    `mapped` holds the fields a mapping row's `when` clause reads, the label aside. A direct
    arm's field in it is a knowledge fact when the arm's `DrawSpec` reads the mapping table;
    else a field with a source id is a donor's, and any other the method and seed's. A
    perturbation arm's changed field is the prompt's under the rewrite arm and the method and
    seed's under PRAM, its other fields read as a direct arm's over `bko-direct`'s spec. A
    Messages record's fields are the prompt's, or the method and seed's when it is `untraced`.

    Raises:
        EvaluationError: As `arm_kind`, or if a direct line's arm is not a direct arm.
    """
    kind = arm_kind(line)
    record = line["record"]
    if kind is ArmKind.BASELINE or not isinstance(record, Mapping):
        return {}
    if kind is ArmKind.MESSAGES:
        held = Attribution.PROMPT if line.get("untraced") is None else Attribution.SEED
        return dict.fromkeys(map(str, record), held)
    name = base(str(line.get("arm")))
    spec = DIRECT_ARMS.get(name, DEFAULT_SPEC if kind is ArmKind.PERTURBATION else None)
    if spec is None:
        msg = f"the arm {name!r} draws its records but is not a direct arm"
        raise EvaluationError(msg)
    sources = line.get("sources")
    donated = set(sources) if isinstance(sources, Mapping) else set()
    perturb = line.get("perturb")
    changed = set(perturb["fields"]) if isinstance(perturb, Mapping) else set()
    rewritten = Attribution.SEED if name == PRAM_ARM else Attribution.PROMPT
    found = {}
    for field in map(str, record):
        if field in changed:
            found[field] = rewritten
        elif spec.mapped and field in mapped:
            found[field] = Attribution.KNOWLEDGE
        elif field in donated:
            found[field] = Attribution.DONOR
        else:
            found[field] = Attribution.SEED
    return found


def attribution(traces: Sequence[Path], *, mapped: frozenset[str]) -> str:
    """Each fill's count per class over the trace files `traces`, as CSV.

    A record arm's fill writes one row per class, zeros included, in the unit `field`; a
    baseline's one `method-seed` row in the unit `fill`. Fills are in the order the files first
    name them. Logs `attribution trace=` per file with its sha256, its lines and its untraced
    records, and closes with `attribution rows=` and the CSV's sha256.

    Raises:
        EvaluationError: If a line is not a JSON object, names no integer holdout or no arm, or
            its fill's lines are of two kinds; or as `classify`.
        OSError: If a file cannot be read.
    """
    kinds: dict[tuple[int, str], ArmKind] = {}
    counts: dict[tuple[int, str], dict[Attribution, int]] = {}
    for path in traces:
        data = path.read_bytes()
        lines = untraced = 0
        for text in data.decode("utf-8").splitlines():
            if not text.strip():
                continue
            line = _object(text, path)
            key = _fill(line, path)
            kind = arm_kind(line)
            if kinds.setdefault(key, kind) is not kind:
                msg = f"{path}: holdout {key[0]}'s {key[1]} fill has {kinds[key]} and {kind} lines"
                raise EvaluationError(msg)
            tally = counts.setdefault(key, dict.fromkeys(Attribution, 0))
            if kind is ArmKind.BASELINE:
                tally[Attribution.SEED] += 1
            for found in classify(line, mapped=mapped).values():
                tally[found] += 1
            lines += 1
            untraced += kind is ArmKind.MESSAGES and line.get("untraced") is not None
        logger.info(
            "attribution trace=%s sha256=%s lines=%d untraced=%d",
            path,
            hashlib.sha256(data).hexdigest(),
            lines,
            untraced,
        )
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(ATTRIBUTION_COLUMNS)
    rows = 0
    for (holdout, arm), tally in counts.items():
        kind = kinds[holdout, arm]
        if kind is ArmKind.BASELINE:
            writer.writerow((holdout, arm, kind, Attribution.SEED, "fill", tally[Attribution.SEED]))
            rows += 1
            continue
        writer.writerows((holdout, arm, kind, each, "field", tally[each]) for each in Attribution)
        rows += len(Attribution)
    text = out.getvalue()
    logger.info(
        "attribution rows=%d sha256=%s files=%d",
        rows,
        hashlib.sha256(text.encode()).hexdigest(),
        len(traces),
    )
    return text


def _object(text: str, path: Path) -> Mapping[str, object]:
    """`text` parsed as a JSON object.

    Raises:
        EvaluationError: If it is not one.
    """
    try:
        found = json.loads(text)
    except json.JSONDecodeError:
        found = None
    if not isinstance(found, dict):
        msg = f"{path}: a line is not a JSON object: {text[:80]!r}"
        raise EvaluationError(msg)
    return found


def _fill(line: Mapping[str, object], path: Path) -> tuple[int, str]:
    """The holdout and arm `line` names.

    Raises:
        EvaluationError: If it names no integer holdout or no arm.
    """
    holdout, arm = line.get("holdout"), line.get("arm")
    if not isinstance(holdout, int) or isinstance(holdout, bool) or not isinstance(arm, str):
        msg = f"{path}: a line names no integer holdout and arm: {holdout!r}, {arm!r}"
        raise EvaluationError(msg)
    return holdout, arm


def mapped_fields(dataset: Dataset, context: Context) -> frozenset[str]:
    """The fields a `when` clause of `dataset`'s field rows reads, its frame's label aside.

    The rows are `bko.evaluate.data.field_rows`', so no data is loaded.
    """
    label = _frame(dataset).label
    rows = field_rows(dataset, context)
    return frozenset(c.field for row in rows for c in row.when if c.field != label)


def _frame(dataset: Dataset) -> FrameSpec:
    """The frame `bko.evaluate.data.load` gives `dataset`'s records."""
    match dataset:
        case Dataset.UCI:
            return UCI_FRAME
        case Dataset.SYNTHEA:
            return SYNTHEA_FRAME
        case Dataset.DIABETES:
            return DIABETES_FRAME
        case _:
            assert_never(dataset)


def run_attribution(args: argparse.Namespace, named: Context) -> str:
    """The `attribution` command over `named`, the context of the run `args.run`.

    Logs the `versions` line first.
    """
    log_versions()
    return attribution(args.traces, mapped=mapped_fields(named.dataset, named))


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `attribution` subcommand to `commands`."""
    reading = commands.add_parser(
        "attribution",
        help="count what each arm's record fields can be attributed to, from trace files",
    )
    reading.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    reading.add_argument(
        "--trace",
        dest="traces",
        type=Path,
        nargs="+",
        required=True,
        metavar="PATH",
        help="a trace.jsonl that `trace` wrote; one or more",
    )
