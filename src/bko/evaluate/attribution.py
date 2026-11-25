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
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from bko.evaluate.data import Dataset, field_rows, frame
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARMS, PERTURBED_ARM, PRAM_ARM
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.provenance import log_versions
from bko.evaluate.run_folder import base
from bko.generate.draw_spec import DEFAULT_SPEC


if TYPE_CHECKING:
    from bko.datasets.frames import FrameSpec
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

PERTURBATION_ARMS: Final = frozenset({PERTURBED_ARM, PRAM_ARM})
"""The arms whose trace lines carry a `perturb` key, by name without a `-replica` suffix."""

ARM_NAME: Final = re.compile(r"[a-z0-9-]+")
"""The shape of every arm name, so a CSV cell holding one cannot start a spreadsheet formula."""


def arm_kind(line: Mapping[str, object]) -> ArmKind:
    """The kind of the arm whose trace `line` is.

    A null `record` is a baseline's; a `perturb` key a perturbation arm's; a `request` holding
    `quota` a direct arm's; any other record a Messages arm's.

    Raises:
        EvaluationError: If `line` lacks `record`, or its `record`, `request`, `sources` or
            `perturb` is not the shape `trace` writes, or a line holding `perturb` names an arm
            outside `PERTURBATION_ARMS`.
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
    sources = line.get("sources")
    if not isinstance(sources, Mapping):
        msg = f"a trace line's sources is not a JSON object: {sources!r}"
        raise EvaluationError(msg)
    if "perturb" in line:
        perturb, arm = line["perturb"], line.get("arm")
        fields = perturb.get("fields") if isinstance(perturb, Mapping) else None
        if not (isinstance(fields, list) and all(isinstance(each, str) for each in fields)):
            msg = f"a trace line's perturb is not an object with string fields: {perturb!r}"
            raise EvaluationError(msg)
        if not isinstance(arm, str) or base(arm) not in PERTURBATION_ARMS:
            msg = f"a trace line holds perturb but its arm is not a perturbation arm: {arm!r}"
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
    A changed field's class comes first, since its value is no longer the drawn fact.

    Raises:
        EvaluationError: As `arm_kind`, or if a direct line's arm is not a direct arm.
    """
    kind = arm_kind(line)
    if kind is ArmKind.BASELINE:
        return {}
    # arm_kind refused a record of any other kind that is not a JSON object.
    record = cast("Mapping[str, object]", line["record"])
    if kind is ArmKind.MESSAGES:
        held = Attribution.PROMPT if line.get("untraced") is None else Attribution.SEED
        return dict.fromkeys(map(str, record), held)
    name = base(str(line.get("arm")))
    spec = DIRECT_ARMS.get(name, DEFAULT_SPEC if kind is ArmKind.PERTURBATION else None)
    if spec is None:
        msg = f"the arm {name!r} draws its records but is not a direct arm"
        raise EvaluationError(msg)
    # arm_kind refused sources that are not a JSON object and perturb fields that are not strings.
    donated = set(cast("Mapping[str, object]", line["sources"]))
    perturb = cast("Mapping[str, list[str]] | None", line.get("perturb"))
    changed = set(perturb["fields"]) if perturb is not None else set()
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


def attribution(
    traces: Sequence[Path], *, run: str, mapped: frozenset[str], columns: frozenset[str]
) -> str:
    """Each fill's count per class over the trace files `traces`, as CSV.

    A record arm's fill writes one row per class, zeros included, in the unit `field`; a
    baseline's one `method-seed` row in the unit `fill`. Fills are in the order the files first
    name them. Every record's fields must be `columns`, the frame columns of the run `run`. Logs
    `attribution spec=` once per fill of a perturbation arm that reads `DEFAULT_SPEC`,
    `attribution trace=` per file with its sha256, its lines and its untraced records, and closes
    with `attribution rows=` and the CSV's sha256.

    Raises:
        EvaluationError: If a file is not UTF-8; a line is not a JSON object, names no integer
            holdout, no arm of lowercase letters, digits and hyphens, or a record id that is
            neither an integer nor null; two lines across the files name one holdout, arm and
            record id; a record's fields are not `columns`; a fill's lines are of two kinds; or
            as `classify`.
        OSError: If a file cannot be read.
    """
    kinds: dict[tuple[int, str], ArmKind] = {}
    counts: dict[tuple[int, str], dict[Attribution, int]] = {}
    named: set[tuple[int, str, int | None]] = set()
    for path in traces:
        data = path.read_bytes()
        lines = untraced = 0
        for number, raw in enumerate(data.split(b"\n"), start=1):
            text = _text(raw, path, number)
            if not text.strip():
                continue
            line = _object(text, path, number)
            key = _fill(line, path, number)
            record = (*key, _record_id(line, path, number))
            if record in named:
                msg = f"{path}:{number}: holdout {key[0]}'s {key[1]} record {record[2]} recurs"
                raise EvaluationError(msg)
            named.add(record)
            kind = arm_kind(line)
            if kinds.setdefault(key, kind) is not kind:
                msg = f"{path}: holdout {key[0]}'s {key[1]} fill has {kinds[key]} and {kind} lines"
                raise EvaluationError(msg)
            _check_columns(line, columns, run=run, at=f"{path}:{number}")
            if (
                key not in counts
                and kind is ArmKind.PERTURBATION
                and base(key[1]) not in DIRECT_ARMS
            ):
                logger.info(
                    "attribution spec=%s holdout=%d arm=%s",
                    json.dumps(DEFAULT_SPEC.values(), separators=(",", ":")),
                    *key,
                )
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


def _text(raw: bytes, path: Path, number: int) -> str:
    """`raw`, line `number` of `path`, decoded as UTF-8.

    Raises:
        EvaluationError: If it is not UTF-8.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"{path}:{number}: a line is not UTF-8: {error.reason}"
        raise EvaluationError(msg) from error


def _object(text: str, path: Path, number: int) -> Mapping[str, object]:
    """`text`, line `number` of `path`, parsed as a JSON object.

    Raises:
        EvaluationError: If it is not one.
    """
    try:
        found = json.loads(text)
    except json.JSONDecodeError:
        found = None
    if not isinstance(found, dict):
        msg = f"{path}:{number}: a line is not a JSON object: {text[:80]!r}"
        raise EvaluationError(msg)
    return found


def _fill(line: Mapping[str, object], path: Path, number: int) -> tuple[int, str]:
    """The holdout and arm `line`, line `number` of `path`, names.

    Raises:
        EvaluationError: If it names no integer holdout, or no arm of lowercase letters, digits
            and hyphens.
    """
    holdout, arm = line.get("holdout"), line.get("arm")
    if not isinstance(holdout, int) or isinstance(holdout, bool) or not isinstance(arm, str):
        msg = f"{path}:{number}: a line names no integer holdout and arm: {holdout!r}, {arm!r}"
        raise EvaluationError(msg)
    if ARM_NAME.fullmatch(arm) is None:
        msg = f"{path}:{number}: an arm name is not lowercase letters, digits and hyphens: {arm!r}"
        raise EvaluationError(msg)
    return holdout, arm


def _record_id(line: Mapping[str, object], path: Path, number: int) -> int | None:
    """The record id `line`, line `number` of `path`, names; None for a baseline's.

    Raises:
        EvaluationError: If it is neither an integer nor null.
    """
    found = line.get("record_id")
    if found is not None and (not isinstance(found, int) or isinstance(found, bool)):
        msg = f"{path}:{number}: a line's record id is not an integer or null: {found!r}"
        raise EvaluationError(msg)
    return found


def _check_columns(
    line: Mapping[str, object], columns: frozenset[str], *, run: str, at: str
) -> None:
    """Refuse `line`, at `at`, unless its record is null or its fields are run `run`'s `columns`.

    Raises:
        EvaluationError: If they are not.
    """
    record = line.get("record")
    if not isinstance(record, Mapping) or set(record) == columns:
        return
    missing = ",".join(sorted(columns - set(record))) or "none"
    extra = ",".join(sorted(map(str, set(record) - columns))) or "none"
    msg = (
        f"{at}: a record's fields are not the run {run}'s columns: missing {missing}, extra {extra}"
    )
    raise EvaluationError(msg)


def mapped_fields(dataset: Dataset, context: Context) -> frozenset[str]:
    """The fields a `when` clause of `dataset`'s field rows reads, its frame's label aside.

    The rows are `bko.evaluate.data.field_rows`', so no data is loaded.

    Raises:
        EvaluationError: As `_frame`.
    """
    label = _frame(dataset).label
    rows = field_rows(dataset, context)
    return frozenset(c.field for row in rows for c in row.when if c.field != label)


def run_attribution(args: argparse.Namespace, named: Context) -> str:
    """The `attribution` command over `named`, the context of the run `args.run`.

    Logs the `versions` line first.

    Raises:
        EvaluationError: As `_frame`, or as `attribution`.
    """
    columns = frozenset(_frame(named.dataset).columns)
    log_versions()
    return attribution(
        args.traces, run=args.run, mapped=mapped_fields(named.dataset, named), columns=columns
    )


def _frame(dataset: Dataset) -> FrameSpec:
    """`dataset`'s frame, read with no data loaded, as attribution reads it.

    Raises:
        EvaluationError: For `diabetes-codes` and the TableShift datasets, whose frames are built
            from their loaded records.
    """
    if dataset in {Dataset.DIABETES_CODES, Dataset.TABLESHIFT_ID, Dataset.TABLESHIFT_OOD}:
        msg = (
            f"attribution reads no data, and the {dataset} frame's codebook is built from its "
            "loaded records; attribution does not run on it"
        )
        raise EvaluationError(msg)
    return frame(dataset)


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
