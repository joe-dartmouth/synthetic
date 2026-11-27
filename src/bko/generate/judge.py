"""Judge an automatic mapping's candidate states with a pinned on-device model.

`judge` asks the model, one call per candidate state of `bko.ontology.automatic.candidates`, which
of the state's top `TOP` options it shows, or `NONE`, under a strict `{"concept": enum}` schema.
`verdict_payload` is the request body `bko.generate.local.local_send` posts for it: a run's
`payload` at temperature 0 with `seed` 0 and the schema named `verdict`. A reply outside the enum
is refused. The verdict table holds every state the cross-reference route maps beside the judged
ones, so `bko.ontology.automatic.rows_from_verdicts` of the committed table gives the `auto-local`
rows (ADR `research/automatic-mapping`).

A table too long for one module, `diabetes-codes`', is written as two, `--part low` and
`--part high`, split at the field `SPLIT`. `--model` takes a local model with thinking off, since a
verdict's `MAX_TOKENS` leaves no room for reasoning; it defaults to `MODEL`.

    python -m bko.generate.judge <dataset> [--model <model>] [--part {low,high}] [--data-dir <dir>]
        [--url <url>]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from bko.core.logs import configure
from bko.datasets.errors import DatasetError
from bko.datasets.frames import FrameSpec
from bko.generate.errors import GenerationError
from bko.generate.local import LOCAL_URL, check_server, local_send, loopback, payload
from bko.generate.messages import AsyncSend, Call, Level, Reply, Settings
from bko.generate.weights import LOCAL_MODELS
from bko.ontology.automatic import NONE, TOP, Candidate, state, survey
from bko.ontology.automatic_inputs import DATASETS, inputs
from bko.ontology.code_frames import DIABETES_CODES_NAME
from bko.ontology.codes import literal
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.generate.judge")

MODEL: Final = "gemma-4-26b-a4b-q8_0"
"""The pinned local model the `auto-local` tables are judged with."""

PARTS: Final = ("low", "high")
"""A split table's two modules: the states of the fields sorting before `SPLIT`, then the rest."""

SPLIT: Final = "diag_3"
"""The first field of a split table's `high` part."""

MAX_TOKENS: Final = 256
"""The output limit of one verdict call, room for one curie in JSON and no reasoning."""

JUDGES: Final = tuple(sorted(model for model, pin in LOCAL_MODELS.items() if not pin.thinking))
"""The local models `--model` takes: those served with thinking off."""

SYSTEM: Final = (
    "You map one coded state of a tabular dataset to an ontology concept. A state shows a concept "
    "when every record in that state shows it. Answer with the id of the one option the state "
    f"shows, or {NONE} when no option fits."
)
"""The system turn of every verdict call."""


@dataclass(frozen=True, slots=True)
class Judged:
    """A verdict table.

    Attributes:
        verdicts: Each `<field>=<code>` state's concept or `NONE`, in (field, code) order: the
            cross-reference states, then the judged ones.
        labels: Each verdict concept's name in its release.
        calls: How many states the model was asked about.
    """

    verdicts: Mapping[str, str]
    labels: Mapping[str, str]
    calls: int = 0


def verdict_schema(candidate: Candidate) -> dict[str, object]:
    """The reply schema: one `concept` among the top `TOP` options' curies and `NONE`."""
    allowed = [option.curie for option in candidate.options[:TOP]] + [NONE]
    return {
        "type": "object",
        "properties": {"concept": {"type": "string", "enum": allowed}},
        "required": ["concept"],
        "additionalProperties": False,
    }


def verdict_call(candidate: Candidate, frame: FrameSpec, *, model: str) -> Call:
    """The call that asks `model` about `candidate`; its `sex` and `label` are unread."""
    options = "".join(f"- {option.curie}: {option.label}\n" for option in candidate.options[:TOP])
    user = (
        f"Dataset: a study of {frame.study}; each record is one {frame.cohort}.\n"
        f"Field: {candidate.field}, {frame.descriptions[candidate.field]}.\n"
        f'State: code {candidate.code}, meaning "{candidate.meaning}".\n'
        f"Options:\n{options}- {NONE}: no option fits\n"
    )
    settings = Settings(
        model=model, effort="low", max_tokens=MAX_TOKENS, per_call=1, guidance=Level.CELL
    )
    return Call(settings, 0, 0, SYSTEM, user, frame=frame, schema=verdict_schema(candidate))


def verdict_payload(call: Call) -> dict[str, object]:
    """`payload(call)` at temperature 0 with `seed` 0, its response format named `verdict`.

    Raises:
        GenerationError: As `payload`.
    """
    body = payload(call)
    body["temperature"] = 0.0
    body["seed"] = 0
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "strict": True, "schema": call.reply_schema()},
    }
    return body


def read_verdict(reply: Reply, candidate: Candidate) -> str:
    """The concept `reply` names for `candidate`.

    Raises:
        GenerationError: Not retryable, if the reply is not a JSON object with a `concept` string
            among the offered options and `NONE`.
    """
    allowed = {option.curie for option in candidate.options[:TOP]} | {NONE}
    try:
        body = json.loads(reply.text) if reply.text is not None else None
    except ValueError:
        body = None
    concept = body.get("concept") if isinstance(body, dict) else None
    if not isinstance(concept, str):
        msg = f"{candidate.key}: the reply is not a verdict: {reply.text!r}"
        raise GenerationError(msg)
    if concept not in allowed:
        msg = f"{candidate.key}: the verdict {concept} is not among the offered {sorted(allowed)}"
        raise GenerationError(msg)
    return concept


async def judge(
    frame: FrameSpec,
    ontologies: Mapping[str, Ontology],
    send: AsyncSend,
    *,
    model: str,
    codes: Sequence[str] = (),
    part: str | None = None,
    dataset: str | None = None,
) -> Judged:
    """The verdict table of `frame`: the cross-reference states, then one call per candidate.

    Logs a `judge start` line with the calls to make, then one `verdict i=<i>/<n>` line per call
    with its seconds.

    Args:
        frame: The frame whose codebook is read, surveyed once.
        ontologies: Each release prefix's parsed release.
        send: Answers each verdict call, in candidate order, one at a time.
        model: The model each call names.
        codes: As `bko.ontology.automatic.build_lexical` takes them.
        part: One of `PARTS`: only the states of that part's fields; None for every state.
        dataset: The name the start line gives; `frame`'s name when None.

    Raises:
        GenerationError: As `send`, or `read_verdict` for a reply that is not a verdict, its
            message naming the candidate and its place, `i/n`.
        ValueError: As `bko.ontology.automatic.survey`, or if no candidate state is in `part`.
    """
    found = survey(frame, ontologies, codes)
    crossed = [row for row in found.rows if state(row)[0] in codes and in_part(state(row)[0], part)]
    asked = [c for c in found.candidates if in_part(c.field, part)]
    if not asked:
        msg = f"no candidate state of {frame.name} in part {part or 'all'}: nothing to judge"
        raise ValueError(msg)
    logger.info(
        "judge start dataset=%s part=%s calls=%d", dataset or frame.name, part or "all", len(asked)
    )
    verdicts = {"{}={}".format(*state(row)): row.concept for row in crossed}
    labels = {row.concept: row.label for row in crossed}
    for i, candidate in enumerate(asked, start=1):
        started = time.monotonic()
        try:
            reply = await send(verdict_call(candidate, frame, model=model))
            concept = read_verdict(reply, candidate)
        except GenerationError as exc:
            msg = f"{candidate.key} ({i}/{len(asked)}): {exc}"
            raise GenerationError(msg, retryable=exc.retryable) from exc
        logger.info(
            "verdict i=%d/%d field=%s code=%d concept=%s options=%d seconds=%.3f",
            i,
            len(asked),
            candidate.field,
            candidate.code,
            concept,
            min(len(candidate.options), TOP),
            time.monotonic() - started,
        )
        verdicts[candidate.key] = concept
        labels.update((o.curie, o.label) for o in candidate.options if o.curie == concept)
    return Judged(verdicts=verdicts, labels=dict(sorted(labels.items())), calls=len(asked))


def in_part(field: str, part: str | None) -> bool:
    """Whether `field`'s states belong to `part` of a split table; every field's when None.

    Raises:
        ValueError: If `part` is neither None nor one of `PARTS`.
    """
    if part is None:
        return True
    if part not in PARTS:
        msg = f"part {part!r} is not one of {list(PARTS)}"
        raise ValueError(msg)
    return (field < SPLIT) == (part == PARTS[0])


def render_verdicts(judged: Judged, *, dataset: str, model: str, part: str | None = None) -> str:
    """The `auto-local` module text holding `judged`, one entry per line between `# fmt` markers.

    A part's module names it in `PART`.
    """
    named = "" if part is None else f"PART: Final = {json.dumps(part)}\n"
    lines = [
        _HEADER,
        f"DATASET: Final = {json.dumps(dataset)}\n",
        f"MODEL: Final = {json.dumps(model)}\n{named}\n# fmt: off\n",
        "VERDICTS: Final[Mapping[str, str]] = {\n",
    ]
    lines.extend(
        f"    {literal(key)}: {literal(judged.verdicts[key])},\n"
        for key in sorted(judged.verdicts, key=_state_order)
    )
    lines.append("}\n\nLABELS: Final[Mapping[str, str]] = {\n")
    lines.extend(f"    {literal(c)}: {literal(judged.labels[c])},\n" for c in sorted(judged.labels))
    lines.append("}\n# fmt: on\n")
    return "".join(lines)


def main(argv: Sequence[str] | None = None, *, out: BinaryIO | None = None) -> int:
    """Write `<dataset>`'s verdict module to `out` (stdout when omitted) as UTF-8 bytes.

    The server is checked (`bko.generate.local.check_server`) before the releases are read. The
    closing `judge` line logs the calls, the seconds elapsed and the sha256 of the bytes written.

    Returns:
        0 done; a `DatasetError`'s exit code when a snapshot cannot be read; 2 a refused server, a
        failed call, a reply that is not a verdict, no candidate state, or another I/O failure.
    """
    parser = argparse.ArgumentParser(prog="python -m bko.generate.judge", description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--model", default=MODEL, choices=JUDGES)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--url", default=LOCAL_URL)
    parser.add_argument("--part", choices=PARTS, default=None)
    args = parser.parse_args(argv)
    if args.part is not None and args.dataset != DIABETES_CODES_NAME:
        parser.error(f"--part splits only {DIABETES_CODES_NAME}'s table")
    started = time.monotonic()
    try:
        check_server(args.model, loopback(args.url))
        data_dir = None if args.data_dir is None else args.data_dir.resolve()
        frame, ontologies, codes = inputs(args.dataset, data_dir)
        send = local_send(args.model, url=args.url, build=verdict_payload)
        judged = asyncio.run(
            judge(
                frame,
                ontologies,
                send,
                model=args.model,
                codes=codes,
                part=args.part,
                dataset=args.dataset,
            )
        )
    except DatasetError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except (GenerationError, OSError, ValueError) as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    data = render_verdicts(judged, dataset=args.dataset, model=args.model, part=args.part).encode()
    logger.info(
        "judge dataset=%s model=%s part=%s states=%d mapped=%d calls=%d elapsed=%.1f sha256=%s",
        args.dataset,
        args.model,
        args.part or "all",
        len(judged.verdicts),
        sum(concept != NONE for concept in judged.verdicts.values()),
        judged.calls,
        time.monotonic() - started,
        hashlib.sha256(data).hexdigest(),
    )
    sink = sys.stdout.buffer if out is None else out
    sink.write(data)
    sink.flush()
    return 0


def _state_order(key: str) -> tuple[str, int]:
    field, _, code = key.partition("=")
    return field, int(code)


_HEADER: Final = '''"""Local judge verdicts, written by `python -m bko.generate.judge`.

Regenerate this module rather than editing it. `VERDICTS` holds each candidate state's concept,
or `none`, keyed `<field>=<code>`, the states the cross-reference route maps included, and
`LABELS` each verdict concept's name. The rows are
`bko.ontology.automatic.rows_from_verdicts(frame, VERDICTS, LABELS)`, a split table's two parts
merged first (ADR `research/automatic-mapping`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final


'''


if __name__ == "__main__":
    configure()
    sys.exit(main())
