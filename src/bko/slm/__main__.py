"""Command line for the language-model classifier: `python -m bko.slm {fetch,train,score}`.

`fetch` makes the pinned base model's files present and verified under the data root. `train`
fine-tunes a LoRA adapter on one holdout's training set for one arm of a recorded run, read from
the run's folder and its `rows` store: `none` the prepared training split, `real` the un-ablated
split, any other arm the split plus its stored fill. It writes the adapter under
`data/models/slm-adapters/<folder>/<holdout>-<arm>/` and refuses a path that exists. `score`
writes each test record's P(yes) as CSV to stdout, from the adapter of `--arm` or the base model
with `--base`, and logs the `utility` measures and the CSV's sha256. The base model runs on MPS in
BF16. It makes no API call. ADR `research/slm-comparison-design` holds the design, and
`docs/DEVELOPMENT.md` (Language-model classifier) the commands.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run (no
MPS, an absent model directory or download, an adapter path that exists or is absent); 3 a file
or a record contradicts its pin or contract.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, TextIO

import pandas as pd
import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from bko.baselines.request import Stratum
from bko.core.logs import configure
from bko.datasets.errors import DatasetError, UnavailableError
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.datasets.frames import FrameSpec
from bko.evaluate.context import Context as EvaluateContext
from bko.evaluate.data import load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import flagged
from bko.evaluate.named_runs import RUNS
from bko.evaluate.privacy_stores import Store, checked
from bko.evaluate.provenance import log_versions
from bko.evaluate.rows_read import bind, frame_of, read_rows
from bko.evaluate.run_folder import read_folder
from bko.evaluate.utility import utility
from bko.slm.serialize import examples
from bko.slm.train import ADAPTER_FILE, LORA, LoraSettings, probabilities, save_adapter, train
from bko.slm.weights import OLMO_3_7B, BaseModel, fetch, load_base


logger = logging.getLogger("bko.slm")

ADAPTERS: Final = Path("models") / "slm-adapters"
"""Where adapters live under the data root."""

NONE_ARM: Final = "none"
"""The arm whose training set is the prepared training split alone."""

REAL_ARM: Final = "real"
"""The arm whose training set is the un-ablated training split."""

DTYPE: Final = torch.bfloat16
"""The base model's dtype on MPS."""


@dataclass(frozen=True, slots=True)
class Request:
    """Which run, folder, store, holdout and arm a `train` or `score` reads."""

    run: str
    folder: Path
    rows: Path
    holdout: int
    arm: str | None


@dataclass(frozen=True, slots=True)
class Holdout:
    """One holdout's sets for one arm.

    Attributes:
        frame: The data's frame.
        train: The arm's training set.
        test: The holdout's test split.
        flagged: The cells the run flags, which `utility` scopes.
    """

    frame: FrameSpec
    train: pd.DataFrame
    test: pd.DataFrame
    flagged: tuple[Stratum, ...]


def _mps() -> torch.device:
    """The MPS device the pinned base model runs on.

    Raises:
        UnavailableError: If MPS is unavailable.
    """
    if not torch.backends.mps.is_available():
        msg = "MPS is unavailable; the pinned base model fine-tunes on MPS only"
        raise UnavailableError(msg)
    return torch.device("mps")


def read_holdout(request: Request, data_dir: Path) -> Holdout:
    """The run's holdout, prepared and checked against the folder its store is bound to.

    Raises:
        EvaluationError: If no run has the name.
        IdentityError: As `bind`, `checked` and `frame_of`, or if the store holds no fill of the
            arm at the holdout.
        DatasetError: As `bko.evaluate.data.load`.
    """
    named = RUNS.get(request.run)
    if named is None:
        msg = f"no run named {request.run!r}"
        raise EvaluationError(msg)
    reach = replace(EvaluateContext(), experiment=named.experiment, dataset=named.dataset)
    data = load(named.dataset, data_dir, reach)
    rows, folder = read_rows(request.rows), read_folder(request.folder)
    bind(rows, folder, data.frame, run=request.run, data=data.digest)
    ready = checked(request.holdout, Store(request.rows, rows, folder), data, named.experiment)
    split = ready.ablated
    if request.arm == REAL_ARM:
        training = ready.sized.train
    elif request.arm in {None, NONE_ARM}:
        training = split.train
    else:
        key = (request.holdout, request.arm)
        if key not in rows.lines or key not in folder.arms:
            msg = f"{request.rows} holds no fill of {request.arm} at holdout {request.holdout}"
            raise EvaluationError(msg)
        added = frame_of(rows.lines[key], arm_line=folder.arms[key], frame=data.frame)
        training = pd.concat([split.train, added])
    cells = flagged(split.train, named.experiment, edges=data.frame.edges)
    return Holdout(data.frame, training, split.test, cells)


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener `fetch` downloads through.
        load_base: Verifies and loads the base model and its tokenizer.
        base: The pinned base model.
        device: The device the model runs on.
        holdout: Reads a holdout's sets for an arm.
        settings: The fine-tune's LoRA settings.
    """

    open_url: UrlOpener = open_https
    load_base: Callable[..., tuple[PreTrainedModel, PreTrainedTokenizerBase]] = load_base
    base: BaseModel = OLMO_3_7B
    device: Callable[[], torch.device] = _mps
    holdout: Callable[[Request, Path], Holdout] = read_holdout
    settings: LoraSettings = LORA


def adapter_dir(data_dir: Path, folder: Path, holdout: int, arm: str) -> Path:
    """Where the adapter of `arm` at `holdout` of the run in `folder` lives."""
    return data_dir / ADAPTERS / folder.name / f"{holdout}-{arm}"


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Run one subcommand and return the exit code.

    Returns:
        0 done, 2 the command could not run, 3 an input contradicts its pin or contract. Any other
            exception propagates, so the process exits 1.
    """
    args = _parser().parse_args(argv)
    ctx = Context() if context is None else context
    data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
    try:
        text = args.handler(args, ctx, data_dir)
    except (DatasetError, EvaluationError) as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    (out or sys.stdout).write(text)
    return 0


def _fetch(args: argparse.Namespace, context: Context, data_dir: Path) -> str:
    del args
    directory = fetch(context.base, data_dir, open_url=context.open_url)
    logger.info("fetch files=%d path=%s", len(context.base.files), directory)
    return ""


def _request(args: argparse.Namespace) -> Request:
    return Request(args.run, args.folder, args.rows, args.holdout, getattr(args, "arm", None))


def _train(args: argparse.Namespace, context: Context, data_dir: Path) -> str:
    log_versions()
    path = adapter_dir(data_dir, args.folder, args.holdout, args.arm)
    if path.exists():
        msg = f"{path} exists; an adapter is never written over"
        raise DatasetError(msg)
    device = context.device()
    held = context.holdout(_request(args), data_dir)
    logger.info("train holdout=%d arm=%s rows=%d", args.holdout, args.arm, len(held.train))
    model, tokenizer = context.load_base(context.base, data_dir, dtype=DTYPE, device=device)
    fitted = train(
        model,
        tokenizer,
        examples(held.train, held.frame),
        settings=context.settings,
        device=device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = save_adapter(fitted, path)
    size = (path / ADAPTER_FILE).stat().st_size
    logger.info("adapter bytes=%d path=%s sha256=%s", size, path, digest)
    return ""


def _score(args: argparse.Namespace, context: Context, data_dir: Path) -> str:
    log_versions()
    device = context.device()
    held = context.holdout(_request(args), data_dir)
    model, tokenizer = context.load_base(context.base, data_dir, dtype=DTYPE, device=device)
    if args.arm is not None:
        path = adapter_dir(data_dir, args.folder, args.holdout, args.arm)
        weights = path / ADAPTER_FILE
        if not weights.is_file():
            msg = f"{weights} is absent; run `python -m bko.slm train` first"
            raise UnavailableError(msg)
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        logger.info("adapter bytes=%d path=%s sha256=%s", weights.stat().st_size, path, digest)
        model = PeftModel.from_pretrained(model, str(path), is_trainable=False)
    scored = probabilities(model, tokenizer, held.test, frame=held.frame, device=device)
    measures = utility(held.test, scored, held.flagged, edges=held.frame.edges, calibration=True)
    for (measure, scope), value in measures.items():
        logger.info("measure name=%s scope=%s value=%.6f", measure.value, scope.value, value)
    buffer = io.StringIO()
    scored.to_csv(buffer, index_label="record_id", lineterminator="\n")
    text = buffer.getvalue()
    sha = hashlib.sha256(text.encode()).hexdigest()
    logger.info("probabilities rows=%d sha256=%s", len(scored), sha)
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.slm", description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetching = commands.add_parser("fetch", help="fetch and verify the pinned base model")
    fetching.set_defaults(handler=_fetch)
    training = commands.add_parser("train", help="fine-tune one arm's adapter at one holdout")
    _holdout_arguments(training)
    training.add_argument("--arm", required=True, help="the arm whose training set is fitted")
    training.set_defaults(handler=_train)
    scoring = commands.add_parser("score", help="write each test record's P(yes)")
    _holdout_arguments(scoring)
    which = scoring.add_mutually_exclusive_group(required=True)
    which.add_argument("--arm", default=None, help="score the adapter fitted for this arm")
    which.add_argument("--base", action="store_true", help="score the base model, no adapter")
    scoring.set_defaults(handler=_score)
    return parser


def _holdout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="the named run the folder recorded")
    parser.add_argument("--folder", type=Path, required=True, help="the run's folder")
    parser.add_argument("--rows", type=Path, required=True, help="the folder's rows store")
    parser.add_argument("--holdout", type=int, required=True, help="the holdout's seed")


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr, keeping stdout for the CSV."""
    configure()


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
