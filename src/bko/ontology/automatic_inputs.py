"""What an automatic mapping of each dataset reads: its frame, the pinned releases, its code fields.

`bko.ontology.automatic`'s `lexical` command and `bko.generate.judge` read a dataset through
`inputs`, so both builders see the same codebook. `diabetes-codes`' codebook is the codes its
cohort holds, read through `bko.datasets.diabetes.load_diabetes` and
`bko.ontology.code_frames.diabetes_codes_frame`; no record-level statistic is read (ADR
`research/automatic-mapping`, the leakage rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from bko.datasets.diabetes import DIAGNOSES, load_diabetes
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import DIABETES_130
from bko.datasets.uci import UCI_FRAME
from bko.ontology.code_frames import DIABETES_CODES_NAME, diabetes_codes_frame
from bko.ontology.frames import DIABETES_FRAME, DIABETES_NAME
from bko.ontology.mapping import RELEASES
from bko.ontology.releases import load_releases
from bko.ontology.terms import Ontology


DATASETS: Final = ("uci", DIABETES_NAME, DIABETES_CODES_NAME)
"""The datasets an automatic mapping is built for."""


def inputs(
    dataset: str, data_dir: Path | None = None, *, open_url: UrlOpener = open_https
) -> tuple[FrameSpec, dict[str, Ontology], tuple[str, ...]]:
    """`dataset`'s frame, its releases by prefix, and its fields whose codes are ICD-9 tokens.

    The MONDO and HPO releases, and for `diabetes-codes` the archive, are fetched into `data_dir`
    if absent.

    Raises:
        ValueError: If `dataset` is not one of `DATASETS`.
        DatasetError: If a snapshot cannot be fetched or contradicts its pin.
    """
    if dataset not in DATASETS:
        msg = f"no automatic mapping for {dataset!r}; the datasets are {list(DATASETS)}"
        raise ValueError(msg)
    root = default_data_dir() if data_dir is None else data_dir
    ontologies = load_releases(RELEASES, root, open_url=open_url)
    if dataset == "uci":
        return UCI_FRAME, ontologies, ()
    if dataset == DIABETES_NAME:
        return DIABETES_FRAME, ontologies, ()
    ensure(DIABETES_130, root, open_url=open_url)
    records = load_diabetes(root, indicators={}, diagnoses=True, source=DIABETES_130)
    return diabetes_codes_frame(records), ontologies, DIAGNOSES
