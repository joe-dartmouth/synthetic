"""The data a named run scores: one `Data` value, chosen by `NamedRun.dataset`, every stage reads.

`score` loads it once (`load`) and hands it to the harness, the BKO arms, the rehearsal and the
candidates file, so each reads the same records, frame, field mappings and digest, and the run's
logs name the frame. ADR `research/uci-evaluation-frame` records why the run, not a flag, chooses
it. The Synthea and Diabetes 130 frames, their indicators and their field rows are in
`bko.ontology.frames`, and the `diabetes-codes` frame and its rows in `bko.ontology.code_frames`.
The two TableShift datasets hold a published partition's test records out at every holdout seed
(`Data.held`, `held_split`; ADR `research/tableshift-comparison`). A variant (`VARIANTS`) is a base
dataset's records under other field rows or projected onto `supported_frame`, under its own name
(ADR `research/dataset-variants`).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, assert_never

import pandas as pd

from bko.datasets.acs import ACS_NAME
from bko.datasets.diabetes import load_diabetes
from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.frames import FrameSpec, check_records, digest, digested, log_data
from bko.datasets.nhamcs import NHAMCS_NAME
from bko.datasets.nhanes import NHANES_NAME
from bko.datasets.physionet2012 import PHYSIONET2012_NAME
from bko.datasets.sources import Agreement, Source
from bko.datasets.split import Split, fixed, split
from bko.datasets.support2 import SUPPORT2_NAME
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, frame_records, load_population
from bko.datasets.synthea_run import SyntheaRun, check_population, population_dir
from bko.datasets.tableshift import load_tableshift
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.ontology.catalog import Catalogued
from bko.ontology.code_frames import (
    DIABETES_CODES_FIELD_ROWS,
    DIABETES_CODES_NAME,
    diabetes_codes_frame,
)
from bko.ontology.frames import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    DIABETES_INDICATORS,
    DIABETES_NAME,
    LAB_FIELD_ROWS,
    SYNTHEA_ADULTS_FROM,
    SYNTHEA_FIELD_ROWS,
    SYNTHEA_FRAME,
    SYNTHEA_INDICATORS,
    SYNTHEA_NAME,
)
from bko.ontology.schema import Codes, FieldMapping


logger = logging.getLogger("bko.evaluate")


class Dataset(StrEnum):
    """The datasets a named run can score."""

    UCI = "uci"
    SYNTHEA = "synthea"
    DIABETES = DIABETES_NAME
    DIABETES_CODES = DIABETES_CODES_NAME
    TABLESHIFT_ID = "tableshift-id"
    TABLESHIFT_OOD = "tableshift-ood"
    SUPPORT2 = SUPPORT2_NAME
    ACS = ACS_NAME
    NHANES = NHANES_NAME
    NHAMCS = NHAMCS_NAME
    PHYSIONET2012 = PHYSIONET2012_NAME
    DIABETES_SUPPORTED = "diabetes-supported"


@dataclass(frozen=True, slots=True, eq=False)
class Data:
    """A run's records and what every stage reads about them.

    Attributes:
        records: The complete records, indexed by `record_id`, in `frame`'s columns.
        frame: Their frame.
        field_rows: The field mappings the BKO arms observe cells by.
        name: The dataset's source name; each cell's ontology fragment takes it.
        agreement: The data use agreement the records are held under; None for open data. Every
            Messages arm refuses records whose agreement grants no permission to send them.
        digest: The sha256 of `records` as `digest` renders them, computed from them when the
            value is made, which the candidates file's header names; with `held`, the sha256
            of that text followed by a `held=` line of the sorted held ids (`held_digest`).
        held: The record ids every holdout holds out, a published partition's test split; None
            splits each holdout by its seed (`bko.datasets.split.split`).
    """

    records: pd.DataFrame
    frame: FrameSpec
    field_rows: tuple[FieldMapping, ...]
    name: str
    agreement: Agreement | None
    held: frozenset[int] | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute `digest` from `records` and `held`, so they never disagree."""
        found = digest(self.records) if self.held is None else held_digest(self.records, self.held)
        object.__setattr__(self, "digest", found)


@dataclass(frozen=True, slots=True)
class Variant:
    """A dataset read as its base's records under other rows or on fewer columns.

    Attributes:
        base: The dataset whose records, split, source name and releases it reads.
        rows: Its field rows; None keeps the base's.
        supported: Whether its records are projected onto the base frame's `supported_frame`.
    """

    base: Dataset
    rows: tuple[FieldMapping, ...] | None = None
    supported: bool = False


def supported_frame(frame: FrameSpec, rows: Sequence[FieldMapping]) -> FrameSpec:
    """`frame` on `age`, `sex`, the label and every field a `Codes` clause of `rows` reads.

    Its name is `frame`'s with `-supported` after it; every other value is `frame`'s, kept for
    the columns it keeps.
    """
    coded = {c.field for row in rows for c in row.when if isinstance(c, Codes)}
    kept = {"age", "sex", frame.label} | coded

    def keep(names: Sequence[str]) -> tuple[str, ...]:
        return tuple(name for name in names if name in kept)

    return FrameSpec(
        name=f"{frame.name}-supported",
        features=keep(frame.features),
        numeric=keep(frame.numeric),
        fractional=keep(frame.fractional),
        fraction_digits=frame.fraction_digits,
        categorical=keep(frame.categorical),
        codebook={name: codes for name, codes in frame.codebook.items() if name in kept},
        descriptions={name: text for name, text in frame.descriptions.items() if name in kept},
        ranges=tuple(bound for bound in frame.ranges if bound.field in kept),
        label_codes=frame.label_codes,
        label_meanings=frame.label_meanings,
        edges=frame.edges,
        study=frame.study,
        cohort=frame.cohort,
        label=frame.label,
    )


VARIANTS: Final[Mapping[Dataset, Variant]] = MappingProxyType(
    {Dataset.DIABETES_SUPPORTED: Variant(Dataset.DIABETES, supported=True)}
)
"""Each variant over its base (ADR `research/dataset-variants`)."""

_SUPPORTED_FRAMES: Final[Mapping[Dataset, FrameSpec]] = MappingProxyType(
    {Dataset.DIABETES_SUPPORTED: supported_frame(DIABETES_FRAME, DIABETES_FIELD_ROWS)}
)
"""Each supported variant's one frame, built once so frame identity checks hold."""


class Reach(Protocol):
    """What `load` reads of a command's context: the evaluate and app contexts both have it."""

    @property
    def open_url(self) -> UrlOpener:
        """The HTTPS opener."""
        ...

    @property
    def uci(self) -> Source:
        """The pinned UCI archive."""
        ...

    @property
    def field_rows(self) -> tuple[FieldMapping, ...]:
        """The field mappings."""
        ...

    @property
    def synthea(self) -> SyntheaRun:
        """The pinned Synthea population's run."""
        ...

    @property
    def synthea_jar(self) -> Source:
        """The jar the pinned Synthea population was generated with."""
        ...

    @property
    def diabetes(self) -> Source:
        """The pinned Diabetes 130 archive."""
        ...

    @property
    def releases(self) -> Mapping[str, Source]:
        """The pinned ontology release for each concept prefix."""
        ...

    @property
    def catalog(self) -> Mapping[str, Catalogued]:
        """The catalogued datasets by name (`bko.ontology.catalog`)."""
        ...


def held_digest(records: pd.DataFrame, held: frozenset[int]) -> str:
    """The sha256 of `records` as `digested` renders them, then `held=` and the sorted held ids.

    Two runs whose records agree but whose held test splits differ get different digests.
    """
    ids = ",".join(str(record) for record in sorted(held))
    return hashlib.sha256(f"{digested(records)[0]}held={ids}\n".encode()).hexdigest()


def held_split(data: Data, seed: int) -> Split:
    """Holdout `seed`'s split of `data`'s records under its frame's edges.

    `bko.datasets.split.split` when `data.held` is None, else `bko.datasets.split.fixed`, the
    same test records at every seed.

    Raises:
        ContractError: As `fixed`, if a held id is not among the records.
    """
    return records_split(data.records, held=data.held, seed=seed, edges=data.frame.edges)


def records_split(
    records: pd.DataFrame, *, held: frozenset[int] | None, seed: int, edges: Sequence[int]
) -> Split:
    """`held_split` of bare records: `split` when `held` is None, else `fixed`.

    Raises:
        ContractError: As `fixed`.
    """
    if held is None:
        return split(records, seed=seed, edges=edges)
    return fixed(records, held=held, seed=seed)


def frame(dataset: Dataset) -> FrameSpec:
    """The frame `load` gives `dataset`'s records, read with no data loaded.

    A supported variant's is its one module-level frame; a row-swap variant's is its base's.

    Raises:
        ValueError: For `diabetes-codes`, the TableShift datasets and the catalogued datasets,
            whose frames are built from their records.
    """
    match dataset:
        case Dataset.UCI:
            return UCI_FRAME
        case Dataset.SYNTHEA:
            return SYNTHEA_FRAME
        case Dataset.DIABETES:
            return DIABETES_FRAME
        case Dataset.DIABETES_SUPPORTED:
            return _SUPPORTED_FRAMES[dataset]
        case (
            Dataset.DIABETES_CODES
            | Dataset.TABLESHIFT_ID
            | Dataset.TABLESHIFT_OOD
            | Dataset.SUPPORT2
            | Dataset.ACS
            | Dataset.NHANES
            | Dataset.NHAMCS
            | Dataset.PHYSIONET2012
        ):
            msg = f"the {dataset} frame is built from its records; load them"
            raise ValueError(msg)
        case _:
            assert_never(dataset)


def field_rows(dataset: Dataset, context: Reach) -> tuple[FieldMapping, ...]:
    """The field rows `load` gives `dataset`'s data, read with no data loaded.

    UCI's are the context's; Synthea's are `SYNTHEA_FIELD_ROWS`, Diabetes 130's
    `DIABETES_FIELD_ROWS`, `diabetes-codes`' `DIABETES_CODES_FIELD_ROWS` and the TableShift
    datasets' `LAB_FIELD_ROWS`, whatever the context's; a catalogued dataset's are its entry's in
    the context's catalog. A variant's are its own, else its base's.
    """
    match dataset:
        case Dataset.UCI:
            return context.field_rows
        case Dataset.SYNTHEA:
            return SYNTHEA_FIELD_ROWS
        case Dataset.DIABETES:
            return DIABETES_FIELD_ROWS
        case Dataset.DIABETES_CODES:
            return DIABETES_CODES_FIELD_ROWS
        case Dataset.TABLESHIFT_ID | Dataset.TABLESHIFT_OOD:
            return LAB_FIELD_ROWS
        case (
            Dataset.SUPPORT2 | Dataset.ACS | Dataset.NHANES | Dataset.NHAMCS | Dataset.PHYSIONET2012
        ):
            return context.catalog[dataset.value].rows
        case Dataset.DIABETES_SUPPORTED:
            variant = VARIANTS[dataset]
            return field_rows(variant.base, context) if variant.rows is None else variant.rows
        case _:
            assert_never(dataset)


def dataset_releases(dataset: Dataset, context: Reach) -> Mapping[str, Source]:
    """The pinned ontology release for each concept prefix `dataset`'s field rows may name.

    Every dataset reads the context's releases, HP and MONDO; a dataset whose rows name another
    ontology gets its own map here (ADR `research/ontology-mapping-design`). A catalogued dataset
    reads its entry's `releases`, the context's when they are None. A variant reads its base's.
    """
    match dataset:
        case (
            Dataset.UCI
            | Dataset.SYNTHEA
            | Dataset.DIABETES
            | Dataset.DIABETES_CODES
            | Dataset.TABLESHIFT_ID
            | Dataset.TABLESHIFT_OOD
        ):
            return context.releases
        case (
            Dataset.SUPPORT2 | Dataset.ACS | Dataset.NHANES | Dataset.NHAMCS | Dataset.PHYSIONET2012
        ):
            return context.catalog[dataset.value].resolved_releases(context.releases)
        case Dataset.DIABETES_SUPPORTED:
            return dataset_releases(VARIANTS[dataset].base, context)
        case _:
            assert_never(dataset)


def load(dataset: Dataset, data_dir: Path, context: Reach) -> Data:
    """Load `dataset` from `data_dir`, fetching a pinned archive the data root lacks.

    UCI's records are read from UCI Heart Disease. Synthea's are the pinned population's, checked
    against its `run.json` with no Java and never generated (`make synthea` generates it), kept
    from `SYNTHEA_ADULTS_FROM`. Diabetes 130's are `bko.datasets.diabetes.load_diabetes`'s over
    `DIABETES_INDICATORS`, checked against `DIABETES_FRAME`
    (`bko.datasets.frames.check_records`). `diabetes-codes`' are the same cohort with the
    diagnosis codes in place of the indicators, checked against `diabetes_codes_frame` of
    themselves. The TableShift datasets' are `bko.datasets.tableshift.load_tableshift`'s training
    split and in-domain or out-of-domain test split, the test split held, checked against the
    `tableshift` frame of every kept row. A catalogued dataset's are its context catalog entry's
    (`bko.ontology.catalog.Catalogued.read`): each source ensured, the records loaded, their frame
    built from them and checked. A variant's are its base's, its rows swapped or its records
    projected onto its frame, under its base's source name. Each takes
    `field_rows(dataset, context)`, and a fetched source's data carries the source's agreement.
    Logs one `data` line naming the dataset, its frame, its record count, its digest (the
    records' sha256, with the held ids when there are some: `Data.digest`) and its agreement:
    `none` for open data, else the agreement's name quoted; a variant logs its own line alone.

    Raises:
        DatasetError: As `bko.datasets.fetch.ensure`, `bko.datasets.uci.load_uci` and
            `bko.datasets.diabetes.load_diabetes` and `bko.datasets.tableshift.load_tableshift`,
            as `check_records`, or as
            `bko.datasets.synthea_run.check_population` and
            `bko.datasets.synthea.load_population`.
        OSError: If the data root cannot be written.
    """
    data = _read(dataset, data_dir, context)
    log_data(
        logger,
        dataset=dataset,
        frame=data.frame,
        records=len(data.records),
        records_digest=data.digest,
        agreement=data.agreement,
    )
    return data


def _read(dataset: Dataset, data_dir: Path, context: Reach) -> Data:
    """`load`'s data, logging no line.

    Raises:
        DatasetError: As `load`.
        OSError: As `load`.
    """
    rows = field_rows(dataset, context)
    match dataset:
        case Dataset.UCI:
            ensure(context.uci, data_dir, open_url=context.open_url)
            records = load_uci(data_dir, source=context.uci)
            data = Data(records, frame(dataset), rows, context.uci.name, context.uci.agreement)
        case Dataset.SYNTHEA:
            run = context.synthea
            directory = population_dir(run, data_dir, context.synthea_jar)
            check_population(directory, run, context.synthea_jar)
            population = load_population(directory, snapshot_date=run.snapshot_date)
            records = frame_records(
                population,
                label=ISCHEMIC_HEART_DISEASE,
                indicators=SYNTHEA_INDICATORS,
                adults_from=SYNTHEA_ADULTS_FROM,
            )
            data = Data(records, frame(dataset), rows, SYNTHEA_NAME, None)
        case Dataset.DIABETES:
            source = context.diabetes
            ensure(source, data_dir, open_url=context.open_url)
            records = load_diabetes(data_dir, indicators=DIABETES_INDICATORS, source=source)
            check_records(records, frame(dataset))
            data = Data(records, frame(dataset), rows, source.name, source.agreement)
        case Dataset.DIABETES_CODES:
            source = context.diabetes
            ensure(source, data_dir, open_url=context.open_url)
            records = load_diabetes(data_dir, indicators={}, diagnoses=True, source=source)
            spec = diabetes_codes_frame(records)
            # The diagnosis codebook is built from these records, and `load_diabetes` has checked
            # each token against `DIAGNOSIS_PATTERN`, so this check cannot fail on a diag column.
            check_records(records, spec)
            data = Data(records, spec, rows, source.name, source.agreement)
        case Dataset.TABLESHIFT_ID | Dataset.TABLESHIFT_OOD:
            source = context.diabetes
            ensure(source, data_dir, open_url=context.open_url)
            domain = "id" if dataset is Dataset.TABLESHIFT_ID else "ood"
            records, held, spec = load_tableshift(data_dir, source=source, domain=domain)
            check_records(records, spec)
            data = Data(records, spec, rows, source.name, source.agreement, held=held)
        case (
            Dataset.SUPPORT2 | Dataset.ACS | Dataset.NHANES | Dataset.NHAMCS | Dataset.PHYSIONET2012
        ):
            entry = context.catalog[dataset.value]
            records, spec = entry.read(data_dir, open_url=context.open_url)
            data = Data(records, spec, rows, entry.name, entry.agreement)
        case Dataset.DIABETES_SUPPORTED:
            data = _read_variant(dataset, rows, data_dir, context)
        case _:
            assert_never(dataset)
    return data


def _read_variant(
    dataset: Dataset, rows: tuple[FieldMapping, ...], data_dir: Path, context: Reach
) -> Data:
    """A variant's data: its base's, under `rows`, projected onto its frame when supported.

    Raises:
        DatasetError: As `load` of its base.
        OSError: As `load`.
    """
    variant = VARIANTS[dataset]
    data = _read(variant.base, data_dir, context)
    spec = frame(dataset) if variant.supported else data.frame
    records = data.records[list(spec.columns)] if variant.supported else data.records
    return Data(records, spec, rows, data.name, data.agreement, held=data.held)
