"""TableShift's Hospital Readmission partition of Diabetes 130, rebuilt in numpy.

TableShift (`mlfoundations/tableshift` at `COMMIT`) drops the rows whose `race` is `?`, labels any
readmission and puts the rows admitted from source 7 out of domain. Its three
`train_test_split` calls at seed `SEED` are `shuffle_split` here: the in-domain rows split 0.2
held, the held part 0.5 in-domain test, and the out-of-domain rows 0.1 validation. A record id is
the row's 1-based position among the archive's rows, in file order. `load_tableshift` loads the
training split with one domain's test split held. ADR `research/tableshift-comparison` records
the partition and what differs from TableShift.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from bko.datasets.archive import member_text
from bko.datasets.diabetes import MEMBER, checked, parse_diabetic
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import DIABETES_130, Source
from bko.datasets.tableshift_frame import (
    DOMAIN,
    DRUG_COLUMNS,
    ID_FIELDS,
    STATES,
    encode,
    tableshift_frame,
)


logger = logging.getLogger(__name__)

COMMIT: Final = "fca9429814703a07e3902d005d46563a207b7f0a"
"""The `mlfoundations/tableshift` commit the task is read from."""

SEED: Final = 264738
"""TableShift's `DEFAULT_RANDOM_STATE`, the seed of each of its three splits."""

OOD_SOURCE: Final = 7
"""The `admission_source_id` TableShift puts out of domain."""

SHARE: Final = 0.1
"""TableShift's in-domain test, in-domain validation and out-of-domain validation share."""

MISSING_RACE: Final = "?"
"""The `race` of the rows TableShift drops."""

_WHOLE: Final = r"[0-9]{1,18}"

type Domain = Literal["id", "ood"]
"""Which test split a loaded dataset holds: in domain or out of domain."""


@dataclass(frozen=True, slots=True)
class Partition:
    """TableShift's five parts, each a record id index in its split's permuted order.

    Attributes:
        train: The in-domain training split.
        validation: The in-domain validation split.
        test: The in-domain test split.
        ood_validation: The out-of-domain validation split.
        ood_test: The out-of-domain test split.
    """

    train: pd.Index
    validation: pd.Index
    test: pd.Index
    ood_validation: pd.Index
    ood_test: pd.Index


def shuffle_split(positions: np.ndarray, share: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split `positions` as scikit-learn's `train_test_split` does at `test_size=share`.

    The first `ceil(share * n)` of `RandomState(seed).permutation(n)` are held, the rest kept.

    Returns:
        The rest, then the held part, each in the permutation's order.
    """
    count = math.ceil(share * len(positions))
    order = np.random.RandomState(seed).permutation(len(positions))
    return positions[order[count:]], positions[order[:count]]


def kept_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """The rows of `raw` TableShift keeps, once every value they are read for fits the schema.

    Args:
        raw: The output of `bko.datasets.diabetes.parse_diabetic`, every row in file order.

    Returns:
        The rows whose `race` is not `MISSING_RACE`, as strings, indexed by `record_id`, each
        row's 1-based position in `raw`.

    Raises:
        ContractError: As `bko.datasets.diabetes.checked`, or if a drug column, an id column or
            the domain column holds a value outside its codes.
    """
    checked(raw)
    rows = raw.reset_index(drop=True)
    rows.index = pd.RangeIndex(1, len(rows) + 1, name="record_id")
    tests = [
        *((name, STATES[name], f"not one of {sorted(STATES[name])}") for name in DRUG_COLUMNS),
        *((name, None, "not a whole number") for name in (*ID_FIELDS, DOMAIN)),
    ]
    for name, codes, why in tests:
        column = rows[name]
        fits = column.isin(list(codes)) if codes is not None else column.str.fullmatch(_WHOLE)
        bad = ~fits.to_numpy(dtype=bool)
        if bad.any():
            first = int(bad.argmax())
            msg = f"{MEMBER} row {rows.index[first]}: {name} is {column.iloc[first]!r}, {why}"
            raise ContractError(msg)
    return rows.loc[rows["race"] != MISSING_RACE]


def partition(raw: pd.DataFrame) -> Partition:
    """TableShift's partition of `raw`, the output of `parse_diabetic`.

    Raises:
        ContractError: As `kept_rows`.
    """
    return _partition(kept_rows(raw))


def _partition(kept: pd.DataFrame) -> Partition:
    ids = kept.index.to_numpy()
    outside = (kept[DOMAIN].astype("int64") == OOD_SOURCE).to_numpy()
    train, held = shuffle_split(ids[~outside], SHARE + SHARE, SEED)
    validation, test = shuffle_split(held, SHARE / (SHARE + SHARE), SEED)
    ood_test, ood_validation = shuffle_split(ids[outside], SHARE, SEED)
    return Partition(
        *(pd.Index(part, dtype="int64") for part in (train, validation, test)),
        ood_validation=pd.Index(ood_validation, dtype="int64"),
        ood_test=pd.Index(ood_test, dtype="int64"),
    )


def load_tableshift(
    data_dir: Path, *, source: Source = DIABETES_130, domain: Domain
) -> tuple[pd.DataFrame, frozenset[int], FrameSpec]:
    """Load the training split and `domain`'s test split from the verified snapshot of `source`.

    This never fetches. Logs one `tableshift partition` line with every part's size.

    Returns:
        The records (`bko.datasets.tableshift_frame.encode`) of the training split and the test
        split, in record id order; the test split's record ids, which every holdout holds out;
        and the frame of every kept row.

    Raises:
        UnavailableError: If the snapshot is absent.
        ContractError: If the archive contradicts its pin or schema (`kept_rows`), or no row
            records a race.
    """
    data = read_verified(source, data_dir)
    raw = parse_diabetic(member_text(data, MEMBER, source))
    kept = kept_rows(raw)
    if kept.empty:
        msg = f"{source.name}: no row records a race"
        raise ContractError(msg)
    parts = _partition(kept)
    logger.info(
        "tableshift partition commit=%s seed=%d train=%d validation=%d test=%d "
        "ood_validation=%d ood_test=%d dropped_race=%d",
        COMMIT,
        SEED,
        len(parts.train),
        len(parts.validation),
        len(parts.test),
        len(parts.ood_validation),
        len(parts.ood_test),
        len(raw) - len(kept),
    )
    test = parts.test if domain == "id" else parts.ood_test
    chosen = np.sort(np.concatenate([parts.train.to_numpy(), test.to_numpy()]))
    records = encode(kept).loc[chosen]
    return records, frozenset(int(record) for record in test), tableshift_frame(kept)
