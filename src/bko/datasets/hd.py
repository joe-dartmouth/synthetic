"""Lautrup et al.'s `hd` files read as records of the UCI frame.

Their heart-disease files (`experiments/datasets/heart_train.csv` and `heart_test.csv` at commit
`a07cd9c007b58bee37b0837cfdc65252303afadc` of `notna07/disjoint-synthetic-data-generation`)
recode UCI's processed Cleveland file: `cp`, `restecg`, `slope` and `thal` under other codes, a
missing `ca` as 4 and a missing `thal` as 0, and `target` 1 where UCI's `num` is 0. `from_hd`
reads one file into `bko.datasets.uci.UCI_FRAME`'s columns and codes. The maps are those of a
match of every row of theirs to the Cleveland file; ADR `research/syntheval-reading` records it.
"""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Iterable, Mapping
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES
from bko.datasets.errors import ContractError
from bko.datasets.uci import CODEBOOK, FEATURES, FRACTIONAL, FRAME_COLUMNS, FRAME_DTYPES


logger = logging.getLogger(__name__)

HD_COLUMNS: Final = (*FEATURES, LABEL)
"""Their files' columns, in file order: UCI's 13 predictors under UCI's names, then `target`."""

CP_CODES: Final[Mapping[int, int]] = {0: 4, 1: 2, 2: 3, 3: 1}
"""Their `cp` code to UCI's."""

RESTECG_CODES: Final[Mapping[int, int]] = {0: 2, 1: 0, 2: 1}
"""Their `restecg` code to UCI's."""

SLOPE_CODES: Final[Mapping[int, int]] = {0: 3, 1: 2, 2: 1}
"""Their `slope` code to UCI's."""

THAL_CODES: Final[Mapping[int, int]] = {1: 6, 2: 3, 3: 7}
"""Their `thal` code to UCI's."""

MISSING_CA: Final = 4
"""Their code for a `ca` UCI holds as missing."""

MISSING_THAL: Final = 0
"""Their code for a `thal` UCI holds as missing."""

RECODED: Final[Mapping[str, Mapping[int, int]]] = {
    "cp": CP_CODES,
    "restecg": RESTECG_CODES,
    "slope": SLOPE_CODES,
    "thal": THAL_CODES,
}
"""Each recoded field's map from their code to UCI's."""

MISSING: Final[Mapping[str, int]] = {"ca": MISSING_CA, "thal": MISSING_THAL}
"""Each field's code for a missing value."""


def from_hd(text: str) -> pd.DataFrame:
    """Read one `hd` file into the UCI frame's columns, codes and dtypes.

    A row holding a missing code is dropped; one `hd rows=%d kept=%d dropped=%d missing_ca=%d
    missing_thal=%d` line counts them. The index is `record_id`, the row's 1-based position among
    the file's data rows.

    Args:
        text: The file's text, a header line and one line per row.

    Returns:
        The kept rows in `FRAME_COLUMNS`, at `FRAME_DTYPES`, `target` 1 where theirs is 0.

    Raises:
        ContractError: If the columns are not `HD_COLUMNS`, a value is not a finite number, a
            whole-number field holds a fraction, a code is outside its field's map, UCI's
            codebook or the label codes, or no row is complete.
    """
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (ValueError, pd.errors.ParserError) as exc:
        msg = f"the hd file does not read as CSV: {exc}"
        raise ContractError(msg) from exc
    if tuple(raw.columns) != HD_COLUMNS:
        msg = f"the hd file's columns {list(raw.columns)} are not {list(HD_COLUMNS)}"
        raise ContractError(msg)
    raw.index = pd.Index(range(1, len(raw) + 1), name="record_id")
    values = pd.DataFrame(
        {
            name: [_number(value, name, row) for row, value in raw[name].items()]
            for name in HD_COLUMNS
        },
        index=raw.index,
    )
    missing = {name: values[name] == code for name, code in MISSING.items()}
    dropped = missing["ca"] | missing["thal"]
    kept = values.loc[~dropped]
    logger.info(
        "hd rows=%d kept=%d dropped=%d missing_ca=%d missing_thal=%d",
        len(values),
        len(kept),
        int(dropped.sum()),
        int(missing["ca"].sum()),
        int(missing["thal"].sum()),
    )
    if kept.empty:
        msg = "the hd file holds no complete row"
        raise ContractError(msg)
    frame = kept.copy()
    for name in FEATURES:
        if name in RECODED:
            frame[name] = _recoded(kept[name], name, RECODED[name])
        elif name in CODEBOOK:
            _within(kept[name], name, CODEBOOK[name])
    _within(kept[LABEL], LABEL, LABEL_CODES)
    frame[LABEL] = 1 - kept[LABEL]
    return frame[list(FRAME_COLUMNS)].astype(dict(FRAME_DTYPES))


def _number(value: str, name: str, row: object) -> float:
    """`value` of field `name` in row `row` as a float, refused unless a finite number of its kind.

    Raises:
        ContractError: If it is not a finite number, or `name` is a whole-number field and it
            holds a fraction.
    """
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        msg = f"hd row {row}: {name} is {value!r}, not a finite number"
        raise ContractError(msg)
    if name not in FRACTIONAL and not number.is_integer():
        msg = f"hd row {row}: {name} is {value!r}, not a whole number"
        raise ContractError(msg)
    return number


def _recoded(column: pd.Series, name: str, codes: Mapping[int, int]) -> pd.Series:
    """`column` mapped from their codes to UCI's.

    Raises:
        ContractError: If a code is outside `codes`.
    """
    _within(column, name, codes)
    return column.map({float(theirs): float(ours) for theirs, ours in codes.items()})


def _within(column: pd.Series, name: str, codes: Iterable[int]) -> None:
    """Refuse `column` unless every value is one of `codes`.

    Raises:
        ContractError: Naming the first row whose value is outside them.
    """
    allowed = sorted(codes)
    outside = column[~column.isin(allowed)]
    if not outside.empty:
        msg = f"hd row {outside.index[0]}: {name} is {outside.iloc[0]:g}, outside {allowed}"
        raise ContractError(msg)
