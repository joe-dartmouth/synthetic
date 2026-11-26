"""NHANES's field rows: the abnormal states of its blood pressure, glycohemoglobin and lipids.

Each row is a `Codes` clause on the states `bko.datasets.nhanes` codes at a cited cutoff, so a
missing value (state 0) shows no concept. The blood-pressure cutoffs are the 2017 ACC/AHA
definition of hypertension (`whelton-2018`), read in `muntner-2018`'s Table 1; glycohemoglobin's
are the ADA's prediabetes and diabetes ranges (`ada-2026`, Tables 2.1 and 2.2); the lipid
cutoffs are ATP III's (`ncep-2001`), read in `nhlbi-2001`'s Table 2. `BMXBMI`, coded at tertiles,
maps nothing. Labels are the pinned HPO release's `name:` values. ADR `research/nhanes-frame`
records the rows.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.nhanes import MEASURED
from bko.datasets.states import band_state
from bko.ontology.schema import Codes, FieldMapping


def state_row(
    field: str, bands: tuple[str, ...], concept: str, label: str, criterion: str
) -> FieldMapping:
    """The row of the states of `field` whose bands are `bands`, its source the record read.

    Each state is found by its band's name (`bko.datasets.states.band_state`), so a changed
    cutoff refuses the row instead of moving its concept to another band.

    Raises:
        ValueError: If `field`'s cutoffs cite no source, as tertiles do, or have no such band.
    """
    cut = MEASURED[field]
    if cut.source is None:
        msg = f"{field}'s cutoffs cite no source, so no row maps its states"
        raise ValueError(msg)
    return FieldMapping(
        when=(Codes(field, frozenset(band_state(cut, band) for band in bands)),),
        concept=concept,
        label=label,
        source=cut.source,
        limits=(
            f"{field} {' or '.join(bands)}, {criterion}; one visit's measurement, not a "
            "diagnosis; a missing value is state 0 and shows nothing"
        ),
    )


_ACC_AHA: Final = (
    "the 2017 ACC/AHA definition of hypertension of whelton-2018, read in muntner-2018's "
    "reproduction"
)
_ATP_III: Final = "ATP III's classification of ncep-2001, read in nhlbi-2001's executive summary"

NHANES_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = (
    state_row(
        "BPXOSY", ("at least 130",), "HP:0004421", "Elevated systolic blood pressure", _ACC_AHA
    ),
    state_row(
        "BPXODI", ("at least 80",), "HP:0005117", "Elevated diastolic blood pressure", _ACC_AHA
    ),
    state_row(
        "LBXGH",
        ("at least 5.7 and below 6.5", "at least 6.5"),
        "HP:0040217",
        "Elevated hemoglobin A1c",
        "the ADA's prediabetes range and diabetes cutoff of ada-2026 (Tables 2.2 and 2.1)",
    ),
    state_row("LBXTC", ("at least 240",), "HP:0003124", "Hypercholesterolemia", _ATP_III),
    state_row(
        "LBDHDD", ("below 40",), "HP:0003233", "Decreased circulating HDL-C concentration", _ATP_III
    ),
)
"""The NHANES frame's field rows: blood pressure, glycohemoglobin, then the lipids."""
