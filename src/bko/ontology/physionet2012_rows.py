"""PhysioNet 2012's field rows: the abnormal states of its reduced physiology.

Each row is a `Codes` clause on a state `bko.datasets.physionet2012` codes from a stay's maximum
or minimum over its first 48 hours, so a stay that never measured the variable (state 0) shows no
concept. The cutoffs are the SIRS criteria (`bone-1992`) and SOFA's first abnormal score
(`vincent-1996`), read in Singer et al.'s reproduction (`singer-2016`, Box 1 and Table 1).
`ICUType` and `mechvent` map nothing. Labels are the pinned HPO release's `name:` values. ADR
`research/physionet2012-frame` records the rows.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.physionet2012 import REDUCED
from bko.datasets.states import band_state
from bko.ontology.schema import Codes, FieldMapping


_READ_IN: Final = "read in singer-2016's reproduction"
_SIRS: Final = "a SIRS criterion of bone-1992"
_SOFA: Final = "SOFA's first abnormal score of vincent-1996"


def state_row(field: str, band: str, concept: str, label: str, criterion: str) -> FieldMapping:
    """The row of the state of `field` whose band is `band`, its source the reproduction read.

    The state is found by its band's name (`bko.datasets.states.band_state`), so a changed
    cutoff refuses the row instead of moving its concept to another band.

    Raises:
        ValueError: If `field`'s cutoffs cite no source or have no such band.
    """
    variable, how, cut = REDUCED[field]
    if cut.source is None:
        msg = f"{field}'s cutoffs cite no source, so no row maps its states"
        raise ValueError(msg)
    extreme = "highest" if how == "max" else "lowest"
    return FieldMapping(
        when=(Codes(field, frozenset({band_state(cut, band)})),),
        concept=concept,
        label=label,
        source=cut.source,
        limits=(
            f"{field} {band}, the stay's {extreme} {variable} over its first 48 hours, "
            f"{criterion} ({_READ_IN}); one reading, not a diagnosis; a stay that never "
            "measured it is state 0 and shows nothing"
        ),
    )


PHYSIONET2012_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = (
    state_row("hr_max", "above 90", "HP:0001649", "Tachycardia", _SIRS),
    state_row("resprate_max", "above 20", "HP:0002789", "Tachypnea", _SIRS),
    state_row("temp_max", "above 38", "HP:0001945", "Fever", _SIRS),
    state_row("temp_min", "below 36", "HP:0002045", "Hypothermia", _SIRS),
    state_row("wbc_max", "above 12", "HP:0001974", "Increased total leukocyte count", _SIRS),
    state_row("wbc_min", "below 4", "HP:0001882", "Decreased total leukocyte count", _SIRS),
    state_row("map_min", "below 70", "HP:0002615", "Hypotension", _SOFA),
    state_row(
        "creatinine_max",
        "at least 1.2",
        "HP:0003259",
        "Elevated circulating creatinine concentration",
        _SOFA,
    ),
    state_row("bilirubin_max", "at least 1.2", "HP:0002904", "Hyperbilirubinemia", _SOFA),
    state_row("platelets_min", "below 150", "HP:0001873", "Thrombocytopenia", _SOFA),
    state_row("gcs_min", "below 15", "HP:0004372", "Reduced consciousness", _SOFA),
)
"""The PhysioNet 2012 frame's field rows: the SIRS fields' abnormal states, then SOFA's."""
