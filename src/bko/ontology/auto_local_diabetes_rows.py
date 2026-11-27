"""Local judge verdicts, written by `python -m bko.generate.judge`.

Regenerate this module rather than editing it. `VERDICTS` holds each candidate state's concept,
or `none`, keyed `<field>=<code>`, the states the cross-reference route maps included, and
`LABELS` each verdict concept's name. The rows are
`bko.ontology.automatic.rows_from_verdicts(frame, VERDICTS, LABELS)`, a split table's two parts
merged first (ADR `research/automatic-mapping`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final


DATASET: Final = "diabetes"
MODEL: Final = "gemma-4-26b-a4b-q8_0"

# fmt: off
VERDICTS: Final[Mapping[str, str]] = {
    "A1Cresult=2": "HP:0040217",
    "A1Cresult=3": "HP:0040217",
    "acute_myocardial_infarction=1": "MONDO:0004781",
    "cerebral_artery_occlusion=1": "MONDO:0000944",
    "change=1": "none",
    "chronic_bronchitis=1": "MONDO:0005607",
    "chronic_kidney_disease=1": "MONDO:0005300",
    "chronic_obstructive_pulmonary_disease=1": "MONDO:0005002",
    "chronic_ulcer_of_skin=1": "MONDO:0004605",
    "diabetesMed=1": "none",
    "diabetic_kidney_disease=1": "MONDO:0005016",
    "essential_hypertension=1": "MONDO:0001134",
    "heart_failure=1": "MONDO:0005252",
    "insulin=1": "none",
    "insulin=2": "none",
    "insulin=3": "none",
    "max_glu_serum=2": "none",
    "max_glu_serum=3": "none",
    "pneumonia=1": "MONDO:0005249",
    "race=3": "none",
    "race=4": "none",
    "race=5": "none",
    "sex=0": "none",
    "sex=1": "none",
    "target=1": "none",
}

LABELS: Final[Mapping[str, str]] = {
    "HP:0040217": "Elevated hemoglobin A1c",
    "MONDO:0000944": "cerebral artery occlusion",
    "MONDO:0001134": "essential hypertension",
    "MONDO:0004605": "chronic ulcer of skin",
    "MONDO:0004781": "acute myocardial infarction",
    "MONDO:0005002": "chronic obstructive pulmonary disease",
    "MONDO:0005016": "diabetic kidney disease",
    "MONDO:0005249": "pneumonia",
    "MONDO:0005252": "heart failure",
    "MONDO:0005300": "chronic kidney disease",
    "MONDO:0005607": "chronic bronchitis",
}
# fmt: on
