"""The `diabetes-codes` frame's ICD-9 code rows, generated from MONDO.

`python -m bko.ontology codes` writes this module; regenerate it rather than editing it. Each row
maps one exact diagnosis code of the `diabetes-codes` cohort to the MONDO concept that carries its
`ICD9:` xref under `source="MONDO:equivalentTo"` (ADR `research/diabetes-diagnosis-codes`).
`_ROWS` holds one (code, concept, label) tuple per line, and `_ALSO` the other live holders of
such an xref for a code that has more than one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from bko.ontology.icd9 import Icd9Code


# fmt: off
_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("035", "MONDO:0001266", "erysipelas"),
    ("042", "MONDO:0005109", "HIV infectious disease"),
    ("061", "MONDO:0005502", "dengue disease"),
    ("084", "MONDO:0005136", "malaria"),
    ("135", "MONDO:0008399", "sarcoidosis, susceptibility to, 1"),
    ("179", "MONDO:0002715", "uterine cancer"),
    ("185", "MONDO:0008315", "prostate cancer"),
    ("193", "MONDO:0002108", "thyroid cancer"),
    ("217", "MONDO:0000620", "breast benign neoplasm"),
    ("226", "MONDO:0006107", "benign thyroid gland neoplasm"),
    ("260", "MONDO:0006826", "kwashiorkor"),
    ("261", "MONDO:0006848", "marasmus"),
    ("268", "MONDO:0100471", "vitamin D deficiency"),
    ("319", "MONDO:0001071", "intellectual disability"),
    ("430", "MONDO:0005099", "subarachnoid hemorrhage"),
    ("452", "MONDO:0001339", "portal vein thrombosis"),
    ("460", "MONDO:0005709", "common cold"),
    ("475", "MONDO:0005906", "peritonsillar abscess"),
    ("485", "MONDO:0005682", "bronchopneumonia"),
    ("486", "MONDO:0005249", "pneumonia"),
    ("494", "MONDO:0004822", "bronchiectasis"),
    ("500", "MONDO:0006654", "anthracosis"),
    ("501", "MONDO:0016466", "asbestosis"),
    ("515", "MONDO:0001438", "postinflammatory pulmonary fibrosis"),
    ("570", "MONDO:0002251", "hepatitis"),
    ("585", "MONDO:0005300", "chronic kidney disease"),
    ("586", "MONDO:0001106", "kidney failure"),
    ("591", "MONDO:0005510", "hydronephrosis"),
    ("683", "MONDO:0002052", "lymphadenitis"),
    ("684", "MONDO:0004592", "impetigo"),
    ("690", "MONDO:0006546", "erythematosquamous dermatosis"),
    ("725", "MONDO:0019735", "polymyalgia rheumatica"),
)
# fmt: on

_ALSO: Final[Mapping[str, tuple[str, ...]]] = {}

ICD9_CODES: Final[tuple[Icd9Code, ...]] = tuple(
    Icd9Code(code, concept, label, frozenset({concept, *_ALSO.get(code, ())}))
    for code, concept, label in _ROWS
)
"""Each mapped code, in encoding order, with its concept and every live exact holder of its xref."""
