"""A toy codebook and two toy releases for `bko.ontology.automatic` and `bko.generate.judge`.

Every stanza is invented: it reuses MONDO and HPO ids and names only, and no release file content.
"""

from __future__ import annotations

from pathlib import Path

from bko.datasets.cells import SEXES
from bko.datasets.diagnoses import code_number
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.ontology.obo import parse_obo
from bko.ontology.terms import Ontology
from tests._ontology_fixtures import write_obo


MONDO = """\
format-version: 1.2
data-version: releases/2026-10-06
ontology: mondo

[Term]
id: MONDO:0005015
name: diabetes mellitus

[Term]
id: MONDO:0005044
name: hypertensive disorder
synonym: "hypertension" EXACT []

[Term]
id: MONDO:0001134
name: essential hypertension
is_a: MONDO:0005044 ! hypertensive disorder

[Term]
id: MONDO:0005249
name: pneumonia
xref: ICD9:486 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0005607
name: chronic bronchitis
xref: ICD9:491 {source="MONDO:relatedTo"}

[Term]
id: MONDO:0000002
name: kidney stone

[Term]
id: MONDO:0000001
name: stone kidney

[Term]
id: MONDO:0021187
name: familial hypercholesterolemia

[Term]
id: MONDO:0000003
name: gallstone
is_obsolete: true
"""

HPO = """\
format-version: 1.2
data-version: hp/releases/2026-09-01
ontology: hp

[Term]
id: HP:0000822
name: Hypertension

[Term]
id: HP:0003074
name: Hyperglycemia
"""

PNEUMONIA = code_number("486")
BRONCHITIS = code_number("491")
DIABETES_CODE = code_number("250")


def _range(field: str, unit: str, low: float, high: float) -> RangeConstraint:
    return RangeConstraint(field=field, unit=unit, low=low, high=high, source="toy", limits="toy")


FRAME = FrameSpec(
    name="toy-codebook",
    features=("age", "sex", "diabetes", "htn", "glucose", "chol", "stone", "gallstone", "diag"),
    numeric=("age",),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", "diabetes", "htn", "glucose", "chol", "stone", "gallstone", "diag"),
    codebook={
        "sex": SEXES,
        "diabetes": {0: "absent", 1: "present"},
        "htn": {0: "no", 1: "hypertension"},
        "glucose": {0: "normal", 1: "high glucose", 2: "Hyperglycemia!"},
        "chol": {0: "none", 1: "familial hypercholesterolemia type"},
        "stone": {0: "not recorded", 1: "Kidney  stone"},
        "gallstone": {1: "gallstone"},
        "diag": {PNEUMONIA: "486", BRONCHITIS: "491", DIABETES_CODE: "250"},
    },
    descriptions={
        "age": "age in years",
        "sex": "sex",
        "diabetes": "diabetes mellitus",
        "htn": "blood pressure",
        "glucose": "glucose",
        "chol": "cholesterol disorder",
        "stone": "stones",
        "gallstone": "gallstones",
        "diag": "ICD-9 code of the primary diagnosis",
        "target": "toy readmission",
    },
    ranges=(_range("age", "a", 20, 90),),
    label_codes=(0, 1),
    label_meanings={0: "not readmitted", 1: "readmitted"},
    edges=(45, 70),
    study="toy readmission",
    cohort="toy patient",
)
"""A codebook exercising every route: flags, levels, a tie, an obsolete term and ICD-9 codes."""


def automatic_ontologies(directory: Path) -> dict[str, Ontology]:
    """`MONDO` and `HPO` parsed from files written under `directory`, keyed by prefix."""
    return {
        "MONDO": parse_obo(write_obo(directory, MONDO, "mondo.obo"), name="mondo"),
        "HP": parse_obo(write_obo(directory, HPO, "hp.obo"), name="hpo"),
    }
