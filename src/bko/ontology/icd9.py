"""The Diabetes 130 rows of the mapping table: ICD-9 diagnosis categories and lab states.

ADR `research/ontology-mapping-design` records how the rows are chosen. An ICD-9 row maps a
diagnosis category to a live MONDO concept carrying `xref: ICD9:<code>`, whatever the xref's
qualifier, and declares every live concept that carries it; `bko.ontology.check` refuses a row whose
concept lacks the xref or whose declared holders differ from the release's. Where a category has
two holders, the row maps the broader one, which spans the category's subcodes. `UNMAPPED_ICD9`
records a category the table leaves out because no live concept carries its xref, and the check
refuses one that gains a holder. A lab row maps the archive's result states for a field to an HPO
phenotype, with the archive's state strings verbatim.

Sources and labels follow `bko.ontology.mapping`'s vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Icd9Mapping:
    """An ICD-9 diagnosis category and the MONDO concept that carries its xref.

    Attributes:
        codes: The ICD-9 categories a diagnosis token falls in for the row to hold, matched by
            category rather than by string prefix.
        xref: The `ICD9:` xref id the concept carries: `ICD9:` and the first of `codes`.
        concept: The MONDO curie.
        label: The concept's name in its pinned release, verbatim.
        holders: Every live concept in the pinned release that carries `xref`, `concept` among
            them.
        source: The release that asserts the xref.
        limits: Where the mapping stops being true.
    """

    codes: tuple[str, ...]
    xref: str
    concept: str
    label: str
    holders: frozenset[str]
    source: str
    limits: str

    def __post_init__(self) -> None:
        """Tie the xref to the row's first category.

        Raises:
            ValueError: If there is no category or `xref` is not `ICD9:` and the first one.
        """
        if not self.codes:
            msg = f"{self.label}: no ICD-9 categories"
            raise ValueError(msg)
        expected = f"ICD9:{self.codes[0]}"
        if self.xref != expected:
            msg = f"{self.label}: xref {self.xref} is not {expected}, its first category's"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class UnmappedIcd9:
    """An ICD-9 diagnosis category no live MONDO concept carries as an xref.

    Attributes:
        xref: The `ICD9:` id no live concept carries.
        label: The category's ICD-9-CM title.
        reason: Why no concept maps it.
        source: What supports the reason.
    """

    xref: str
    label: str
    reason: str
    source: str


@dataclass(frozen=True, slots=True)
class LabMapping:
    """Result states of a Diabetes 130 lab field that show an HPO phenotype.

    Attributes:
        field: The archive's column name.
        states: The archive's result strings that show the concept, verbatim.
        concept: The HP curie.
        label: The concept's name in its pinned release, verbatim.
        source: What establishes the mapping.
        limits: Where the mapping stops being true.
    """

    field: str
    states: frozenset[str]
    concept: str
    label: str
    source: str
    limits: str


ICD9_MAPPINGS: Final[tuple[Icd9Mapping, ...]] = (
    Icd9Mapping(
        codes=("250.4",),
        xref="ICD9:250.4",
        concept="MONDO:0005016",
        label="diabetic kidney disease",
        holders=frozenset({"MONDO:0005016"}),
        source="mondo",
        limits="diabetes with renal manifestations; the diagnosis does not stage the disease",
    ),
    Icd9Mapping(
        codes=("401",),
        xref="ICD9:401",
        concept="MONDO:0001134",
        label="essential hypertension",
        holders=frozenset({"MONDO:0001134"}),
        source="mondo",
        limits="a coded diagnosis on the encounter, not a measured blood pressure",
    ),
    Icd9Mapping(
        codes=("410",),
        xref="ICD9:410",
        concept="MONDO:0004781",
        label="acute myocardial infarction",
        holders=frozenset({"MONDO:0004781"}),
        source="mondo",
        limits="the category holds every infarct site and episode of care",
    ),
    Icd9Mapping(
        codes=("428",),
        xref="ICD9:428",
        concept="MONDO:0005252",
        label="heart failure",
        holders=frozenset({"MONDO:0005009", "MONDO:0005252"}),
        source="mondo",
        limits=(
            "the release also puts the xref on congestive heart failure (MONDO:0005009); the "
            "category spans systolic, diastolic and unspecified failure, so the broader concept "
            "is used"
        ),
    ),
    Icd9Mapping(
        codes=("434",),
        xref="ICD9:434",
        concept="MONDO:0000944",
        label="cerebral artery occlusion",
        holders=frozenset({"MONDO:0000944"}),
        source="mondo",
        limits="the category holds thrombosis and embolism, with or without infarction",
    ),
    Icd9Mapping(
        codes=("486",),
        xref="ICD9:486",
        concept="MONDO:0005249",
        label="pneumonia",
        holders=frozenset({"MONDO:0005249"}),
        source="mondo",
        limits="the category is pneumonia with the organism unspecified",
    ),
    Icd9Mapping(
        codes=("491",),
        xref="ICD9:491",
        concept="MONDO:0005607",
        label="chronic bronchitis",
        holders=frozenset({"MONDO:0003781", "MONDO:0005607"}),
        source="mondo",
        limits=(
            "the release also puts the xref on bronchitis (MONDO:0003781); the category is "
            "chronic bronchitis, so the chronic concept is used"
        ),
    ),
    Icd9Mapping(
        codes=("496",),
        xref="ICD9:496",
        concept="MONDO:0005002",
        label="chronic obstructive pulmonary disease",
        holders=frozenset({"MONDO:0005002"}),
        source="mondo",
        limits="the category is chronic airway obstruction not elsewhere classified",
    ),
    Icd9Mapping(
        codes=("585",),
        xref="ICD9:585",
        concept="MONDO:0005300",
        label="chronic kidney disease",
        holders=frozenset({"MONDO:0005300"}),
        source="mondo",
        limits="the category holds every stage; the frame keeps no stage",
    ),
    Icd9Mapping(
        codes=("707",),
        xref="ICD9:707",
        concept="MONDO:0004605",
        label="chronic ulcer of skin",
        holders=frozenset({"MONDO:0004605"}),
        source="mondo",
        limits="the category holds pressure and lower-limb ulcers, diabetic foot ulcers among them",
    ),
)
"""Diagnosis categories, each with the MONDO concept that carries its xref and every live holder."""

UNMAPPED_ICD9: Final[tuple[UnmappedIcd9, ...]] = (
    UnmappedIcd9(
        xref="ICD9:414",
        label="Other forms of chronic ischemic heart disease",
        reason=(
            "no live concept in the pinned MONDO carries xref ICD9:414; its subcodes 414.0 and "
            "414.00 sit on coronary artery disorder and coronary atherosclerosis, neither of "
            "which spans the category"
        ),
        source="mondo",
    ),
)
"""Diagnosis categories the table leaves out because no live MONDO concept carries their xref."""

LAB_MAPPINGS: Final[tuple[LabMapping, ...]] = (
    LabMapping(
        field="A1Cresult",
        states=frozenset({">7", ">8"}),
        concept="HP:0040217",
        label="Elevated hemoglobin A1c",
        source="strack-2014",
        limits=(
            "one HbA1c result above 7% during the encounter; None means not measured and asserts "
            "nothing"
        ),
    ),
    LabMapping(
        field="max_glu_serum",
        states=frozenset({">200", ">300"}),
        concept="HP:0003074",
        label="Hyperglycemia",
        source="ada-2026",
        limits=(
            "one serum glucose result above 200 mg/dl during the encounter, fasting state "
            "unknown; None means not measured and asserts nothing"
        ),
    ),
)
"""Lab result states that show an HPO phenotype, with the archive's state strings verbatim."""
