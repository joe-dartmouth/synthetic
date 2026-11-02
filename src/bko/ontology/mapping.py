"""The concept-to-field mapping table: the one home of its rows.

ADR `tooling/one-home-per-kind-of-statement`: every other document points here rather than
copying a row. ADR `research/ontology-mapping-design` records how the rows are chosen.

The numeric fields' range constraints belong to the frame, so `RANGES` is defined in
`bko.datasets.uci` and re-exported here.

**Sources.** A row's `source` is the `name` of a `Source` in `bko.datasets.sources` (`mondo`,
`hpo`, or `uci-heart-disease`, which means `heart-disease.names` inside that archive), a key in
`docs/literature.md`, or an ADR id under `docs/adr/`.

**Scope.** A concept is in scope when a row maps it or it is an `is_a` ancestor of a mapped concept
in its release, reached through classes with the release's prefix. Every other concept is out of
scope; `OUT_OF_SCOPE` records the near misses considered and why each was rejected, and
`bko.ontology.check` refuses one that is an `is_a` ancestor of a mapped concept. Each dataset's BKO
holds the part of it the dataset's own rows reach: `FIELD_MAPPINGS` for UCI Heart Disease,
`CODE_MAPPINGS` for Synthea (`bko.gaps`); the rows of `bko.ontology.icd9` map Diabetes 130's
diagnosis categories and lab states. A Synthea disorder code maps to the one live MONDO concept
carrying its `skos:exactMatch`; `UNMAPPED_CODES` records the named conditions' disorder codes none
carries, and `bko.ontology.check` refuses one that gains a live match.

**Labels** are the releases' `name:` values, copied verbatim from MONDO (CC BY 4.0) and HPO (its
attribution is in `bko.datasets.sources.HPO`).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from bko.datasets.cells import LABEL
from bko.datasets.sources import HPO, MONDO, Source
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE
from bko.datasets.uci import RANGES  # ruff: ignore[unused-import]  (re-exported)
from bko.ontology.icd9 import ICD9_MAPPINGS, LAB_MAPPINGS, UNMAPPED_ICD9
from bko.ontology.schema import (
    AtLeast,
    CheckedRow,
    CodeMapping,
    Codes,
    ConceptRow,
    FieldMapping,
    OutOfScope,
    Unmapped,
    UnmappedCode,
)


RELEASES: Final[Mapping[str, Source]] = MappingProxyType({"HP": HPO, "MONDO": MONDO})
"""The pinned release for each concept prefix."""

FIELD_MAPPINGS: Final[tuple[FieldMapping, ...]] = (
    FieldMapping(
        when=(Codes(LABEL, frozenset({1})),),
        concept="MONDO:0006715",
        label="coronary stenosis",
        source="uci-heart-disease",
        limits=(
            "num 1 to 4 (over 50% diameter narrowing in a major vessel) collapses to target 1; the "
            "concept holds any coronary narrowing; coronary artery disorder (MONDO:0005010) is "
            "reached through is_a"
        ),
    ),
    FieldMapping(
        when=(Codes(LABEL, frozenset({1})),),
        concept="HP:0005145",
        label="Coronary artery stenosis",
        source="uci-heart-disease",
        limits="the HPO phenotype for the same finding, so the HPO fragment holds the label too",
    ),
    FieldMapping(
        when=(Codes("cp", frozenset({1})),),
        concept="HP:0001681",
        label="Angina pectoris",
        source="uci-heart-disease",
        limits=(
            "HPO defines angina as related to myocardial ischemia; the UCI class is a history "
            "pattern, not a proven ischemic cause"
        ),
    ),
    FieldMapping(
        when=(Codes("cp", frozenset({1, 2, 3})),),
        concept="HP:0100749",
        label="Chest pain",
        source="uci-heart-disease",
        limits=(
            "HPO has no atypical-angina or non-anginal term, so codes 2 and 3 ground only the "
            "broad concept; HPO does not place Angina pectoris under Chest pain"
        ),
    ),
    FieldMapping(
        when=(Codes("exang", frozenset({1})),),
        concept="HP:0001681",
        label="Angina pectoris",
        source="uci-heart-disease",
        limits=(
            "angina during the exercise test; HPO's exertion modifier HP:0025377 is not encoded; "
            "cp 1 reaches the same concept from history"
        ),
    ),
    FieldMapping(
        when=(AtLeast("trestbps", 130),),
        concept="HP:0004421",
        label="Elevated systolic blood pressure",
        source="whelton-2018",
        limits=(
            "one resting reading on admission, not a hypertension diagnosis; the 2017 ACC/AHA "
            "guideline's stage 1 starts at 130 mm Hg systolic, where JNC 7 (chobanian-2003) used "
            "140; that trestbps is systolic is read from its name"
        ),
    ),
    FieldMapping(
        when=(AtLeast("chol", 240),),
        concept="HP:0003124",
        label="Hypercholesterolemia",
        source="ncep-2001",
        limits='ATP III\'s "high" total cholesterol; one reading of unknown fasting state',
    ),
    FieldMapping(
        when=(Codes("fbs", frozenset({1})),),
        concept="HP:0003074",
        label="Hyperglycemia",
        source="ada-2026",
        limits=(
            "one fasting reading above the dataset's 120 mg/dl cut, which lies above the ADA's "
            "normal fasting range; it spans impaired fasting glucose and the diabetes range, so "
            "diabetes is not implied (MONDO:0005015 diabetes mellitus)"
        ),
    ),
    FieldMapping(
        when=(Codes("restecg", frozenset({1, 2})),),
        concept="HP:0003115",
        label="Abnormal EKG",
        source="uci-heart-disease",
        limits=(
            "state 1 is a disjunction (T-wave inversion and/or ST elevation or depression over "
            "0.05 mV) that no single narrower term covers"
        ),
    ),
    FieldMapping(
        when=(Codes("restecg", frozenset({2})),),
        concept="HP:0001712",
        label="Left ventricular hypertrophy",
        source="uci-heart-disease",
        limits=(
            "\"probable or definite\" by Estes' ECG criteria, while HPO's term is anatomical; the "
            "states are exclusive, so code 2 can hide an ST-T abnormality"
        ),
    ),
    FieldMapping(
        when=(AtLeast("oldpeak", 1.0), Codes("slope", frozenset({2, 3}))),
        concept="HP:0012250",
        label="ST segment depression",
        source="gibbons-2002",
        limits=(
            "the guideline's positive exercise test, at least 1 mm horizontal or downsloping; the "
            "measurement point after the J point is undocumented; mm is the team's reading of "
            "oldpeak's unit"
        ),
    ),
    FieldMapping(
        when=(Codes("ca", frozenset({1, 2, 3})),),
        concept="HP:0001717",
        label="Coronary artery calcification",
        source="detrano-1985a",
        limits=(
            "Detrano's Cleveland Clinic protocol counts calcified coronary arteries on "
            "fluoroscopy, and the reading of ca as that count is inferred from it; "
            'heart-disease.names says only "colored by flourosopy"; fluoroscopy misses calcium, '
            "so ca 0 asserts nothing"
        ),
    ),
    FieldMapping(
        when=(Codes("thal", frozenset({6, 7})),),
        concept="HP:0500018",
        label="Abnormal cardiac exercise stress test",
        source="detrano-1985b",
        limits=(
            "Detrano's Cleveland Clinic protocol scores thallium scintigraphy abnormal on a fixed "
            "or reversible perfusion defect, and the reading of thal as that result is inferred "
            "from it; HPO has no perfusion-defect term, so both ground only the broad concept"
        ),
    ),
)
"""Field states that show a concept; a record shows the concept when every clause holds."""

UNMAPPED: Final[tuple[Unmapped, ...]] = (
    Unmapped(
        field=LABEL,
        codes=frozenset({0}),
        reason=(
            "num 0 (under 50% narrowing) does not establish absence: nonobstructive coronary "
            "artery disease (MONDO:0850371) is_a coronary artery disorder"
        ),
        source="mondo",
    ),
    Unmapped(
        field="sex",
        codes=frozenset({0, 1}),
        reason=(
            "demographic; `bko.datasets.cells` stratifies by it; MONDO and HPO hold no sex concept"
        ),
        source="uci-heart-disease",
    ),
    Unmapped(
        field="cp",
        codes=frozenset({4}),
        reason="asymptomatic records an absence; the table records presence only",
        source="uci-heart-disease",
    ),
    Unmapped(
        field="fbs",
        codes=frozenset({0}),
        reason="120 mg/dl or less includes impaired fasting glucose",
        source="ada-2026",
    ),
    Unmapped(
        field="restecg",
        codes=frozenset({0}),
        reason="normal records an absence",
        source="uci-heart-disease",
    ),
    Unmapped(
        field="exang",
        codes=frozenset({0}),
        reason="no exercise-induced angina records an absence",
        source="uci-heart-disease",
    ),
    Unmapped(
        field="slope",
        codes=frozenset({1}),
        reason="upsloping depression is not a positive exercise test",
        source="gibbons-2002",
    ),
    Unmapped(
        field="ca",
        codes=frozenset({0}),
        reason=(
            "fluoroscopy misses coronary calcium, so no visible calcium does not establish its "
            "absence"
        ),
        source="detrano-1985a",
    ),
    Unmapped(
        field="thal",
        codes=frozenset({3}),
        reason="normal records an absence",
        source="uci-heart-disease",
    ),
)
"""Categorical states no concept defensibly matches, with the reason for each."""

OUT_OF_SCOPE: Final[tuple[OutOfScope, ...]] = (
    OutOfScope(
        concept="HP:0000822",
        label="Hypertension",
        reason=(
            'trestbps is one admission reading; HPO\'s "Increased blood pressure" says one '
            "measurement does not imply hypertension"
        ),
        source="hpo",
    ),
    OutOfScope(
        concept="HP:0033992",
        label="Chronotropic incompetence",
        reason=(
            "needs resting heart rate or beta-blocker status, and neither is among the 14 fields"
        ),
        source="uci-heart-disease",
    ),
    OutOfScope(
        concept="HP:0001658",
        label="Myocardial infarction",
        reason="no field among the 14 records infarction",
        source="uci-heart-disease",
    ),
    OutOfScope(
        concept="MONDO:0021661",
        label="coronary atherosclerosis",
        reason="angiographic narrowing does not establish atheroma as its cause",
        source="mondo",
    ),
)
"""Concepts considered and rejected, with why no field grounds each."""

CODE_MAPPINGS: Final[tuple[CodeMapping, ...]] = (
    CodeMapping(
        snomed=ISCHEMIC_HEART_DISEASE,
        concept="MONDO:0024644",
        label="myocardial ischemia",
        source="mondo",
    ),
    CodeMapping(
        snomed="59621000",
        concept="MONDO:0001134",
        label="essential hypertension",
        source="mondo",
    ),
    CodeMapping(
        snomed="44054006",
        concept="MONDO:0005148",
        label="type 2 diabetes mellitus",
        source="mondo",
    ),
    CodeMapping(
        snomed="55822004",
        concept="MONDO:0021187",
        label="hyperlipidemia",
        source="mondo",
    ),
    CodeMapping(
        snomed="22298006",
        concept="MONDO:0005068",
        label="myocardial infarction",
        source="mondo",
    ),
    CodeMapping(
        snomed="401303003",
        concept="MONDO:0041656",
        label="ST-elevation myocardial infarction",
        source="mondo",
    ),
    CodeMapping(
        snomed="230690007",
        concept="MONDO:0005098",
        label="stroke disorder",
        source="mondo",
    ),
)
"""Synthea's SNOMED-CT disorder codes, each with the MONDO concept the release equates with it by
`skos:exactMatch`: the ablation's code first."""

UNMAPPED_CODES: Final[tuple[UnmappedCode, ...]] = (
    UnmappedCode(
        snomed="401314000",
        label="Acute non-ST segment elevation myocardial infarction (disorder)",
        reason="no skos:exactMatch in the pinned MONDO",
        source="mondo",
    ),
)
"""The named conditions' Synthea disorder codes no MONDO concept is equated with, each with its
reason; the population holds other codes with no match, which no row records."""

CONCEPT_ROWS: Final[tuple[ConceptRow, ...]] = (
    *FIELD_MAPPINGS,
    *CODE_MAPPINGS,
    *OUT_OF_SCOPE,
    *ICD9_MAPPINGS,
    *LAB_MAPPINGS,
)
"""Every row that names a concept, in table order, the Diabetes 130 rows of `bko.ontology.icd9`
last."""

CHECKED_ROWS: Final[tuple[CheckedRow, ...]] = (*CONCEPT_ROWS, *UNMAPPED_CODES, *UNMAPPED_ICD9)
"""Every concept row, then every excluded code and ICD-9 category: what `bko.ontology.check`
checks."""

MAPPED: Final = frozenset(
    row.concept for row in (*FIELD_MAPPINGS, *CODE_MAPPINGS, *ICD9_MAPPINGS, *LAB_MAPPINGS)
)
"""The concepts a row maps; with their is_a ancestors, the table's scope.

Each dataset's BKO takes the part its own rows map.
"""
