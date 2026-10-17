"""Offline inputs for the `bko.gaps` tests: two small releases and mapping rows over them.

The `MONDO:900000n` and `HP:900000n` ids are invented; `BFO:0000016` and the labels reuse real
names, and no stanza copies release content.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.ablation import AblationSpec
from bko.datasets.cells import LABEL
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE
from bko.ontology.schema import CodeMapping, Codes, FieldMapping
from tests._datasets_fixtures import condition, patient


LADDER_MONDO: Final = """\
format-version: 1.2
data-version: releases/2026-01-01
ontology: mondo

[Term]
id: BFO:0000016
name: disposition

[Term]
id: MONDO:9000001
name: disease
is_a: BFO:0000016

[Term]
id: MONDO:9000002
name: heart disorder
is_a: MONDO:9000001

[Term]
id: MONDO:9000003
name: myocardial ischemia
is_a: MONDO:9000002

[Term]
id: MONDO:9000004
name: coronary stenosis
is_a: MONDO:9000002

[Term]
id: MONDO:9000005
name: unrelated disorder
is_a: MONDO:9000001
"""
"""Disease above heart disorder above two leaves, a sibling branch, and a root outside the prefix,
as MONDO's BFO roots are."""

LADDER_HPO: Final = """\
format-version: 1.2
data-version: hp/releases/2026-01-01
ontology: hp.obo

[Term]
id: HP:9000001
name: Phenotypic abnormality

[Term]
id: HP:9000002
name: Chest pain
is_a: HP:9000001

[Term]
id: HP:9000003
name: Angina pectoris
is_a: HP:9000002
"""
"""A three-step HPO-shaped chain."""

ANGINA: Final = FieldMapping(
    when=(Codes("cp", frozenset({1})),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="uci-heart-disease",
    limits="test",
)
STENOSIS: Final = FieldMapping(
    when=(Codes(LABEL, frozenset({1})),),
    concept="MONDO:9000004",
    label="coronary stenosis",
    source="uci-heart-disease",
    limits="test",
)
ISCHEMIA: Final = CodeMapping(
    snomed=ISCHEMIC_HEART_DISEASE,
    concept="MONDO:9000003",
    label="myocardial ischemia",
    source="mondo",
)

UNSEEN: Final = CodeMapping(
    snomed="999999999", concept="MONDO:9000004", label="coronary stenosis", source="mondo"
)
"""A code row no patient has, so its concept's statement is a gap before and after a removal."""


CELL: Final = AblationSpec(
    codes=frozenset({ISCHEMIC_HEART_DISEASE}), age_band=">=60", sex=1, edges=(60,)
)
CONDITION: Final = AblationSpec(
    codes=frozenset({ISCHEMIC_HEART_DISEASE}), age_band=None, sex=None, edges=(60,)
)
"""E2's two ablations over one age edge, so the grid has eight cells."""

E2_POPULATION: Final = (
    [
        *(patient(f"m{index:02d}", "1961-07-15", "M") for index in range(20)),
        patient("w01", "1962-01-01", "F"),
        patient("y01", "1990-05-05", "M"),
        *(patient(f"f{index:02d}", "1990-05-05", "F") for index in range(20)),
    ],
    [
        *(condition(f"m{index:02d}", ISCHEMIC_HEART_DISEASE, "2019-06-01") for index in range(20)),
        condition("w01", ISCHEMIC_HEART_DISEASE, "2020-01-01"),
        condition("y01", "22298006", "2021-01-01"),
    ],
)
"""Snapshot 2026-01-01: twenty men aged 64 and a woman aged 64 with ischemic heart disease, a man
aged 35 with a myocardial infarction only, and twenty women aged 35 with no condition, whose cell
holds the threshold throughout, so it is never a gap."""
