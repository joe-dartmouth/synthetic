"""Offline inputs for the `bko.gaps` tests: two small releases and mapping rows over them.

The `MONDO:900000n` and `HP:900000n` ids are invented; `BFO:0000016` and the labels reuse real
names, and no stanza copies release content.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.cells import LABEL
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE
from bko.ontology.schema import CodeMapping, Codes, FieldMapping


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
