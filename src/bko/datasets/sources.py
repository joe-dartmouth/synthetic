"""The pinned downloads: where each comes from and the exact bytes it must be.

A download is trusted only once its size and sha256 match the values here. This module is their one
home (ADR `tooling/one-home-per-kind-of-statement`), so a change of upstream bytes fails closed
until someone re-pins it on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Source:
    """One pinned download.

    Attributes:
        name: The directory under the data root that holds this source's snapshots.
        url: The HTTPS address it is fetched from.
        filename: The file name inside a snapshot directory.
        size: The exact byte count.
        sha256: The lowercase hex sha256 of the bytes.
    """

    name: str
    url: str
    filename: str
    size: int
    sha256: str


UCI_HEART_DISEASE: Final = Source(
    name="uci-heart-disease",
    url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    filename="heart-disease.zip",
    size=128_894,
    sha256="b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc",
)
"""UCI Heart Disease (CC BY 4.0); `bko.datasets.uci` reads its Cleveland file."""

SYNTHEA_JAR: Final = Source(
    name="synthea-jar",
    url=(
        "https://github.com/synthetichealth/synthea/releases/download/v4.0.0/"
        "synthea-with-dependencies.jar"
    ),
    filename="synthea-with-dependencies.jar",
    size=201_164_144,
    sha256="ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1",
)
"""Synthea v4.0.0 (Apache-2.0); size and sha256 are GitHub's release asset metadata."""

MONDO: Final = Source(
    name="mondo",
    url="https://github.com/monarch-initiative/mondo/releases/download/v2026-10-06/mondo.obo",
    filename="mondo.obo",
    size=53_320_617,
    sha256="2631e8e165d41def1a46749adb1ce0a336c8aba7f6bc0978729592cee8fed60b",
)
"""MONDO release 2026-10-06 (CC BY 4.0); `bko.ontology.obo` reads it.

Size and sha256 are GitHub's release asset metadata.
"""

HPO: Final = Source(
    name="hpo",
    url=(
        "https://github.com/obophenotype/human-phenotype-ontology/releases/download/"
        "v2026-09-01/hp.obo"
    ),
    filename="hp.obo",
    size=10_863_613,
    sha256="93dace952fcb3ec4728818857f6ba76bc2d5312f4d83266519b8694b8e798f22",
)
"""HPO release 2026-09-01, under the HPO license; `bko.ontology.obo` reads it.

Size and sha256 are GitHub's release asset metadata. The attribution the license asks for: this
service/product uses the Human Phenotype Ontology (version 2026-09-01). Find out more at
http://www.human-phenotype-ontology.org
"""

DO: Final = Source(
    name="do",
    url=(
        "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/"
        "v2026-09-30/src/ontology/doid.obo"
    ),
    filename="doid.obo",
    size=7_264_205,
    sha256="51e717b6b9f5d391f2793f9e16faf9c8629a661ab8f21c3a024458129f182ad2",
)
"""Human Disease Ontology release 2026-09-30 (CC0 1.0); `bko.ontology.santos` reads it.

The file is the release tag's `src/ontology/doid.obo`: DO's GitHub releases carry no assets, so
size and sha256 were measured from the bytes at the tag.
"""
