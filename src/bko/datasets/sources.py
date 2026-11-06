"""The pinned downloads: where each comes from and the exact bytes it must be.

A download is trusted only once its size and sha256 match the values here. This module is their one
home (ADR `tooling/one-home-per-kind-of-statement`), so a change of upstream bytes fails closed
until someone re-pins it on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Agreement:
    """The terms a dataset is held under, when it is held under a data use agreement.

    Attributes:
        name: The agreement's name.
        sending: Where it permits sending records to a third-party API (a clause or a written
            permission); None when it grants no such permission.
    """

    name: str
    sending: str | None = None


def permits_sending(agreement: Agreement | None) -> bool:
    """Whether records held under `agreement` may go to a third-party API.

    Open data may; data under an agreement may only when its `sending` names a permission that is
    not blank.
    """
    return agreement is None or bool((agreement.sending or "").strip())


@dataclass(frozen=True, slots=True)
class Source:
    """One pinned download.

    Attributes:
        name: The directory under the data root that holds this source's snapshots.
        url: The HTTPS address it is fetched from.
        filename: The file name inside a snapshot directory.
        size: The exact byte count.
        sha256: The lowercase hex sha256 of the bytes.
        agreement: The data use agreement it is held under; None for open data.
    """

    name: str
    url: str
    filename: str
    size: int
    sha256: str
    agreement: Agreement | None


UCI_HEART_DISEASE: Final = Source(
    name="uci-heart-disease",
    url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    filename="heart-disease.zip",
    size=128_894,
    sha256="b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc",
    agreement=None,
)
"""UCI Heart Disease (CC BY 4.0); `bko.datasets.uci` reads its Cleveland file."""

DIABETES_130: Final = Source(
    name="uci-diabetes-130",
    url=(
        "https://archive.ics.uci.edu/static/public/296/"
        "diabetes+130-us+hospitals+for+years+1999-2008.zip"
    ),
    filename="diabetes-130-us-hospitals-for-years-1999-2008.zip",
    size=3_170_254,
    sha256="f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a",
    agreement=None,
)
"""UCI Diabetes 130-US Hospitals for Years 1999-2008 (doi:10.24432/C5230J).

The archive holds `diabetic_data.csv` and `IDS_mapping.csv`; size and sha256 were measured from
the bytes UCI serves. Its terms are CC BY 4.0 alone, so it is open data. The UCI page says: "This
dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license."
Strack et al. 2014 say of the records: "All data were deidentified in compliance with the Health
Insurance Portability and Accountability Act of 1996 before being provided to the investigators."
"""

SYNTHEA_JAR: Final = Source(
    name="synthea-jar",
    url=(
        "https://github.com/synthetichealth/synthea/releases/download/v4.0.0/"
        "synthea-with-dependencies.jar"
    ),
    filename="synthea-with-dependencies.jar",
    size=201_164_144,
    sha256="ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1",
    agreement=None,
)
"""Synthea v4.0.0 (Apache-2.0); size and sha256 are GitHub's release asset metadata."""

MONDO: Final = Source(
    name="mondo",
    url="https://github.com/monarch-initiative/mondo/releases/download/v2026-10-06/mondo.obo",
    filename="mondo.obo",
    size=53_320_617,
    sha256="2631e8e165d41def1a46749adb1ce0a336c8aba7f6bc0978729592cee8fed60b",
    agreement=None,
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
    agreement=None,
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
    agreement=None,
)
"""Human Disease Ontology release 2026-09-30 (CC0 1.0); `bko.ontology.santos` reads it.

The file is the release tag's `src/ontology/doid.obo`: DO's GitHub releases carry no assets, so
size and sha256 were measured from the bytes at the tag.
"""

MONDO_OWL: Final = Source(
    name="mondo-owl",
    url="https://github.com/monarch-initiative/mondo/releases/download/v2026-10-06/mondo.owl",
    filename="mondo.owl",
    size=254_759_953,
    sha256="cbf3e02cc0aefed4fa6925162a5b1d1a48fbf0418586341967d5fd68fd7185d1",
    agreement=None,
)
"""`MONDO`'s release as OWL (CC BY 4.0), which `bko.ontology.elk` classifies with ELK.

Size and sha256 are GitHub's release asset metadata.
"""

HPO_OWL: Final = Source(
    name="hpo-owl",
    url=(
        "https://github.com/obophenotype/human-phenotype-ontology/releases/download/"
        "v2026-09-01/hp.owl"
    ),
    filename="hp.owl",
    size=76_854_086,
    sha256="8da35ec3b4cb494369b249acb9c2b3874a5a5839a91d5fae74b0d215b355138b",
    agreement=None,
)
"""`HPO`'s release as OWL, under the HPO license (its attribution is `HPO`'s), which
`bko.ontology.elk` classifies with ELK.

Size and sha256 are GitHub's release asset metadata.
"""

ROBOT_JAR: Final = Source(
    name="robot-jar",
    url="https://github.com/ontodev/robot/releases/download/v1.9.11/robot.jar",
    filename="robot.jar",
    size=82_708_624,
    sha256="53cd275c8e4c0dad1fc2c1c34174876795cd7d6b6926e1444adcbb0cb3c7ab26",
    agreement=None,
)
"""ROBOT v1.9.11 (BSD-3-Clause), which runs ELK for `bko.ontology.elk`; size and sha256 are
GitHub's release asset metadata."""
