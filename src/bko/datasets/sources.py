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
