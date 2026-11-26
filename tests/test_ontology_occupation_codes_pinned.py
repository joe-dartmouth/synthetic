"""The committed occupation rows against the pinned OccO release and ACS archive.

`tests/test_ontology_occupation_codes.py` checks the generator against stanzas built in
`tmp_path`, so a new release or a changed generator passes `make check`. These cases regenerate
both parts from the release, converted beside its OWL in the data root, and the archive's kept
rows, and are skipped unless `--releases` is passed, since the gate reads no `data/`. They read
and never fetch: an absent snapshot or converted file fails them rather than skipping them.
`docs/DEVELOPMENT.md` (Ontologies) says when to run them.
"""

from __future__ import annotations

import functools
from collections import Counter
from functools import partial
from pathlib import Path
from typing import BinaryIO

import pytest

from bko.datasets.acs import occupation_pairs, read_kept
from bko.datasets.fetch import default_data_dir
from bko.datasets.sources import ACS_MA_2024, OCCO
from bko.ontology import occupation_codes_high, occupation_codes_low
from bko.ontology.convert import converted
from bko.ontology.occupations import generate, part, render
from bko.ontology.releases import load_releases
from bko.ontology.schema import OccupationCode


pytestmark = [pytest.mark.releases, pytest.mark.integration]

UNMAPPED = Counter({"no class": 1, "several classes": 0, "several SOC ids": 0})
"""The unmapped tallies ADR `research/acs-frame` states for the pinned release and archive."""


def _refusing(url: str, timeout: float) -> BinaryIO:
    msg = f"a release test fetched {url} ({timeout} s)"
    raise AssertionError(msg)


def _no_java() -> tuple[Path, str]:
    msg = "a release test ran Java: the converted OccO is absent from the data root"
    raise AssertionError(msg)


@functools.cache
def _generated() -> tuple[tuple[OccupationCode, ...], Counter[str]]:
    data_dir = default_data_dir()
    convert = partial(converted, resolve_java=_no_java)
    occo = load_releases({"OCCO": OCCO}, data_dir, open_url=_refusing, convert=convert)["OCCO"]
    rows, _ = read_kept(data_dir, ACS_MA_2024)
    generated, unmapped = generate(occupation_pairs(rows), occo)
    return tuple(generated), unmapped


@pytest.mark.parametrize(
    "module", [occupation_codes_low, occupation_codes_high], ids=["low", "high"]
)
def test_the_rendered_modules_are_the_committed_ones(module) -> None:
    which = "low" if module is occupation_codes_low else "high"
    rows, _ = _generated()
    text = render([row for row in rows if part(row) == which], which)
    assert Path(module.__file__).read_text(encoding="utf-8") == text


def test_the_unmapped_tallies_are_the_adrs() -> None:
    _, unmapped = _generated()
    assert unmapped == UNMAPPED
