"""Every pinned download is shaped so `bko.datasets.fetch` can trust it."""

from __future__ import annotations

import re

import pytest

from bko.datasets import sources
from bko.datasets.sources import DO, HPO, MONDO, Source


PINS = tuple(value for value in vars(sources).values() if isinstance(value, Source))


def test_the_scan_finds_every_pin() -> None:
    assert {pin.name for pin in PINS} >= {"uci-heart-disease", "synthea-jar", "mondo", "hpo", "do"}


@pytest.mark.parametrize("pin", PINS, ids=lambda pin: pin.name)
def test_pin_shape(pin: Source) -> None:
    assert pin.url.startswith("https://")
    assert re.fullmatch(r"[0-9a-f]{64}", pin.sha256)
    assert pin.size > 0
    assert "/" not in pin.filename
    assert ".." not in pin.filename


def test_pins_are_distinct() -> None:
    assert len({pin.name for pin in PINS}) == len(PINS)
    assert len({pin.sha256[:16] for pin in PINS}) == len(PINS)


@pytest.mark.parametrize("pin", [MONDO, HPO], ids=lambda pin: pin.name)
def test_ontology_pins_name_a_tagged_release(pin: Source) -> None:
    assert "/releases/download/v" in pin.url
    assert "/latest/" not in pin.url


def test_the_do_pin_names_a_tag() -> None:
    assert re.search(r"/v\d{4}-\d{2}-\d{2}/", DO.url)
    assert "/main/" not in DO.url
    assert "/master/" not in DO.url
