"""Every pinned download is shaped so `bko.datasets.fetch` can trust it."""

from __future__ import annotations

import re

import pytest

from bko.datasets import sources
from bko.datasets.sources import DO, HPO, HPO_OWL, MONDO, MONDO_OWL, ROBOT_JAR, Source


PINS = tuple(value for value in vars(sources).values() if isinstance(value, Source))


def test_the_scan_finds_every_pin() -> None:
    assert {pin.name for pin in PINS} >= {
        "uci-heart-disease",
        "uci-diabetes-130",
        "synthea-jar",
        "mondo",
        "hpo",
        "do",
        "mondo-owl",
        "hpo-owl",
        "robot-jar",
    }


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


@pytest.mark.parametrize(
    "pin", [MONDO, HPO, MONDO_OWL, HPO_OWL, ROBOT_JAR], ids=lambda pin: pin.name
)
def test_ontology_pins_name_a_tagged_release(pin: Source) -> None:
    assert "/releases/download/v" in pin.url
    assert "/latest/" not in pin.url


def test_the_do_pin_names_a_tag() -> None:
    assert re.search(r"/v\d{4}-\d{2}-\d{2}/", DO.url)
    assert "/main/" not in DO.url
    assert "/master/" not in DO.url


@pytest.mark.parametrize(
    ("pin", "url", "size", "sha256"),
    [
        (
            MONDO_OWL,
            "https://github.com/monarch-initiative/mondo/releases/download/v2026-10-06/mondo.owl",
            254_759_953,
            "cbf3e02cc0aefed4fa6925162a5b1d1a48fbf0418586341967d5fd68fd7185d1",
        ),
        (
            HPO_OWL,
            (
                "https://github.com/obophenotype/human-phenotype-ontology/releases/download/"
                "v2026-09-01/hp.owl"
            ),
            76_854_086,
            "8da35ec3b4cb494369b249acb9c2b3874a5a5839a91d5fae74b0d215b355138b",
        ),
        (
            ROBOT_JAR,
            "https://github.com/ontodev/robot/releases/download/v1.9.11/robot.jar",
            82_708_624,
            "53cd275c8e4c0dad1fc2c1c34174876795cd7d6b6926e1444adcbb0cb3c7ab26",
        ),
    ],
    ids=lambda value: value.name if isinstance(value, Source) else None,
)
def test_the_elk_pins_are_githubs_release_asset_metadata(
    pin: Source, url: str, size: int, sha256: str
) -> None:
    assert (pin.url, pin.size, pin.sha256, pin.agreement) == (url, size, sha256, None)
    assert pin.filename == url.rpartition("/")[2]


def test_each_owl_pin_is_its_obo_pins_release() -> None:
    assert MONDO_OWL.url.rpartition("/")[0] == MONDO.url.rpartition("/")[0]
    assert HPO_OWL.url.rpartition("/")[0] == HPO.url.rpartition("/")[0]
