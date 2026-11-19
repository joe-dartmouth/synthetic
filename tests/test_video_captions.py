"""`video.captions`: the demo video's labels, length and figures, held to `docs/EVIDENCE.md`."""

from __future__ import annotations

import re

import pytest

from bko.gaps.contradictions import KINDS
from tests._corpus import REPO_ROOT
from video import captions


pytestmark = pytest.mark.filterwarnings("error")

EVIDENCE = (REPO_ROOT / "docs" / "EVIDENCE.md").read_text(encoding="utf-8")


def test_the_planned_length_is_inside_the_bound():
    """The video runs 3 to 7 minutes: the 3-8 minute brief, with a minute's margin for beats."""
    assert 180 <= captions.planned_seconds() <= 420


@pytest.mark.parametrize(("arm", "figure"), captions.RECALL_BARS.items())
def test_each_recall_bar_is_its_arms_figure_in_evidence(arm: str, figure: str):
    assert f"`{arm}` {figure} /" in EVIDENCE


@pytest.mark.parametrize("figure", captions.EVIDENCE_FIGURES)
def test_every_evidence_figure_appears_in_evidence(figure: str):
    assert re.search(rf"(?<![\d.]){re.escape(figure)}(?!\d)", EVIDENCE)
    assert any(figure in caption for caption in captions.section("DoesItHelp").captions)


def test_no_caption_states_a_four_decimal_figure_outside_the_lists():
    stated = {
        figure
        for section in captions.SECTIONS
        for caption in section.captions
        for figure in re.findall(r"[+-]?\d\.\d{4}", caption)
    }
    assert stated <= set(captions.EVIDENCE_FIGURES)


@pytest.mark.parametrize(("stated", "doc", "sentence"), captions.SOURCED)
def test_each_sourced_figure_is_in_a_caption_and_in_its_doc(stated: str, doc: str, sentence: str):
    assert any(stated in caption for s in captions.SECTIONS for caption in s.captions)
    assert sentence in (REPO_ROOT / doc).read_text(encoding="utf-8")


def test_the_kinds_are_the_census_kinds_in_order():
    assert tuple(name for name, _ in captions.KINDS) == KINDS
