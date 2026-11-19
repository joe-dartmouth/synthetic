"""`video.captions`: the demo video's text and length, and the facts it takes from the docs."""

from __future__ import annotations

import re

import pytest

from bko.gaps.contradictions import KINDS
from tests._corpus import REPO_ROOT
from video import captions


pytestmark = pytest.mark.filterwarnings("error")

TEXT = [text for section in captions.SECTIONS for beat in section.beats for text in beat]


def test_the_planned_length_is_inside_the_bound():
    """3 to 7 min 45 s: the 3-8 minute brief, less a margin for beats that overrun their hold."""
    assert 180 <= captions.planned_seconds() <= 465


def test_every_beat_has_a_label_and_an_explainer():
    assert all(label.strip() and note.strip() for s in captions.SECTIONS for label, note in s.beats)


def test_no_beat_states_an_experiment_result():
    """`docs/EVIDENCE.md` gives results to four decimals; the video explains and reports none."""
    assert [text for text in TEXT if re.search(r"\d\.\d{4}", text)] == []


@pytest.mark.parametrize(("stated", "doc", "sentence"), captions.SOURCED)
def test_each_sourced_fact_is_on_screen_and_in_its_doc(stated: str, doc: str, sentence: str):
    assert any(stated in text for text in TEXT)
    assert sentence in (REPO_ROOT / doc).read_text(encoding="utf-8")


def test_the_kinds_are_the_census_kinds_in_order():
    assert tuple(name for name, _ in captions.KINDS) == KINDS
