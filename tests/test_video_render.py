"""`video.render`: the concat list the driver hands FFmpeg."""

from __future__ import annotations

import pytest

from video.render import concat_lines


pytestmark = pytest.mark.filterwarnings("error")


def test_the_concat_list_names_each_scene_file_in_order():
    assert concat_lines(("A", "B")) == "file 'A.mp4'\nfile 'B.mp4'\n"
