"""The scene modules against `video.captions`, read as source since the gate has no ManimGL.

`video/main.py` raises at import when its order differs from `captions.ORDER`, but only `make video`
imports it. These cases read the same facts from the syntax tree, so a drifted order or a beat with
no caption fails the gate instead of the render.
"""

from __future__ import annotations

import ast
import re

import pytest

from tests._corpus import REPO_ROOT
from video import captions


pytestmark = pytest.mark.filterwarnings("error")

VIDEO = REPO_ROOT / "video"
BEAT = re.compile(r"_beat_(\d+)")


def _scenes() -> dict[str, tuple[str, list[int]]]:
    """Each scene class with a `name`: that name and the numbers of its `_beat_<n>` methods."""
    found: dict[str, tuple[str, list[int]]] = {}
    for path in sorted((VIDEO / "sections").glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef):
                continue
            names = [
                statement.value.value
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and [getattr(t, "id", None) for t in statement.targets] == ["name"]
                and isinstance(statement.value, ast.Constant)
            ]
            beats = [
                int(match.group(1))
                for statement in node.body
                if isinstance(statement, ast.FunctionDef)
                and (match := BEAT.fullmatch(statement.name))
            ]
            if names:
                found[node.name] = (str(names[0]), beats)
    return found


def test_each_scene_class_is_named_for_its_section():
    assert {cls: name for cls, (name, _) in _scenes().items()} == {n: n for n in captions.ORDER}


def test_the_render_order_is_the_captions_order():
    tree = ast.parse((VIDEO / "main.py").read_text(encoding="utf-8"))
    (listed,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "SCENES_IN_ORDER"
    ]
    assert isinstance(listed, ast.Tuple)
    assert tuple(getattr(element, "id", None) for element in listed.elts) == captions.ORDER


@pytest.mark.parametrize("name", captions.ORDER)
def test_each_beat_has_a_caption(name: str):
    _, beats = _scenes()[name]
    assert all(1 <= beat <= len(captions.section(name).captions) for beat in beats)
