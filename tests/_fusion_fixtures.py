"""Archives, training splits and candidates files for `bko.fusion`'s tests."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.fetch import ensure
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.gaps.observe import Cell
from bko.generate.decode import Undecoded
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ladder_releases
from tests._generate_fixtures import SIZES, uci
from tests._ontology_fixtures import RoutingOpener
from tests._validate_fixtures import planted


UCI_X: Final = uci({(40, 0): "4.0"}, "1.0")
"""Angina everywhere but in young women: per-subgroup gaps in the two young-women cells only."""
UCI_Y: Final = uci({}, "4.0")
"""No angina anywhere: per-subgroup gaps in every cell, and population-level gaps."""
UCI_FULL: Final = uci({}, "1.0", {**SIZES, (40, 0, 1): 25})
"""Angina in every cell and records of both labels in every cell: no per-subgroup gap."""
CELL0: Final[Cell] = ("<55", "female", 0)
CELL1: Final[Cell] = ("<55", "female", 1)


def training_split(data_dir: Path, data: bytes) -> pd.DataFrame:
    """The default training split of the archive `data`, fetched into `data_dir`."""
    source = pin(data)
    ensure(source, data_dir, open_url=RoutingOpener({source.url: data}))
    return split(load_uci(data_dir, source=source), edges=UCI_FRAME.edges).train


def young(target: int, **changes: object) -> dict[str, object]:
    """A valid candidate aged 40, female, with `target`."""
    return planted(age=40, sex=0, target=target, **changes)


def candidate_lines(calls: Iterable[tuple[str, Sequence[object]]]) -> bytes:
    """A candidates file written independently of `bko.generate.candidates`, the tests' oracle."""
    lines: list[str] = []
    for index, (key, items) in enumerate(calls, start=1):
        for position, item in enumerate(items):
            line: dict[str, object] = {"call": index, "cell": key, "position": position}
            if isinstance(item, Undecoded):
                line["undecoded"] = item.reason.value
            else:
                line["candidate"] = item
            lines.append(json.dumps(line, sort_keys=True) + "\n")
    return "".join(lines).encode()


def scenario_a() -> bytes:
    """Cell 0 asked twenty records without angina, cell 1 twenty with it."""
    return candidate_lines(
        [("<55/female/0", [young(0, cp=4)] * 20), ("<55/female/1", [young(1, cp=1)] * 20)]
    )


def fusion_header(
    archive: bytes,
    candidates: bytes,
    *,
    plan: Sequence[tuple[str, int]] | None = None,
    **fields: object,
) -> bytes:
    """`candidates` behind the header generate would write for `archive` and the ladder releases.

    The plan defaults to each call's cell and line count, read from `candidates`. `fields`
    overrides any header field. The header is written here, not by `render_header`, so it stays
    the tests' oracle.
    """
    source = pin(archive)
    with tempfile.TemporaryDirectory() as scratch:
        ensure(source, Path(scratch), open_url=RoutingOpener({source.url: archive}))
        fingerprint = split(
            load_uci(Path(scratch), source=source), edges=UCI_FRAME.edges
        ).fingerprint
    if plan is None:
        cells: dict[int, str] = {}
        counts: Counter[int] = Counter()
        for raw in candidates.decode("utf-8").splitlines():
            line = json.loads(raw)
            cells[line["call"]] = line["cell"]
            counts[line["call"]] += 1
        plan = [(cells[call], counts[call]) for call in sorted(cells)]
    pins, _ = ladder_releases()
    header: dict[str, object] = {
        "kind": "header",
        "schema": 1,
        "request": "completeness",
        "definition": "per-subgroup",
        "guidance": "ontology",
        "uci": source.sha256,
        "fingerprint": fingerprint,
        "releases": {prefix: pins[prefix].sha256 for prefix in ("HP", "MONDO")},
        "calls": [list(call) for call in plan],
    } | fields
    return (json.dumps(header, sort_keys=True) + "\n").encode() + candidates
