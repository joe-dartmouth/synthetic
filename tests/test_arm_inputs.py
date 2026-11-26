"""Each arm of `ARMS` reads its declared inputs and no other.

`tests._arm_inputs.ARM_INPUTS` declares, per arm, which of seven inputs its fill reads (ADR
`research/matched-knowledge-controls`, Decision 14). The arms of `ARMS` are filled under each
perturbation below; every other arm of the table is held to the ADR's prose table alone. Each
perturbation changes one input's statistics and leaves the other six as they were:

- `cell`: every `d` of 0 in the thin cell's rows becomes 9.
- `label-group`: every `d` of 0 in the rest of the label's rows becomes 9. Those rows include the
  emptied cell's band, so a banded method reads them, and the emptied cell makes an own-first
  draw fall back to them.
- `other-label`: every `d` of 0 in the other label's rows becomes 9.
- `mapping`: one row of the two-field group `b`-`c` dropped, either one: `JOINT`, which frees `b`
  and so shrinks what the groups hold, or `C3`, which keeps the group's fields. An input is read
  when either changes the fill: `bko-pool-random` sees only the first, which changes the count of
  grouped fields, and `bko-kb` refuses the first, whose freed `b` K gives no value, and reads the
  second. A refusal counts as a changed fill.
- `hierarchy`: `NARROW` and `JOINT` swap parents.
- `codebook`: `c`'s code 0, which no record holds and no row names, dropped.

`d` is a measured field no row names, so a recode moves no concept, code or age. The age range
is held apart from the six: `test_kb_age_follows_the_age_range_alone` reads it.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.methods import augment
from bko.baselines.request import Method, Request
from bko.datasets.cells import LABEL
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.evaluate.data import supported_frame
from bko.evaluate.direct_arms import DIRECT_ARM, DIRECT_ARMS, KB_ARM
from bko.evaluate.named_runs import RUNS
from bko.gaps.observe import Cell
from bko.generate.direct import Draw, sample
from bko.generate.errors import SamplingError
from bko.generate.knowledge_draw import kb_sample
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, Term
from tests._arm_inputs import ARM_INPUTS, INPUTS


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ADR: Final = Path(__file__).parents[1] / "docs/adr/research/matched-knowledge-controls.md"

FRAME: Final = FrameSpec(
    name="inputs-toy",
    features=("age", "sex", "a", "b", "c", "d"),
    numeric=("age", "d"),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", "a", "b", "c"),
    codebook={
        "sex": {0: "female", 1: "male"},
        "a": {0: "none", 1: "mild", 2: "severe"},
        "b": {0: "no", 1: "yes"},
        "c": {0: "zero", 1: "one", 2: "two", 3: "three"},
    },
    descriptions={
        "age": "age",
        "sex": "sex",
        "a": "a grade",
        "b": "b flag",
        "c": "c level",
        "d": "a free count",
        "target": "the condition",
    },
    ranges=(
        RangeConstraint("age", "a", 20, 90, "toy", "toy"),
        RangeConstraint("d", "u", 0, 9, "toy", "toy"),
    ),
    label_codes=(0, 1),
    label_meanings={0: "absent", 1: "present"},
    edges=(40, 60),
    study="toy",
    cohort="toy patient",
)
"""Two mapping groups, `a` and the two-field `b`-`c`, and one free field, `d`."""


def _row(concept: str, *when: Codes) -> FieldMapping:
    return FieldMapping(when=when, concept=concept, label=concept, source="toy", limits="toy")


BROAD: Final = _row("MONDO:0000002", Codes("a", frozenset({1, 2})))
NARROW: Final = _row("MONDO:0000003", Codes("a", frozenset({2})))
JOINT: Final = _row("MONDO:0000010", Codes("b", frozenset({1})), Codes("c", frozenset({2, 3})))
C3: Final = _row("MONDO:0000011", Codes("c", frozenset({3})))
ROWS: Final = (BROAD, NARROW, JOINT, C3)

ROOT: Final = "MONDO:0000001"
PARENTS: Final[Mapping[str, str]] = {
    BROAD.concept: ROOT,
    NARROW.concept: BROAD.concept,
    JOINT.concept: ROOT,
    C3.concept: ROOT,
}
"""Each mapped concept's parent: `NARROW` is_a `BROAD`, the rest under the root."""


def _ontologies(parents: Mapping[str, str]) -> Mapping[str, Ontology]:
    terms = [Term(ROOT, ROOT, (), (), (), False, ())]
    terms += [Term(c, c, (p,), (), (), False, ()) for c, p in parents.items()]
    return {"MONDO": Ontology("mondo", "test", {t.curie: t for t in sorted(terms, key=str)})}


THIN: Final[Cell] = ("<40", "female", 1)
EMPTY: Final[Cell] = ("40-59", "female", 1)
QUOTA: Final[Mapping[Cell, int]] = {THIN: 40, EMPTY: 40}
"""A thin cell of 3 records and an emptied one, both of label 1."""

SEED: Final = 0
NAME: Final = "inputs-toy"
LOW: Final = {"<40": 25, "40-59": 40, ">=60": 60}


def _train() -> pd.DataFrame:
    """8 records per cell, 3 in `THIN` (each `d` 0) and none in `EMPTY`; no `c` of 0."""
    records: list[dict[str, int]] = []
    for band, low in LOW.items():
        for sex, sex_name in enumerate(("female", "male")):
            for label in (0, 1):
                cell = (band, sex_name, label)
                count = {THIN: 3, EMPTY: 0}.get(cell, 8)
                for j in range(count):
                    i = len(records)
                    records.append(
                        {
                            "age": low + (3 * j) % (15 if band == "<40" else 20),
                            "sex": sex,
                            "a": (i // 6) % 3,
                            "b": i % 2,
                            "c": 1 + (i // 2) % 3,
                            "d": 0 if cell == THIN else j % 4,
                            LABEL: label,
                        }
                    )
    frame = pd.DataFrame(records).astype(dict(FRAME.dtypes))
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
    return frame


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class World:
    """What a fill reads: the training split, K and the frame."""

    train: pd.DataFrame
    rows: tuple[FieldMapping, ...]
    ontologies: Mapping[str, Ontology]
    frame: FrameSpec


BASE: Final = World(_train(), ROWS, _ontologies(PARENTS), FRAME)


def _recode(where: Callable[[pd.DataFrame], pd.Series]) -> World:
    """`BASE` with every `d` of 0 in the rows `where` picks set to 9."""
    train = BASE.train.copy()
    picked = where(train) & train["d"].eq(0)
    assert picked.any()
    train.loc[picked, "d"] = 9
    return dataclasses.replace(BASE, train=train)


def _in_cell(train: pd.DataFrame) -> pd.Series:
    return train["age"].lt(40) & train["sex"].eq(0) & train[LABEL].eq(1)


def _rest_of_label(train: pd.DataFrame) -> pd.Series:
    return train[LABEL].eq(1) & ~_in_cell(train)


def _other_label(train: pd.DataFrame) -> pd.Series:
    return train[LABEL].eq(0)


def _drop_code(frame: FrameSpec) -> FrameSpec:
    codebook = {**frame.codebook, "c": {k: v for k, v in frame.codebook["c"].items() if k}}
    return dataclasses.replace(frame, codebook=codebook)


SWAPPED: Final = {
    **PARENTS,
    NARROW.concept: PARENTS[JOINT.concept],
    JOINT.concept: PARENTS[NARROW.concept],
}

PERTURBED: Final[Mapping[str, tuple[World, ...]]] = {
    "cell": (_recode(_in_cell),),
    "label-group": (_recode(_rest_of_label),),
    "other-label": (_recode(_other_label),),
    "mapping": tuple(
        dataclasses.replace(BASE, rows=tuple(r for r in ROWS if r is not dropped))
        for dropped in (JOINT, C3)
    ),
    "hierarchy": (dataclasses.replace(BASE, ontologies=_ontologies(SWAPPED)),),
    "codebook": (dataclasses.replace(BASE, frame=_drop_code(FRAME)),),
}
"""Each record and knowledge input's perturbations of `BASE`."""

ARMS: Final = (
    DIRECT_ARM,
    "bko-nomap",
    "bko-logic",
    "bko-donor",
    "bko-pool-random",
    "bko-pool-all",
    KB_ARM,
    str(Method.LABEL_BOOTSTRAP),
    str(Method.ROW_RESAMPLE),
    str(Method.MARGINALS),
    str(Method.EB_MARGINALS),
    str(Method.HB_MARGINALS),
)
"""The arms whose fills this module perturbs: E19's row-adding draws, and `bko-direct`,
`bko-logic` and `label-bootstrap` beside them. `smote` refuses `QUOTA`'s emptied cell, which holds
fewer than the two records it needs; `bayes-net` keeps only the codes a training record holds, so
dropping the unheld code 0 cannot move its fill. No other arm of `ARM_INPUTS` is filled here;
`test_the_adr_table_mirrors_arm_inputs` holds every arm to the ADR table."""


def _kb_frame(frame: FrameSpec) -> FrameSpec:
    """`frame` on the fields `ROWS` codes: `bko-kb`'s supported projection, fixed by `ROWS`."""
    return supported_frame(frame, ROWS)


def _draws(arm: str, world: World) -> list[Draw]:
    frame = _kb_frame(world.frame) if arm == KB_ARM else world.frame
    drawn = sample(
        world.train.loc[:, list(frame.columns)],
        QUOTA,
        rows=world.rows,
        ontologies=world.ontologies,
        name=NAME,
        seed=SEED,
        frame=frame,
        spec=DIRECT_ARMS[arm],
    )
    return [d for draws in drawn.values() for d in draws]


def _records(arm: str, world: World) -> Sequence[object]:
    if arm in DIRECT_ARMS:
        return [dict(d.record) for d in _draws(arm, world)]
    request = Request(quota=QUOTA, seed=SEED, frame=world.frame)
    added = augment(world.train, Method(arm), request).added
    return added.to_dict(orient="records")


def _digest(arm: str, world: World) -> str:
    """The sha256 of `arm`'s added records on `world`, or the refusal it raises."""
    try:
        records = _records(arm, world)
    except (ValueError, SamplingError, BaselineError) as error:
        return f"refused {type(error).__name__}: {error}"
    text = json.dumps(records, default=lambda value: value.item())
    return hashlib.sha256(text.encode()).hexdigest()


@functools.cache
def _outcome(arm: str, key: tuple[str, int] | None) -> str:
    """`_digest` on `BASE` at None, else on `key`'s perturbation, computed once per module."""
    return _digest(arm, BASE if key is None else PERTURBED[key[0]][key[1]])


@pytest.mark.parametrize("given", tuple(PERTURBED))
@pytest.mark.parametrize("arm", ARMS)
def test_an_arm_changes_only_when_a_declared_input_changes(arm, given):
    base = _outcome(arm, None)
    assert not base.startswith("refused"), base
    changed = [_outcome(arm, (given, k)) != base for k in range(len(PERTURBED[given]))]
    assert any(changed) is (given in ARM_INPUTS[arm]), changed


def _kb(age_range: tuple[int, int]) -> list[Draw]:
    frame = _kb_frame(FRAME)
    drawn = kb_sample(
        QUOTA, rows=ROWS, ontologies=BASE.ontologies, frame=frame, seed=SEED, age_range=age_range
    )
    return [d for draws in drawn.values() for d in draws]


def test_kb_age_follows_the_age_range_alone():
    narrow, wide = _kb((30, 50)), _kb((25, 59))
    rest = [c for c in _kb_frame(FRAME).columns if c != "age"]
    assert [{c: d.record[c] for c in rest} for d in narrow] == [
        {c: d.record[c] for c in rest} for d in wide
    ]
    assert [d.concepts for d in narrow] == [d.concepts for d in wide]
    assert [d.record["age"] for d in narrow] != [d.record["age"] for d in wide]


def test_a_hierarchy_swap_leaves_bko_direct_records_equal():
    before = _draws(DIRECT_ARM, BASE)
    after = _draws(DIRECT_ARM, PERTURBED["hierarchy"][0])
    assert [d.record for d in before] == [d.record for d in after]
    assert [d.concepts for d in before] == [d.concepts for d in after]
    assert [d.closed for d in before] != [d.closed for d in after]


def _table(heading: str) -> list[list[str]]:
    """The cells of the markdown table that first follows `heading` in the ADR."""
    lines = ADR.read_text(encoding="utf-8").splitlines()
    start = next(k for k, line in enumerate(lines) if heading in line)
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
    return rows


def _name(cell: str) -> str:
    match = re.fullmatch(r"`([^`]+)`", cell)
    assert match, cell
    return match[1]


def test_the_adr_table_mirrors_arm_inputs():
    header, rule, *body = _table("**Declared inputs.**")
    assert header == ["Arm", *INPUTS]
    assert set("".join(rule)) <= {"-"}
    found: dict[str, frozenset[str]] = {}
    for arm, *marks in body:
        assert set(marks) <= {"yes", "no"}, arm
        found[_name(arm)] = frozenset(
            i for i, mark in zip(INPUTS, marks, strict=True) if mark == "yes"
        )
    assert list(found) == list(ARM_INPUTS)
    assert found == dict(ARM_INPUTS)


def _registered(prefixes: Sequence[str]) -> set[str]:
    """Every arm of a `RUNS` entry named a prefix of `prefixes` or the prefix and a hyphen."""
    return {
        arm
        for name, run in RUNS.items()
        if any(name == p or name.startswith(f"{p}-") for p in prefixes)
        for arm in run.experiment.arms
    }


def test_every_e19_arm_declares_its_inputs():
    design = {_name(arm) for arm, *_ in _table("2. **The arms.**")[2:]}
    families = _table("12. **The runs.**")[2:]
    e19 = {_name(a.strip()) for *_, arms in families for a in arms.split(",")}
    assert design <= e19
    assert all(arm in ARM_INPUTS for arm in ARMS)
    assert set(ARM_INPUTS) == _registered(("e9", "e12", "e13", "e15", "e19")) | e19
    assert all(set(inputs) <= set(INPUTS) for inputs in ARM_INPUTS.values())
