"""`bko.ontology.automatic` builds mapping rows from a codebook and releases, with no model."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from bko.datasets.cells import SEXES
from bko.datasets.diagnoses import code_number
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.ontology import automatic, automatic_inputs
from bko.ontology.automatic import (
    SOURCE,
    THRESHOLD,
    build_lexical,
    candidates,
    permuted,
    precision_recall,
    render,
    rows_from_verdicts,
)
from bko.ontology.check import check
from bko.ontology.obo import parse_obo
from bko.ontology.schema import AtLeast, Codes, FieldMapping
from bko.ontology.terms import Ontology
from tests._corpus import REPO_ROOT
from tests._ontology_fixtures import write_obo


MONDO = """\
format-version: 1.2
data-version: releases/2026-10-06
ontology: mondo

[Term]
id: MONDO:0005015
name: diabetes mellitus

[Term]
id: MONDO:0005044
name: hypertensive disorder
synonym: "hypertension" EXACT []

[Term]
id: MONDO:0001134
name: essential hypertension
is_a: MONDO:0005044 ! hypertensive disorder

[Term]
id: MONDO:0005249
name: pneumonia
xref: ICD9:486 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0005607
name: chronic bronchitis
xref: ICD9:491 {source="MONDO:relatedTo"}

[Term]
id: MONDO:0000002
name: kidney stone

[Term]
id: MONDO:0000001
name: stone kidney

[Term]
id: MONDO:0021187
name: familial hypercholesterolemia

[Term]
id: MONDO:0000003
name: gallstone
is_obsolete: true
"""

HPO = """\
format-version: 1.2
data-version: hp/releases/2026-09-01
ontology: hp

[Term]
id: HP:0000822
name: Hypertension

[Term]
id: HP:0003074
name: Hyperglycemia
"""

PNEUMONIA = code_number("486")
BRONCHITIS = code_number("491")
DIABETES_CODE = code_number("250")


def _range(field: str, unit: str, low: float, high: float) -> RangeConstraint:
    return RangeConstraint(field=field, unit=unit, low=low, high=high, source="toy", limits="toy")


FRAME = FrameSpec(
    name="toy-codebook",
    features=("age", "sex", "diabetes", "htn", "glucose", "chol", "stone", "gallstone", "diag"),
    numeric=("age",),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", "diabetes", "htn", "glucose", "chol", "stone", "gallstone", "diag"),
    codebook={
        "sex": SEXES,
        "diabetes": {0: "absent", 1: "present"},
        "htn": {0: "no", 1: "hypertension"},
        "glucose": {0: "normal", 1: "high glucose", 2: "Hyperglycemia!"},
        "chol": {0: "none", 1: "familial hypercholesterolemia type"},
        "stone": {0: "not recorded", 1: "Kidney  stone"},
        "gallstone": {1: "gallstone"},
        "diag": {PNEUMONIA: "486", BRONCHITIS: "491", DIABETES_CODE: "250"},
    },
    descriptions={
        "age": "age in years",
        "sex": "sex",
        "diabetes": "diabetes mellitus",
        "htn": "blood pressure",
        "glucose": "glucose",
        "chol": "cholesterol disorder",
        "stone": "stones",
        "gallstone": "gallstones",
        "diag": "ICD-9 code of the primary diagnosis",
        "target": "toy readmission",
    },
    ranges=(_range("age", "a", 20, 90),),
    label_codes=(0, 1),
    label_meanings={0: "not readmitted", 1: "readmitted"},
    edges=(45, 70),
    study="toy readmission",
    cohort="toy patient",
)
"""A codebook exercising every route: flags, levels, a tie, an obsolete term and ICD-9 codes."""


def _ontologies(directory: Path) -> dict[str, Ontology]:
    return {
        "MONDO": parse_obo(write_obo(directory, MONDO, "mondo.obo"), name="mondo"),
        "HP": parse_obo(write_obo(directory, HPO, "hp.obo"), name="hpo"),
    }


def _rendered(directory: str) -> str:
    """The fixture's rendered rows; a subprocess calls this under another hash seed."""
    return render(build_lexical(FRAME, _ontologies(Path(directory)), codes=("diag",)))


@pytest.fixture
def ontologies(tmp_path: Path) -> dict[str, Ontology]:
    return _ontologies(tmp_path)


def _row(field: str, code: int, concept: str, label: str = "x") -> FieldMapping:
    return FieldMapping(
        when=(Codes(field, frozenset({code})),),
        concept=concept,
        label=label,
        source="hand",
        limits="hand",
    )


def _concepts(rows: tuple[FieldMapping, ...]) -> dict[tuple[str, int], str]:
    found: dict[tuple[str, int], str] = {}
    for row in rows:
        (clause,) = row.when
        assert isinstance(clause, Codes)
        (code,) = clause.codes
        found[clause.field, code] = row.concept
    return found


def test_a_coded_value_maps_through_its_cross_reference(ontologies):
    rows = build_lexical(FRAME, ontologies, codes=("diag",))
    found = _concepts(rows)
    assert found["diag", PNEUMONIA] == "MONDO:0005249"
    assert ("diag", BRONCHITIS) not in found
    assert ("diag", DIABETES_CODE) not in found
    assert all(row.source == SOURCE == "research/automatic-mapping" for row in rows)


def test_a_relatedto_code_is_a_candidate_with_its_holders(ontologies):
    found = {(c.field, c.code): c for c in candidates(FRAME, ontologies, codes=("diag",))}
    assert [option.curie for option in found["diag", BRONCHITIS].options] == ["MONDO:0005607"]
    assert ("diag", PNEUMONIA) not in found
    assert ("diag", DIABETES_CODE) not in found


def test_a_flag_maps_to_its_best_label_at_or_above_the_threshold(ontologies):
    found = _concepts(build_lexical(FRAME, ontologies, codes=("diag",)))
    # A generic meaning reads the field's description: "diabetes mellitus", Jaccard 1.
    assert found["diabetes", 1] == "MONDO:0005015"
    assert found["glucose", 2] == "HP:0003074"
    # "familial hypercholesterolemia type" against its name: Jaccard 2/3, below 4/5.
    assert Fraction(2, 3) < THRESHOLD
    assert ("chol", 1) not in found
    assert ("glucose", 1) not in found
    by_state = {(c.field, c.code): c for c in candidates(FRAME, ontologies, codes=("diag",))}
    best = by_state["chol", 1].options[0]
    assert (best.curie, best.score) == ("MONDO:0021187", Fraction(2, 3))


def test_a_name_and_a_synonym_tie_goes_to_the_first_release_prefix(ontologies):
    found = _concepts(build_lexical(FRAME, ontologies, codes=("diag",)))
    # HP's name "Hypertension" and MONDO's EXACT synonym "hypertension" tie at 1; HP sorts first.
    assert found["htn", 1] == "HP:0000822"


def test_a_negative_state_is_skipped(ontologies):
    states = {(c.field, c.code) for c in candidates(FRAME, ontologies, codes=("diag",))}
    rows = _concepts(build_lexical(FRAME, ontologies, codes=("diag",)))
    for state in (("diabetes", 0), ("htn", 0), ("glucose", 0), ("chol", 0), ("stone", 0)):
        assert state not in states
        assert state not in rows
    assert ("target", 0) not in states


def test_an_obsolete_term_is_no_target(ontologies):
    assert ("gallstone", 1) not in _concepts(build_lexical(FRAME, ontologies, codes=("diag",)))


def test_numeric_columns_stay_unmapped(ontologies):
    rows = build_lexical(FRAME, ontologies, codes=("diag",))
    fields = {clause.field for row in rows for clause in row.when}
    assert "age" not in fields
    assert all(c.field != "age" for c in candidates(FRAME, ontologies, codes=("diag",)))


def test_ties_go_to_the_smaller_curie(ontologies):
    # "Kidney  stone" scores 1 against "kidney stone" and "stone kidney" alike.
    assert _concepts(build_lexical(FRAME, ontologies))["stone", 1] == "MONDO:0000001"


def test_the_builder_takes_no_records():
    for function in (build_lexical, candidates):
        parameters = inspect.signature(function).parameters
        assert list(parameters) == ["frame", "ontologies", "codes"]
        assert "DataFrame" not in str(inspect.signature(function))


def test_the_module_imports_no_later_stage():
    for module in (automatic, automatic_inputs):
        tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
        names = [
            name
            for node in ast.walk(tree)
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
        ]
        assert names
        assert not [n for n in names if n.startswith(("bko.generate", "bko.evaluate"))]


def test_rows_are_sorted_by_field_code_and_concept(ontologies):
    rows = build_lexical(FRAME, ontologies, codes=("diag",))
    keys = list(_concepts(rows).items())
    assert keys == sorted(keys)


DRIVER = (
    "import sys\n"
    "from tests.test_ontology_automatic import _rendered\n"
    "sys.stdout.write(_rendered(sys.argv[1]))\n"
)


def test_render_is_byte_equal_under_two_hash_seeds(tmp_path):
    outputs = []
    for seed in ("0", "1"):
        path = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": path}
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                sys.executable,
                "-c",
                DRIVER,
                str(tmp_path),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1] == _rendered(str(tmp_path))
    assert "MONDO:0005249" in outputs[0]


def test_the_rendered_module_holds_the_rows(ontologies):
    rows = build_lexical(FRAME, ontologies, codes=("diag",))
    namespace: dict[str, object] = {}
    exec(compile(render(rows), "<rendered>", "exec"), namespace)  # ruff: ignore[exec-builtin]
    assert namespace["ROWS"] == rows


def test_the_rows_pass_the_structural_check(ontologies):
    rows = build_lexical(FRAME, ontologies, codes=("diag",))
    assert rows
    verdicts = check(rows, ontologies)
    assert [v.problem for v in verdicts] == [None] * len(rows)


def test_precision_and_recall_on_the_fixture_codebook(ontologies):
    frame = dataclasses.replace(
        FRAME,
        features=("age", "sex", "diabetes", "htn", "glucose"),
        categorical=("sex", "diabetes", "htn", "glucose"),
        codebook={key: FRAME.codebook[key] for key in ("sex", "diabetes", "htn", "glucose")},
        descriptions={
            key: FRAME.descriptions[key]
            for key in ("age", "sex", "diabetes", "htn", "glucose", "target")
        },
    )
    built = build_lexical(frame, ontologies)
    assert len(built) == 3
    hand = (
        _row("diabetes", 1, "MONDO:0005015"),
        _row("htn", 1, "MONDO:0005044"),
        _row("glucose", 2, "HP:0003074"),
        _row("chol", 1, "MONDO:0021187"),
    )
    assert precision_recall(built, hand) == (Fraction(2, 3), Fraction(1, 2))


def test_a_row_of_several_codes_or_a_cutoff_counts_as_its_atoms():
    hand = (
        FieldMapping(
            when=(Codes("cp", frozenset({1, 2})),),
            concept="HP:1",
            label="x",
            source="hand",
            limits="hand",
        ),
        FieldMapping(
            when=(AtLeast("chol", 240),), concept="HP:2", label="y", source="h", limits="h"
        ),
    )
    assert precision_recall((_row("cp", 2, "HP:1"),), hand) == (Fraction(1), Fraction(1, 3))


def test_precision_and_recall_are_none_on_empty_denominators():
    assert precision_recall((), (_row("a", 1, "HP:1"),)) == (None, Fraction(0))
    assert precision_recall((_row("a", 1, "HP:1"),), ()) == (Fraction(0), None)
    assert precision_recall((), ()) == (None, None)


def test_permuted_rows_keep_every_field_state_and_move_concepts():
    hand = (
        _row("a", 1, "MONDO:0000001", "one"),
        _row("b", 1, "MONDO:0000002", "two"),
        _row("c", 1, "MONDO:0000003", "three"),
        _row("c", 2, "MONDO:0000003", "three"),
        _row("d", 1, "HP:0000001", "first"),
        _row("e", 1, "HP:0000002", "second"),
    )
    shuffled = permuted(hand, seed=0)
    assert shuffled == permuted(hand, seed=0)
    assert [row.when for row in shuffled] == [row.when for row in hand]
    assert {row.concept for row in shuffled} == {row.concept for row in hand}
    assert any(a.concept != b.concept for a, b in zip(shuffled, hand, strict=True))
    labels = {row.concept: row.label for row in hand}
    for row, original in zip(shuffled, hand, strict=True):
        assert row.label == labels[row.concept]
        assert row.concept.partition(":")[0] == original.concept.partition(":")[0]
        assert (row.source, row.limits) == (original.source, original.limits)
    # Rows that shared a concept still share one.
    assert shuffled[2].concept == shuffled[3].concept


def test_rows_from_verdicts_skip_none_and_name_each_concept():
    verdicts = {"htn=1": "HP:0000822", "glucose=2": "none", "diabetes=1": "MONDO:0005015"}
    labels = {"HP:0000822": "Hypertension", "MONDO:0005015": "diabetes mellitus"}
    rows = rows_from_verdicts(FRAME, verdicts, labels)
    assert _concepts(rows) == {("diabetes", 1): "MONDO:0005015", ("htn", 1): "HP:0000822"}
    assert [row.label for row in rows] == ["diabetes mellitus", "Hypertension"]
    assert rows == rows_from_verdicts(FRAME, dict(reversed(verdicts.items())), labels)


@pytest.mark.parametrize(
    ("verdicts", "message"),
    [
        ({"age=40": "HP:0000822"}, "not a coded field"),
        ({"htn=7": "HP:0000822"}, "not a code of htn"),
        ({"htn": "HP:0000822"}, "<field>=<code>"),
        ({"htn=1": "HP:0003074"}, "no label"),
    ],
)
def test_rows_from_verdicts_refuse_an_unknown_state_or_label(verdicts, message):
    with pytest.raises(ValueError, match=message):
        rows_from_verdicts(FRAME, verdicts, {"HP:0000822": "Hypertension"})
