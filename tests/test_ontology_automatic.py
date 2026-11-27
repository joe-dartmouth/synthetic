"""`bko.ontology.automatic` builds mapping rows from a codebook and releases, with no model."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from bko.datasets.diagnoses import code_number
from bko.datasets.errors import ContractError
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.ontology import auto_measures, automatic, automatic_inputs, lexical
from bko.ontology.auto_measures import precision_recall
from bko.ontology.automatic import (
    SOURCE,
    THRESHOLD,
    TOP,
    build_lexical,
    candidates,
    permuted,
    render,
    rows_from_verdicts,
    survey,
    unmovable,
)
from bko.ontology.check import check
from bko.ontology.codes import REASONS
from bko.ontology.obo import parse_obo
from bko.ontology.schema import AtLeast, Codes, FieldMapping
from bko.ontology.terms import Ontology
from tests._automatic_fixtures import (
    BRONCHITIS,
    DIABETES_CODE,
    FRAME,
    PNEUMONIA,
    automatic_ontologies,
)
from tests._corpus import REPO_ROOT
from tests._ontology_fixtures import write_obo


def _rendered(directory: str) -> str:
    """The fixture's rendered rows; a subprocess calls this under another hash seed."""
    return render(build_lexical(FRAME, automatic_ontologies(Path(directory)), codes=("diag",)))


@pytest.fixture
def ontologies(tmp_path: Path) -> dict[str, Ontology]:
    return automatic_ontologies(tmp_path)


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
    for module in (automatic, automatic_inputs, auto_measures, lexical):
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


def _code_release(directory: Path) -> Ontology:
    """A MONDO release exercising every branch of the cross-reference route."""
    weaker = "".join(
        f"[Term]\nid: MONDO:01{i:05d}\nname: weak {i}\n"
        'xref: ICD9:403 {source="MONDO:relatedTo"}\n\n'
        for i in range(TOP + 2, 0, -1)
    )
    text = (
        "format-version: 1.2\ndata-version: releases/2026-10-06\nontology: mondo\n\n"
        '[Term]\nid: MONDO:0000010\nname: broad\nxref: ICD9:401 {source="MONDO:equivalentTo"}\n\n'
        "[Term]\nid: MONDO:0000011\nname: narrow\nis_a: MONDO:0000010 ! broad\n"
        'xref: ICD9:401 {source="MONDO:equivalentTo"}\n\n'
        '[Term]\nid: MONDO:0000020\nname: left\nxref: ICD9:402 {source="MONDO:equivalentTo"}\n\n'
        '[Term]\nid: MONDO:0000021\nname: right\nxref: ICD9:402 {source="MONDO:equivalentTo"}\n\n'
        '[Term]\nid: MONDO:0000030\nname: v\nxref: ICD9:V45 {source="MONDO:equivalentTo"}\n\n'
        + weaker
    )
    return parse_obo(write_obo(directory, text, "codes.obo"), name="mondo")


CODES = {
    0: "?",
    code_number("401"): "401",
    code_number("402"): "402",
    code_number("403"): "403",
    code_number("404"): "404",
    code_number("V45"): "V45",
    code_number("E888"): "E888",
}


def _code_frame() -> FrameSpec:
    return dataclasses.replace(
        FRAME,
        features=("age", "sex", "diag"),
        categorical=("sex", "diag"),
        codebook={"sex": FRAME.codebook["sex"], "diag": CODES},
        descriptions={"age": "a", "sex": "sex", "diag": "d", "target": "t"},
    )


def test_the_cross_reference_route_on_each_kind_of_code(tmp_path):
    found = survey(_code_frame(), {"MONDO": _code_release(tmp_path)}, ("diag",))
    assert _concepts(found.rows) == {("diag", code_number("401")): "MONDO:0000010"}
    (candidate,) = found.candidates
    assert candidate.code == code_number("403")
    assert [o.curie for o in candidate.options] == [f"MONDO:01{i:05d}" for i in range(1, TOP + 1)]
    assert {o.kind for o in candidate.options} == {"xref"}
    assert found.dropped == {
        **dict.fromkeys(REASONS, 0),
        "V code": 1,
        "E code": 1,
        "holders not nested": 1,
        "no live xref": 1,
    }


@pytest.mark.parametrize(
    ("codes", "ontologies", "message"),
    [(("age",), True, "not a coded field"), (("diag",), False, "need the MONDO release")],
)
def test_the_survey_refuses_a_bad_code_field(tmp_path, codes, ontologies, message):
    releases = {"MONDO": _code_release(tmp_path)} if ontologies else {}
    with pytest.raises(ValueError, match=message):
        survey(_code_frame(), releases, codes)


def test_the_lexical_command_writes_utf8_bytes_and_logs_their_digest(
    tmp_path, monkeypatch, capsysbinary, caplog
):
    caplog.set_level("INFO")
    frame = dataclasses.replace(FRAME, label_meanings={0: "not readmitted", 1: "Café hypertension"})
    ontologies = automatic_ontologies(tmp_path)
    monkeypatch.setattr(
        automatic, "inputs", lambda dataset, data_dir: (frame, ontologies, ("diag",))
    )
    assert automatic.main(["lexical", "diabetes-codes"]) == 0
    data = capsysbinary.readouterr().out
    assert data == render(build_lexical(frame, ontologies, codes=("diag",))).encode()
    digest = hashlib.sha256(data).hexdigest()
    assert any(
        m.startswith("lexical dataset=diabetes-codes rows=") and m.endswith(f"sha256={digest}")
        for m in caplog.messages
    )
    assert 'lexical dropped reason="V code" states=0' in caplog.messages


def test_the_lexical_command_exits_two_on_a_refused_code_field(tmp_path, monkeypatch, capsysbinary):
    ontologies = automatic_ontologies(tmp_path)
    monkeypatch.setattr(
        automatic, "inputs", lambda dataset, data_dir: (FRAME, ontologies, ("age",))
    )
    assert automatic.main(["lexical", "diabetes"]) == 2
    assert capsysbinary.readouterr().out == b""


def test_the_lexical_command_returns_a_dataset_errors_code(monkeypatch, capsysbinary):
    def refuse(dataset: str, data_dir: Path | None) -> None:
        raise ContractError("the snapshot contradicts its pin")

    monkeypatch.setattr(automatic, "inputs", refuse)
    assert automatic.main(["lexical", "uci"]) == 3
    assert capsysbinary.readouterr().out == b""


def test_uci_is_read_with_no_code_field(tmp_path, monkeypatch):
    ontologies = automatic_ontologies(tmp_path)
    monkeypatch.setattr(automatic_inputs, "load_releases", lambda *args, **kwargs: ontologies)
    frame, releases, codes = automatic_inputs.inputs("uci", tmp_path)
    assert (frame, releases, codes) == (UCI_FRAME, ontologies, ())
    with pytest.raises(ValueError, match="no automatic mapping"):
        automatic_inputs.inputs("synthea", tmp_path)


def test_permuted_moves_every_concept_but_a_prefixs_lone_one():
    hand = (*(_row(f"f{i}", 1, f"MONDO:{i:07d}") for i in range(2)), _row("g", 1, "HP:0000001"))
    assert unmovable(hand) == 1
    for seed in range(20):
        shuffled = permuted(hand, seed=seed)
        assert [r.concept for r in shuffled[:2]] == ["MONDO:0000001", "MONDO:0000000"]
        assert shuffled[2].concept == "HP:0000001"
