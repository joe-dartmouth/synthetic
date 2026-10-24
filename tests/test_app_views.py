"""`bko.app.views`: the frames the page shows, read without streamlit."""

from __future__ import annotations

import dataclasses
import io
import math

import pandas as pd
import pytest

from bko.app.runs import call_totals, discover
from bko.app.steps import Analysis, Outcome, analyze, fill_and_score, scores_csv, training_csv
from bko.app.views import (
    GAP_COLUMNS,
    PREDICTION,
    ROLES,
    TOTALS,
    arm_means,
    balance,
    candidates_table,
    cell_table,
    closure_counts,
    closure_pairs,
    closure_summary,
    e1_validity,
    gap_counts,
    gap_table,
    prediction_column,
    prediction_table,
    role_table,
    score_table,
    status_counts,
    totals_table,
)
from bko.baselines.__main__ import render
from bko.datasets.cells import LABEL_MEANINGS
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1
from bko.evaluate.scores import Measure, Scope, Score, Source, parse
from bko.gaps.definitions import Definition
from bko.gaps.report import labeller
from bko.generate.candidates import render_lines
from bko.generate.decode import Reason, Undecoded
from tests._app_fixtures import app_context, never, one_fill
from tests._fusion_fixtures import young
from tests._runs_fixtures import (
    CLOSURE,
    GAPS_SYNTHEA,
    GAPS_UCI,
    HEAD,
    REPRODUCTION,
    SCORES,
    call_log,
    every_kind,
)


@pytest.fixture
def analysis(tmp_path) -> Analysis:
    return analyze(app_context(tmp_path, send=never)[0])


def test_cell_table_names_each_cells_status_in_words(analysis):
    table = cell_table(analysis)
    expected = analysis.counts["underrepresented"].map({True: "underrepresented", False: "covered"})
    assert table["status"].tolist() == expected.tolist()
    assert set(table["status"]) == {"underrepresented", "covered"}


def test_balance_counts_training_records_by_label(analysis):
    table = balance(analysis)
    assert table["count"].sum() == len(analysis.parts.train)
    assert sorted(table["target"]) == sorted(analysis.parts.train["target"].unique())
    assert table.set_index("target")["meaning"].to_dict() == {
        code: LABEL_MEANINGS[code] for code in table["target"]
    }


def test_gap_table_lists_each_gap_with_its_labels(analysis):
    label = labeller(analysis.knowledge.ontologies)
    for definition in Definition:
        table = gap_table(analysis, definition)
        found = analysis.found[definition]
        assert len(table) == len(found)
        assert tuple(table.columns) == GAP_COLUMNS
        assert table["antecedent_label"].tolist() == [
            label(gap.statement.antecedent) for gap in found
        ]
    assert any(len(analysis.found[definition]) for definition in Definition)


@pytest.fixture(params=["smote", "bko"])
def filled(request, tmp_path) -> tuple[Analysis, Outcome]:
    """A SMOTE fill, or a BKO fill over scripted replies."""
    send = one_fill()
    context, _ = app_context(tmp_path, send=lambda: send)
    analysis = analyze(context)
    return analysis, fill_and_score(analysis, request.param, context)


def test_closure_summary_counts_each_definitions_pairs(filled):
    _, outcome = filled
    table = closure_summary(outcome)
    assert table.values.tolist() == [
        [c.definition.value, len(c.before), len(c.closed), len(c.still_open), len(c.opened)]
        for c in outcome.closures
    ]


def test_closure_pairs_partition_the_listed_pairs(filled):
    analysis, outcome = filled
    table = closure_pairs(analysis, outcome)
    for reading in outcome.closures:
        rows = table[table["definition"] == reading.definition.value]
        counts = rows["status"].value_counts().to_dict()
        assert counts.get("closed", 0) == len(reading.closed)
        assert counts.get("still open", 0) == len(reading.still_open)
        assert counts.get("new", 0) == len(reading.opened)
        assert len(rows) == len(reading.listed(analysis.cells.each))


def test_score_table_has_one_column_per_arm_and_keeps_an_undefined_score(filled):
    _, outcome = filled
    planted = dataclasses.replace(
        outcome,
        scores=(
            *outcome.scores,
            Score(0, "smote", Source.ADDED, Measure.DCR_P5, Scope.FLAGGED, math.nan),
        ),
    )
    table = score_table(planted.scores)
    assert set(table.columns) == {score.arm for score in planted.scores}
    assert list(table.index.names) == ["source", "measure", "scope"]
    assert pd.isna(table.loc[("added", "dcr-p5", "flagged-subgroups"), "smote"])


def test_the_training_csv_is_what_bko_baselines_prints(filled):
    analysis, outcome = filled
    assert training_csv(analysis, outcome) == render(analysis.parts.train, outcome.baseline)


def test_the_scores_csv_reads_back(filled):
    _, outcome = filled
    back = parse(scores_csv(outcome))
    assert len(back) == len(outcome.scores)
    for read, kept in zip(back, outcome.scores, strict=True):
        assert dataclasses.astuple(read)[:5] == dataclasses.astuple(kept)[:5]
        assert read.value == kept.value or (math.isnan(read.value) and math.isnan(kept.value))


def test_score_table_keeps_only_the_named_source(filled):
    _, outcome = filled
    table = score_table(outcome.scores, Source.AUGMENTED)
    assert set(table.index.get_level_values("source")) == {"augmented"}
    assert len(table) < len(score_table(outcome.scores))


def test_prediction_table_puts_the_filled_arm_first_and_keeps_every_augmented_score(filled):
    _, outcome = filled
    table = prediction_table(outcome.scores, outcome.name)
    arms = list(dict.fromkeys(s.arm for s in outcome.scores if s.source is Source.AUGMENTED))
    assert table["arm"].tolist()[0] == outcome.name
    assert sorted(table["arm"]) == sorted(arms)
    assert table["role"].tolist() == [ROLES[0], *[ROLES[1]] * (len(arms) - 1)]
    columns = [prediction_column(m, scope) for scope in Scope for m in PREDICTION]
    assert table.columns.tolist() == ["arm", "role", *columns]
    for s in outcome.scores:
        if s.source is Source.AUGMENTED and s.measure in PREDICTION:
            shown = table.set_index("arm").loc[s.arm, prediction_column(s.measure, s.scope)]
            assert shown == s.value or (math.isnan(shown) and math.isnan(s.value))


def test_prediction_table_reads_no_other_source_and_keeps_an_undefined_score(filled):
    _, outcome = filled
    planted = (
        *outcome.scores,
        Score(0, "smote", Source.ADDED, Measure.RECALL, Scope.ALL, 99.0),
        Score(0, "smote", Source.AUGMENTED, Measure.ROWS, Scope.ALL, 99.0),
    )
    table = prediction_table(planted, outcome.name)
    assert 99.0 not in table.drop(columns=["arm", "role"]).to_numpy()
    assert table.drop(columns=["arm", "role"]).isna().any().any()


def test_prediction_table_of_an_arm_with_no_scores_lists_the_others():
    scores = (Score(0, "none", Source.AUGMENTED, Measure.RECALL, Scope.ALL, 0.5),)
    table = prediction_table(scores, "bko")
    assert table["arm"].tolist() == ["none"]
    assert table["role"].tolist() == [ROLES[1]]


def _csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


def test_arm_means_average_each_measure_over_the_holdouts():
    means = arm_means(SCORES)
    assert set(means.columns) == {"none", "bko", "bko-population", "holdouts"}
    assert (means["holdouts"] == len(E1.holdout_seeds)).all()
    recall = list(means.loc[("augmented", "recall", "all"), ["none", "bko"]])
    assert recall == pytest.approx([0.5, 0.6])


def test_e1_validity_sums_the_candidate_counts():
    table = e1_validity(SCORES)
    holdouts = len(E1.holdout_seeds)
    assert table.to_dict("records") == [
        {
            "arm": "bko",
            "generated": 50 * holdouts,
            "passed": 50 * holdouts,
            "rate": 1.0,
            "verdict": "met",
        },
        {
            "arm": "bko-population",
            "generated": 50 * holdouts,
            "passed": 25 * holdouts,
            "rate": 0.5,
            "verdict": "reported",
        },
    ]
    assert e1_validity(SCORES[:8]).empty


def test_e1_validity_refuses_a_partial_run():
    with pytest.raises(EvaluationError):
        e1_validity([score for score in SCORES if score.holdout != 0])


def test_candidates_table_has_one_row_per_line(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(HEAD + render_lines(1, ("<55", "female", 1), [young(1)] * 2), encoding="utf-8")
    table = candidates_table(path)
    assert len(table) == 2
    assert {"call", "cell", "position", "age"} <= set(table.columns)
    path.write_text(
        HEAD + render_lines(1, ("<55", "female", 1), [Undecoded(Reason.TRUNCATED)]),
        encoding="utf-8",
    )
    assert candidates_table(path)["undecoded"].tolist() == ["truncated"]


def test_closure_counts_classify_each_pair():
    counts = closure_counts(_csv(CLOSURE))
    assert counts["definition"].tolist() == [definition.value for definition in Definition]
    rows = counts.set_index("definition")
    assert list(rows.loc["per-subgroup"]) == [1, 1, 0]
    assert list(rows.loc["population-level"]) == [0, 0, 0]
    assert list(rows.loc["frequency-aware"]) == [0, 0, 1]


def test_role_table_lists_each_role_and_its_size(tmp_path):
    run = next(run for run in discover(every_kind(tmp_path)) if run.name == "e1-3")
    table = role_table(run)
    assert table["role"].tolist() == ["calls.jsonl", "scores.csv", "criteria.csv"]
    assert table["bytes"].tolist() == [path.stat().st_size for path in run.roles.values()]


def test_gap_counts_count_pairs_by_reading():
    synthea = gap_counts(_csv(GAPS_SYNTHEA), ("ablation", "detector"))
    first = synthea.iloc[0]
    assert (first["ablation"], first["detector"]) == ("cell", "per-subgroup")
    assert [first["pairs"], first["before"], first["after"], first["truth"]] == [1, 1, 1, 0]
    uci = gap_counts(_csv(GAPS_UCI), ("detector",))
    assert uci["detector"].tolist() == ["per-subgroup", "count"]


def test_status_counts_count_reproduction_rows():
    counts = status_counts(_csv(REPRODUCTION))
    assert dict(zip(counts["status"], counts["count"], strict=True)) == {"match": 1, "erratum": 1}


def test_totals_table_keeps_the_end_lines_counts(tmp_path):
    totals = call_totals(call_log(tmp_path / "x.jsonl"))
    assert totals is not None
    table = totals_table(totals)
    assert (
        tuple(table.columns)
        == TOTALS
        == (
            "status",
            "calls",
            "failed",
            "retried",
            "input_tokens",
            "output_tokens",
            "usd",
            "live_usd",
            "replay",
            "resume",
        )
    )
    assert table.iloc[0]["calls"] == 1


def test_a_reading_with_a_blank_is_refused():
    blank = CLOSURE.replace("True,False\n", "True,\n", 1)
    with pytest.raises(ValueError, match="'after'"):
        closure_counts(_csv(blank))
    with pytest.raises(ValueError, match="'truth'"):
        gap_counts(_csv(GAPS_SYNTHEA.replace(",False\n", ",\n", 1)), ("ablation", "detector"))
