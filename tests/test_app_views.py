"""`bko.app.views`: the frames the page shows, read without streamlit."""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

from bko.app.steps import Analysis, Outcome, analyze, fill_and_score, scores_csv, training_csv
from bko.app.views import (
    GAP_COLUMNS,
    balance,
    cell_table,
    closure_pairs,
    closure_summary,
    gap_table,
    score_table,
)
from bko.baselines.__main__ import render
from bko.evaluate.scores import Measure, Scope, Score, Source, parse
from bko.gaps.definitions import Definition
from bko.gaps.report import labeller
from tests._app_fixtures import app_context, never, one_fill


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
