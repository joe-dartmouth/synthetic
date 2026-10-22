"""`bko.app.views`: the frames the page shows, read without streamlit."""

from __future__ import annotations

import pytest

from bko.app.steps import Analysis, analyze
from bko.app.views import GAP_COLUMNS, balance, cell_table, gap_table
from bko.gaps.definitions import Definition
from bko.gaps.report import labeller
from tests._app_fixtures import app_context, never


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
