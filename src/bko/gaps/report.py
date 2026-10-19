"""The CSV columns and cells the commands that report gaps share."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from bko.datasets.cells import CELL_COLUMNS
from bko.gaps.detect import Statement
from bko.gaps.observe import Cell
from bko.ontology.terms import Ontology


type Row = tuple[str, ...]
type Label = Callable[[str], str]

STATEMENT_COLUMNS: Final[Row] = (
    "source",
    "antecedent",
    "antecedent_label",
    "consequent",
    "consequent_label",
)


def cell_cells(cell: Cell) -> Row:
    """A cell's `CELL_COLUMNS` values, `target` blank where cells carry no label.

    Args:
        cell: The cell.

    Returns:
        One value per `CELL_COLUMNS` entry.
    """
    return tuple(str(part) for part in cell) + ("",) * (len(CELL_COLUMNS) - len(cell))


def labeller(ontologies: Mapping[str, Ontology]) -> Label:
    """A function naming a CURIE's term in the loaded ontology of its prefix.

    Args:
        ontologies: The loaded ontologies by prefix.

    Returns:
        The term's name, or an empty string when it has none.
    """

    def label(curie: str) -> str:
        return ontologies[curie.partition(":")[0]].terms[curie].name or ""

    return label


def statement_cells(statement: Statement, label: Label) -> Row:
    """A statement's `STATEMENT_COLUMNS` values.

    Args:
        statement: The statement.
        label: Names each concept.

    Returns:
        The source, then each concept and its label.
    """
    return (
        statement.source,
        statement.antecedent,
        label(statement.antecedent),
        statement.consequent,
        label(statement.consequent),
    )
