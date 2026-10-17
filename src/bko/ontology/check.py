"""Check the table's concept rows against parsed releases: `python -m bko.ontology check`'s core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from bko.ontology.schema import Clause, CodeMapping, Codes, ConceptRow, FieldMapping
from bko.ontology.terms import Ontology


type Table = Literal["field", "code", "out-of-scope"]


@dataclass(frozen=True, slots=True)
class Verdict:
    """One concept row against its release.

    Attributes:
        table: Which kind of row it is.
        subject: The dataset side: `cp in 1|2|3`, `chol >= 240`, `SCTID:414545008`, or empty.
        concept: The row's concept.
        label: The row's label.
        problem: None when the release agrees; otherwise how it disagrees.
    """

    table: Table
    subject: str
    concept: str
    label: str
    problem: str | None


def check(rows: Iterable[ConceptRow], ontologies: Mapping[str, Ontology]) -> tuple[Verdict, ...]:
    """Check each row, in order, against the ontology its concept's prefix names."""
    return tuple(_verdict(row, ontologies) for row in rows)


def _verdict(row: ConceptRow, ontologies: Mapping[str, Ontology]) -> Verdict:
    table, subject = _describe(row)
    return Verdict(
        table=table,
        subject=subject,
        concept=row.concept,
        label=row.label,
        problem=_problem(row, ontologies),
    )


def _describe(row: ConceptRow) -> tuple[Table, str]:
    if isinstance(row, FieldMapping):
        return "field", " and ".join(_clause(clause) for clause in row.when)
    if isinstance(row, CodeMapping):
        return "code", f"SCTID:{row.snomed}"
    return "out-of-scope", ""


def _clause(clause: Clause) -> str:
    if isinstance(clause, Codes):
        return f"{clause.field} in {'|'.join(str(code) for code in sorted(clause.codes))}"
    return f"{clause.field} >= {clause.value:g}"


def _problem(row: ConceptRow, ontologies: Mapping[str, Ontology]) -> str | None:
    prefix = row.concept.partition(":")[0]
    ontology = ontologies.get(prefix)
    if ontology is None:
        return f"no release for prefix {prefix}"
    where = f"{ontology.name} {ontology.version}"
    term = ontology.terms.get(row.concept)
    if term is None:
        return f"absent from {where}"
    if term.obsolete:
        return f"obsolete in {where}; replaced by {', '.join(term.replaced_by) or 'nothing'}"
    if term.name != row.label:
        return f"named '{term.name}' in {where}"
    if isinstance(row, CodeMapping):
        return _match_problem(row, ontology, where)
    return None


def _match_problem(row: CodeMapping, ontology: Ontology, where: str) -> str | None:
    match = ("skos:exactMatch", f"SCTID:{row.snomed}")
    if match not in ontology.terms[row.concept].matches:
        return f"{where} has no skos:exactMatch SCTID:{row.snomed} on {row.concept}"
    others = [
        curie
        for curie, term in ontology.terms.items()
        if curie != row.concept and not term.obsolete and match in term.matches
    ]
    if others:
        return f"skos:exactMatch SCTID:{row.snomed} is also on {', '.join(others)}"
    return None
