"""Check the table's concept rows against parsed releases: `python -m bko.ontology check`'s core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, assert_never

from bko.ontology.schema import (
    AtLeast,
    CheckedRow,
    Clause,
    CodeMapping,
    Codes,
    ConceptRow,
    FieldMapping,
    OutOfScope,
    UnmappedCode,
)
from bko.ontology.terms import Ontology, ancestors


type Table = Literal["field", "code", "out-of-scope", "excluded-code"]

EXCLUDED_PREFIX: Final = "MONDO"
"""The release an excluded code is checked against: the one code rows map into."""

type _Holders = Mapping[str, Mapping[str, tuple[str, ...]]]
"""Per release prefix, each wanted `SCTID:` object to the live terms that carry it."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """One concept row against its release.

    Attributes:
        table: Which kind of row it is.
        subject: The dataset side: `cp in 1|2|3`, `chol >= 240`, `SCTID:414545008`, or empty.
        concept: The row's concept; empty for an excluded code.
        label: The row's label.
        problem: None when the release agrees; otherwise how it disagrees.
    """

    table: Table
    subject: str
    concept: str
    label: str
    problem: str | None


def check(rows: Iterable[CheckedRow], ontologies: Mapping[str, Ontology]) -> tuple[Verdict, ...]:
    """Check each row, in order, against the ontology its concept's prefix names.

    An out-of-scope row also disagrees when its concept is an `is_a` ancestor of a concept another
    of `rows` maps, since the scope rule puts every such ancestor in scope. An excluded code
    disagrees when a live `EXCLUDED_PREFIX` concept carries its `skos:exactMatch`. Each release's
    `skos:exactMatch` holders are collected in one pass per call, however many code rows and
    excluded codes name its prefix.
    """
    rows = tuple(rows)
    mapped = frozenset(row.concept for row in rows if isinstance(row, (FieldMapping, CodeMapping)))
    wanted: dict[str, set[str]] = {}
    for row in rows:
        match row:
            case CodeMapping():
                wanted.setdefault(row.concept.partition(":")[0], set()).add(_target(row))
            case UnmappedCode():
                wanted.setdefault(EXCLUDED_PREFIX, set()).add(_target(row))
            case _:
                pass
    holders = {
        prefix: _exact_holders(ontologies[prefix], frozenset(targets))
        for prefix, targets in wanted.items()
        if prefix in ontologies
    }
    return tuple(_verdict(row, ontologies, mapped, holders) for row in rows)


def _verdict(
    row: CheckedRow, ontologies: Mapping[str, Ontology], mapped: frozenset[str], holders: _Holders
) -> Verdict:
    if isinstance(row, UnmappedCode):
        return Verdict(
            table="excluded-code",
            subject=_target(row),
            concept="",
            label=row.label,
            problem=_excluded_problem(row, ontologies, holders),
        )
    table, subject = _describe(row)
    return Verdict(
        table=table,
        subject=subject,
        concept=row.concept,
        label=row.label,
        problem=_problem(row, ontologies, mapped, holders),
    )


def _describe(row: ConceptRow) -> tuple[Table, str]:
    match row:
        case FieldMapping(when=when):
            return "field", " and ".join(_clause(clause) for clause in when)
        case CodeMapping():
            return "code", _target(row)
        case OutOfScope():
            return "out-of-scope", ""
        case _:
            assert_never(row)


def _clause(clause: Clause) -> str:
    match clause:
        case Codes(field=field, codes=codes):
            return f"{field} in {'|'.join(str(code) for code in sorted(codes))}"
        case AtLeast(field=field, value=value):
            return f"{field} >= {value:g}"
        case _:
            assert_never(clause)


def _problem(
    row: ConceptRow, ontologies: Mapping[str, Ontology], mapped: frozenset[str], holders: _Holders
) -> str | None:
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
    if term.name is None:
        return f"has no name in {where}"
    if term.name != row.label:
        return f"named '{term.name}' in {where}"
    match row:
        case CodeMapping():
            # `check` indexes every prefix a code row names that `ontologies` holds.
            return _match_problem(row, ontology, where, holders[prefix])
        case OutOfScope():
            return _scope_problem(row, ontology, where, mapped)
        case FieldMapping():
            return None
        case _:
            assert_never(row)


def _scope_problem(
    row: OutOfScope, ontology: Ontology, where: str, mapped: frozenset[str]
) -> str | None:
    below = sorted(
        curie
        for curie in mapped
        if curie in ontology.terms
        and row.concept in ancestors(ontology, curie, prefix=row.concept.partition(":")[0])
    )
    if below:
        return f"is_a ancestor of mapped {', '.join(below)} in {where}, so it is in scope"
    return None


def _target(row: CodeMapping | UnmappedCode) -> str:
    return f"SCTID:{row.snomed}"


def _excluded_problem(
    row: UnmappedCode, ontologies: Mapping[str, Ontology], holders: _Holders
) -> str | None:
    if EXCLUDED_PREFIX not in ontologies:
        return f"no release for prefix {EXCLUDED_PREFIX}"
    target = _target(row)
    found = holders[EXCLUDED_PREFIX].get(target, ())
    if found:
        return f"skos:exactMatch {target} is on {', '.join(found)}"
    return None


def _exact_holders(ontology: Ontology, targets: frozenset[str]) -> Mapping[str, tuple[str, ...]]:
    """For each of `targets`, the live terms with a `skos:exactMatch` to it, in term order."""
    holders: dict[str, list[str]] = {}
    for curie, term in ontology.terms.items():
        if term.obsolete:
            continue
        for predicate, target in term.matches:
            if predicate == "skos:exactMatch" and target in targets:
                curies = holders.setdefault(target, [])
                if not curies or curies[-1] != curie:
                    curies.append(curie)
    return {target: tuple(curies) for target, curies in holders.items()}


def _match_problem(
    row: CodeMapping, ontology: Ontology, where: str, holders: Mapping[str, tuple[str, ...]]
) -> str | None:
    target = _target(row)
    if ("skos:exactMatch", target) not in ontology.terms[row.concept].matches:
        return f"{where} has no skos:exactMatch {target} on {row.concept}"
    others = [curie for curie in holders.get(target, ()) if curie != row.concept]
    if others:
        return f"skos:exactMatch {target} is also on {', '.join(others)}"
    return None
