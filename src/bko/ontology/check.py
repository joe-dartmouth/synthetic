"""Check the table's rows against parsed releases: `python -m bko.ontology check`'s core.

The rows are the concept rows, the excluded Synthea codes and the excluded ICD-9 categories
(`bko.ontology.mapping.CHECKED_ROWS`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, assert_never

from bko.ontology.icd9 import Icd9Mapping, LabMapping, UnmappedIcd9
from bko.ontology.schema import (
    AtLeast,
    CheckedRow,
    Clause,
    CodedRow,
    CodeMapping,
    Codes,
    ConceptRow,
    FieldMapping,
    OutOfScope,
    UnmappedCode,
)
from bko.ontology.terms import Ontology, ancestors


type Table = Literal["field", "code", "out-of-scope", "excluded-code", "icd9", "excluded-icd9"]

EXCLUDED_PREFIX: Final = "MONDO"
"""The release an excluded code or ICD-9 category is checked against: the one code rows map
into."""

_EXACT: Final = "skos:exactMatch"
_XREF: Final = "xref"

type _Key = tuple[str, str]
"""What a term carries: `(skos:exactMatch, <SCTID: object>)` or `(xref, <ICD9: id>)`."""

type _Holders = Mapping[str, Mapping[_Key, tuple[str, ...]]]
"""Per release prefix, each wanted key to the live terms that carry it."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """One checked row against its release: a concept row or an excluded code.

    Attributes:
        table: Which kind of row it is.
        subject: The dataset side: `cp in 1|2|3`, `chol >= 240`, `A1Cresult in >7|>8`,
            `SCTID:414545008` (a code row or an excluded code), `ICD9:428` (an ICD-9 row or an
            excluded category), or empty.
        concept: The row's concept; empty for an excluded code or category.
        label: The row's label; for an excluded code or category, its description.
        problem: None when the release agrees; otherwise how it disagrees.
    """

    table: Table
    subject: str
    concept: str
    label: str
    problem: str | None


def check(rows: Iterable[CheckedRow], ontologies: Mapping[str, Ontology]) -> tuple[Verdict, ...]:
    """Check each row, in order, against the ontology its concept's prefix names.

    An excluded code or ICD-9 category names no concept, so it is checked against the
    `EXCLUDED_PREFIX` release.

    An out-of-scope row also disagrees when its concept is an `is_a` ancestor of a concept another
    of `rows` maps, since the scope rule puts every such ancestor in scope. An excluded code
    disagrees when a live `EXCLUDED_PREFIX` concept carries its `skos:exactMatch`, and an excluded
    ICD-9 category when one carries its xref. An ICD-9 row disagrees when its concept lacks its
    xref or the release's live holders of the xref differ from the row's declared ones. Each
    release's holders are collected in one pass per call, however many code rows, ICD-9 rows and
    excluded codes or categories name its prefix.
    """
    rows = tuple(rows)
    mapped = frozenset(
        row.concept
        for row in rows
        if isinstance(row, FieldMapping | CodeMapping | Icd9Mapping | LabMapping)
    )
    wanted: dict[str, set[_Key]] = {}
    for row in rows:
        match row:
            case CodeMapping() | Icd9Mapping():
                wanted.setdefault(_prefix(row), set()).add(_key(row))
            case UnmappedCode() | UnmappedIcd9():
                wanted.setdefault(EXCLUDED_PREFIX, set()).add(_key(row))
    holders = {
        prefix: _holders(ontologies[prefix], frozenset(keys))
        for prefix, keys in wanted.items()
        if prefix in ontologies
    }
    return tuple(_verdict(row, ontologies, mapped, holders) for row in rows)


def _verdict(
    row: CheckedRow, ontologies: Mapping[str, Ontology], mapped: frozenset[str], holders: _Holders
) -> Verdict:
    match row:
        case UnmappedCode() | UnmappedIcd9():
            return Verdict(
                table="excluded-code" if isinstance(row, UnmappedCode) else "excluded-icd9",
                subject=_key(row)[1],
                concept="",
                label=row.label,
                problem=_excluded_problem(row, ontologies, holders),
            )
        case FieldMapping() | CodeMapping() | OutOfScope() | Icd9Mapping() | LabMapping():
            table, subject = _describe(row)
            return Verdict(
                table=table,
                subject=subject,
                concept=row.concept,
                label=row.label,
                problem=_problem(row, ontologies, mapped, holders),
            )
        case _:
            assert_never(row)


def _describe(row: ConceptRow) -> tuple[Table, str]:
    match row:
        case FieldMapping(when=when):
            return "field", " and ".join(_clause(clause) for clause in when)
        case CodeMapping():
            return "code", _target(row)
        case OutOfScope():
            return "out-of-scope", ""
        case Icd9Mapping(xref=xref):
            return "icd9", xref
        case LabMapping(field=field, states=states):
            return "field", f"{field} in {'|'.join(sorted(states))}"
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
    prefix = _prefix(row)
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
    return _kind_problem(row, ontology, where, mapped, holders)


def _kind_problem(
    row: ConceptRow,
    ontology: Ontology,
    where: str,
    mapped: frozenset[str],
    holders: _Holders,
) -> str | None:
    """What a live, correctly named concept row's own kind of check finds."""
    # `check` indexes every prefix a code or ICD-9 row names that `ontologies` holds.
    match row:
        case CodeMapping():
            return _match_problem(row, ontology, where, holders[_prefix(row)])
        case Icd9Mapping():
            return _xref_problem(row, ontology, where, holders[_prefix(row)])
        case OutOfScope():
            return _scope_problem(row, ontology, where, mapped)
        case FieldMapping() | LabMapping():
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


def _prefix(row: ConceptRow) -> str:
    return row.concept.partition(":")[0]


def _target(row: CodedRow) -> str:
    return f"SCTID:{row.snomed}"


def _key(row: CodedRow | Icd9Mapping | UnmappedIcd9) -> _Key:
    match row:
        case CodeMapping() | UnmappedCode():
            return _EXACT, _target(row)
        case Icd9Mapping(xref=xref) | UnmappedIcd9(xref=xref):
            return _XREF, xref
        case _:
            assert_never(row)


def _excluded_problem(
    row: UnmappedCode | UnmappedIcd9, ontologies: Mapping[str, Ontology], holders: _Holders
) -> str | None:
    if EXCLUDED_PREFIX not in ontologies:
        return f"no release for prefix {EXCLUDED_PREFIX}"
    key = _key(row)
    # `check` indexes `EXCLUDED_PREFIX` whenever it checks an excluded row and holds its release.
    found = holders[EXCLUDED_PREFIX].get(key, ())
    if found:
        return f"{' '.join(key)} is on {', '.join(found)}"
    return None


def _holders(ontology: Ontology, keys: frozenset[_Key]) -> Mapping[_Key, tuple[str, ...]]:
    """For each of `keys`, the live terms that carry it, in term order.

    A term carries `(skos:exactMatch, object)` through its matches and `(xref, id)` through its
    xref ids, whatever the release's qualifier on the xref.
    """
    holders: dict[_Key, list[str]] = {}
    for curie, term in ontology.terms.items():
        if term.obsolete:
            continue
        for key in (*term.matches, *((_XREF, xref) for xref in term.xrefs)):
            if key in keys:
                curies = holders.setdefault(key, [])
                if not curies or curies[-1] != curie:
                    curies.append(curie)
    return {key: tuple(curies) for key, curies in holders.items()}


def _match_problem(
    row: CodeMapping, ontology: Ontology, where: str, holders: Mapping[_Key, tuple[str, ...]]
) -> str | None:
    key = _key(row)
    target = key[1]
    if key not in ontology.terms[row.concept].matches:
        return f"{where} has no skos:exactMatch {target} on {row.concept}"
    others = [curie for curie in holders.get(key, ()) if curie != row.concept]
    if others:
        return f"skos:exactMatch {target} is also on {', '.join(others)}"
    return None


def _xref_problem(
    row: Icd9Mapping, ontology: Ontology, where: str, holders: Mapping[_Key, tuple[str, ...]]
) -> str | None:
    if row.xref not in ontology.terms[row.concept].xrefs:
        return f"{where} has no xref {row.xref} on {row.concept}"
    live = holders.get(_key(row), ())
    if frozenset(live) != row.holders:
        return (
            f"xref {row.xref} is on {', '.join(live)}; "
            f"the row declares {', '.join(sorted(row.holders))}"
        )
    return None
