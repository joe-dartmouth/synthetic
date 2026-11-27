"""Build mapping rows from a frame's codebook and pinned releases, with no model.

ADR `research/automatic-mapping` records the route. A field whose codes encode ICD-9 diagnosis
tokens (`bko.datasets.diagnoses.code_number`), named in `codes`, maps each code by
`bko.ontology.codes`' rule with no curated row: the one live MONDO holder of its `ICD9:` xref under
`EXACT` that every other live exact holder descends from. A code MONDO links only under another
qualifier is a candidate state, its live holders the options; V and E codes, and codes whose live
exact holders do not nest, map nothing. Every other coded state, the label's included, is matched
by its codebook meaning: skipped when it has no word or its first word is a `NEGATIVE` word
(`negative`), replaced by its field's description (`generic_query`) when every word is a digit
string or a `GENERIC` word, then scored by token-set Jaccard against each live term's name and
EXACT synonyms (`tokens`); `bko.ontology.lexical` holds these rules and the term index. Options
rank by score, the sorted order of the release prefixes, a name over a synonym, then the smaller
curie, one per term; the first maps the state at `THRESHOLD` or above. Numeric columns stay
unmapped. No function reads a record: a frame's codebook is its only input from the data. This
module imports nothing from `bko.generate` or `bko.evaluate`.

    python -m bko.ontology.automatic lexical <dataset> [--data-dir <dir>]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np

from bko.core.logs import configure
from bko.datasets.diagnoses import UNKNOWN, canonical, code_text
from bko.datasets.errors import DatasetError
from bko.datasets.frames import FrameSpec
from bko.ontology.automatic_inputs import DATASETS, inputs
from bko.ontology.codes import (
    REASONS,
    concept_name,
    exact_holders,
    literal,
    spanning_holder,
    unmapped_reason,
)
from bko.ontology.icd9 import EXACT
from bko.ontology.lexical import GENERIC, Index, Option, generic_query, negative, tokens
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.ontology.automatic")

SOURCE: Final = "research/automatic-mapping"
"""Every automatic row's `source`: the ADR id."""

THRESHOLD: Final = Fraction(4, 5)
"""The least token-set Jaccard at which the best option maps a state."""

TOP: Final = 10
"""The most options a candidate state carries."""

NONE: Final = "none"
"""The verdict that maps a candidate state to no concept."""

XREF_PREFIX: Final = "MONDO"
"""The release whose `ICD9:` xrefs map a code field."""

_XREF: Final = "ICD9:"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A coded state and its options, in rank order.

    Attributes:
        field: The column, the label's included.
        code: The state's code.
        meaning: The codebook meaning.
        query: The text matched: the meaning, or a generic state's `generic_query` words.
        options: At most `TOP` options.
    """

    field: str
    code: int
    meaning: str
    query: str
    options: tuple[Option, ...]

    @property
    def key(self) -> str:
        """The verdict table's key, `<field>=<code>`."""
        return f"{self.field}={self.code}"


class Survey(NamedTuple):
    """What the builder finds in one frame.

    Attributes:
        rows: The rows, sorted by (field, code, concept).
        candidates: Each state with an option, sorted by (field, code).
        dropped: Each of `bko.ontology.codes.REASONS` and how many code-field states it leaves
            with no row and no candidate.
    """

    rows: tuple[FieldMapping, ...]
    candidates: tuple[Candidate, ...]
    dropped: Mapping[str, int]


def candidates(
    frame: FrameSpec, ontologies: Mapping[str, Ontology], *, codes: Sequence[str] = ()
) -> tuple[Candidate, ...]:
    """Each state the judge is asked about, sorted by (field, code): one with an option.

    Raises:
        ValueError: As `build_lexical`.
    """
    return survey(frame, ontologies, codes).candidates


def build_lexical(
    frame: FrameSpec, ontologies: Mapping[str, Ontology], *, codes: Sequence[str] = ()
) -> tuple[FieldMapping, ...]:
    """The `auto-lexical` rows of `frame` over `ontologies`, keyed by release prefix.

    Args:
        frame: The frame whose codebook is read.
        ontologies: Each release prefix's parsed release.
        codes: The coded fields whose codes encode ICD-9 diagnosis tokens.

    Returns:
        The cross-reference rows and the label rows, sorted by (field, code, concept).

    Raises:
        ValueError: If a field of `codes` is not a coded field, or `codes` is given and
            `ontologies` lacks `XREF_PREFIX`.
    """
    return survey(frame, ontologies, codes).rows


def rows_from_verdicts(
    frame: FrameSpec, verdicts: Mapping[str, str], labels: Mapping[str, str]
) -> tuple[FieldMapping, ...]:
    """The rows of a verdict table: one per state whose verdict is not `NONE`.

    Args:
        frame: The frame whose states the keys name.
        verdicts: Each `<field>=<code>` state's concept, or `NONE`.
        labels: Each verdict concept's name in its release.

    Returns:
        The rows, sorted by (field, code, concept).

    Raises:
        ValueError: If a key is not `<field>=<code>`, names a field that is not coded or a code
            not in its codebook, or a concept has no label.
    """
    rows: list[FieldMapping] = []
    for key, concept in verdicts.items():
        field, sep, text = key.partition("=")
        if not sep or not text.isdigit():
            msg = f"verdict key {key!r} is not <field>=<code>"
            raise ValueError(msg)
        meanings = _meanings(frame, field)
        if meanings is None:
            msg = f"verdict key {key!r}: {field} is not a coded field of {frame.name}"
            raise ValueError(msg)
        code = int(text)
        if code not in meanings:
            msg = f"verdict key {key!r}: {code} is not a code of {field}"
            raise ValueError(msg)
        if concept == NONE:
            continue
        if concept not in labels:
            msg = f"verdict {key}={concept}: {concept} has no label"
            raise ValueError(msg)
        limits = f'{field} state {code} ("{meanings[code]}") is {concept} by the verdict table'
        rows.append(_row(field, code, concept, labels[concept], limits))
    return tuple(sorted(rows, key=_order))


def render(rows: Sequence[FieldMapping]) -> str:
    """The module text holding `rows` as `ROWS`, one tuple per line between `# fmt` markers.

    Raises:
        ValueError: If a row is not one `Codes` clause of one code.
    """
    lines = [_HEADER, "# fmt: off\n"]
    lines.append("_ROWS: Final[tuple[tuple[str, int, str, str, str, str], ...]] = (\n")
    for row in rows:
        field, code = state(row)
        texts = (row.concept, row.label, row.source, row.limits)
        lines.append(f"    ({literal(field)}, {code}, {', '.join(map(literal, texts))}),\n")
    lines.extend((")\n# fmt: on\n", _FOOTER))
    return "".join(lines)


def permuted(rows: Sequence[FieldMapping], *, seed: int) -> tuple[FieldMapping, ...]:
    """`rows` with each concept and its label moved to another of its release prefix's concepts.

    The distinct concepts of each prefix, sorted, are deranged by `numpy.random.default_rng(seed)`,
    prefixes in sorted order: a prefix's permutation is redrawn until no concept keeps its place,
    so only a prefix's lone concept stays (`unmovable`). Each row keeps its clauses, source and
    limits.
    """
    labels = {row.concept: row.label for row in rows}
    groups = _prefix_groups(labels)
    rng = np.random.default_rng(seed)
    moved: dict[str, str] = {}
    for prefix in sorted(groups):
        order = groups[prefix]
        moved.update(zip(order, (order[i] for i in _derangement(rng, len(order))), strict=True))
    return tuple(
        dataclasses.replace(row, concept=moved[row.concept], label=labels[moved[row.concept]])
        for row in rows
    )


def unmovable(rows: Sequence[FieldMapping]) -> int:
    """How many of `rows` `permuted` leaves on their concept.

    They are the rows whose concept is the only one of its release prefix among `rows`.
    """
    groups = _prefix_groups({row.concept: row.label for row in rows})
    alone = {group[0] for group in groups.values() if len(group) == 1}
    return sum(row.concept in alone for row in rows)


def _prefix_groups(concepts: Mapping[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for concept in sorted(concepts):
        groups.setdefault(concept.partition(":")[0], []).append(concept)
    return groups


def _derangement(rng: np.random.Generator, n: int) -> list[int]:
    """A permutation of `range(n)` with no fixed point, redrawn until found; identity below 2."""
    while True:
        drawn = [int(i) for i in rng.permutation(n)]
        if n < 2 or all(i != j for i, j in enumerate(drawn)):
            return drawn


def survey(
    frame: FrameSpec, ontologies: Mapping[str, Ontology], codes: Sequence[str] = ()
) -> Survey:
    """The rows, the candidates and the dropped code-field states of `frame`.

    Raises:
        ValueError: As `build_lexical`.
    """
    for field in codes:
        if field not in frame.categorical:
            msg = f"{field} is not a coded field of {frame.name}"
            raise ValueError(msg)
    if codes and XREF_PREFIX not in ontologies:
        msg = f"code fields {list(codes)} need the {XREF_PREFIX} release"
        raise ValueError(msg)
    rows, found, dropped = _code_states(frame, ontologies, codes)
    index = Index(ontologies)
    for field in (*frame.categorical, frame.label):
        if field in codes:
            continue
        meanings = _meanings(frame, field) or {}
        for code in sorted(meanings):
            meaning = meanings[code]
            said = tokens(meaning)
            if not said or negative(meaning):
                continue
            generic = all(word.isdigit() or word in GENERIC for word in said)
            query = " ".join(generic_query(frame.descriptions[field])) if generic else meaning
            options = index.options(tokens(query), TOP)
            if not options:
                continue
            found.append(Candidate(field, code, meaning, query, options))
            best = options[0]
            if best.score is not None and best.score >= THRESHOLD:
                limits = (
                    f'{field} state {code} ("{meaning}") matched the {best.kind} "{best.text}" '
                    f"at token-set Jaccard {best.score}"
                )
                rows.append(_row(field, code, best.curie, best.label, limits))
    ordered = tuple(sorted(found, key=lambda c: (c.field, c.code)))
    return Survey(tuple(sorted(rows, key=_order)), ordered, dropped)


def _code_states(
    frame: FrameSpec, ontologies: Mapping[str, Ontology], codes: Sequence[str]
) -> tuple[list[FieldMapping], list[Candidate], dict[str, int]]:
    dropped = dict.fromkeys(REASONS, 0)
    if not codes:
        return [], [], dropped
    mondo = ontologies[XREF_PREFIX]
    exact, related, dead = exact_holders(mondo, _XREF)
    weaker: dict[str, list[str]] = {}
    for curie, term in mondo.terms.items():
        if term.obsolete or term.name is None:
            continue
        for xref in term.xrefs:
            if xref.startswith(_XREF) and EXACT not in term.xref_sources(xref):
                weaker.setdefault(xref.removeprefix(_XREF), []).append(curie)
    rows: list[FieldMapping] = []
    found: list[Candidate] = []
    for field in codes:
        for code, meaning in sorted(frame.codebook[field].items()):
            if code == UNKNOWN:
                continue
            token = canonical(code_text(code))
            holders = exact.get(token, ())
            concept = spanning_holder(mondo, holders) if holders else None
            if token[0] in "VE":
                dropped[f"{token[0]} code"] += 1
            elif concept is not None:
                limits = f"ICD-9 {token} in {field}; {concept} carries its xref under {EXACT}"
                rows.append(_row(field, code, concept, concept_name(mondo, concept), limits))
            elif holders:
                dropped["holders not nested"] += 1
            elif token in weaker:
                options = tuple(
                    Option(c, concept_name(mondo, c), None, "xref", f"{_XREF}{token}")
                    for c in sorted(weaker[token])[:TOP]
                )
                found.append(Candidate(field, code, meaning, f"{_XREF}{token}", options))
            else:
                dropped[unmapped_reason(token, related, dead)] += 1
    return rows, found, dropped


def _meanings(frame: FrameSpec, field: str) -> Mapping[int, str] | None:
    if field == frame.label:
        return frame.label_meanings
    return frame.codebook.get(field) if field in frame.categorical else None


def _row(field: str, code: int, concept: str, label: str, limits: str) -> FieldMapping:
    return FieldMapping(
        when=(Codes(field, frozenset({code})),),
        concept=concept,
        label=label,
        source=SOURCE,
        limits=limits,
    )


def state(row: FieldMapping) -> tuple[str, int]:
    """The (field, code) of an automatic row: one `Codes` clause of one code.

    Raises:
        ValueError: If `row` is not such a row.
    """
    match row.when:
        case (Codes(field=field, codes=codes),) if len(codes) == 1:
            return field, next(iter(codes))
    msg = f"{row.concept}: an automatic row is one Codes clause of one code, not {row.when!r}"
    raise ValueError(msg)


def _order(row: FieldMapping) -> tuple[str, int, str]:
    return (*state(row), row.concept)


_HEADER: Final = '''"""Automatic mapping rows that `python -m bko.ontology.automatic lexical` wrote.

Regenerate this module rather than editing it (ADR `research/automatic-mapping`). `_ROWS` holds
one (field, code, concept, label, source, limits) tuple per line.
"""

from __future__ import annotations

from typing import Final

from bko.ontology.schema import Codes, FieldMapping


'''

_FOOTER: Final = '''
ROWS: Final[tuple[FieldMapping, ...]] = tuple(
    FieldMapping(
        when=(Codes(field, frozenset({code})),),
        concept=concept,
        label=label,
        source=source,
        limits=limits,
    )
    for field, code, concept, label, source, limits in _ROWS
)
"""The rows, sorted by (field, code, concept)."""
'''


def main(argv: Sequence[str] | None = None) -> int:
    """Write `lexical <dataset>`'s module text to stdout as UTF-8 bytes.

    Logs a `lexical dropped` line per reason for a dataset with code fields, and a closing
    `lexical` line whose sha256 is that of the bytes written.

    Returns:
        0 done; a `DatasetError`'s exit code when a snapshot cannot be read; 2 another I/O
        failure, or code fields the builder refuses.
    """
    parser = argparse.ArgumentParser(prog="python -m bko.ontology.automatic", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lexical = commands.add_parser("lexical", help="build a dataset's auto-lexical rows")
    lexical.add_argument("dataset", choices=DATASETS)
    lexical.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        data_dir = None if args.data_dir is None else args.data_dir.resolve()
        frame, ontologies, code_fields = inputs(args.dataset, data_dir)
        found = survey(frame, ontologies, code_fields)
    except DatasetError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    data = render(found.rows).encode()
    if code_fields:
        for reason, count in found.dropped.items():
            logger.info("lexical dropped reason=%s states=%d", json.dumps(reason), count)
    logger.info(
        "lexical dataset=%s rows=%d candidates=%d sha256=%s",
        args.dataset,
        len(found.rows),
        len(found.candidates),
        hashlib.sha256(data).hexdigest(),
    )
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    configure()
    sys.exit(main())
