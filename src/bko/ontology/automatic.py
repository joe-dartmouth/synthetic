"""Build mapping rows from a frame's codebook and pinned releases, with no model.

ADR `research/automatic-mapping` records the route. A field whose codes encode ICD-9 diagnosis
tokens (`bko.datasets.diagnoses.code_number`), named in `codes`, maps each code by
`bko.ontology.codes`' rule with no curated row: the one live MONDO holder of its `ICD9:` xref under
`EXACT` that every other live exact holder descends from. A code MONDO links only under another
qualifier is a candidate state, its live holders the options. Every other coded state, the label's
included, is matched by its codebook meaning: skipped when a token is a `NEGATIVE` word or it has
no token, replaced by the field's description when every token is a digit string or a `GENERIC`
word, then scored by token-set Jaccard against each live term's name and EXACT synonyms
(`tokens`). Options rank by score, the sorted order of the release prefixes, a name over a
synonym, then the smaller curie, one per term; the first maps the state at `THRESHOLD` or above.
Numeric columns stay unmapped. No function reads a record: a frame's codebook is its only input
from the data. This module imports nothing from `bko.generate` or `bko.evaluate`.

    python -m bko.ontology.automatic lexical <dataset> [--data-dir <dir>]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from operator import itemgetter
from pathlib import Path
from typing import Final

import numpy as np

from bko.core.logs import configure
from bko.datasets.diagnoses import UNKNOWN, canonical, code_text
from bko.datasets.errors import DatasetError
from bko.datasets.frames import FrameSpec
from bko.ontology.automatic_inputs import DATASETS, inputs
from bko.ontology.codes import exact_holders, spanning_holder
from bko.ontology.icd9 import EXACT
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

NEGATIVE: Final = frozenset(
    {"absent", "no", "none", "not", "norm", "normal", "false", "negative", "without"}
)
"""Words that make a meaning a negative state, which is skipped."""

GENERIC: Final = frozenset(
    {"present", "yes", "true", "positive", "up", "down", "steady", "ch", "changed", "high", "low"}
)
"""Words that say nothing of a concept: a meaning of these and digits reads its description."""

XREF_PREFIX: Final = "MONDO"
"""The release whose `ICD9:` xrefs map a code field."""

_XREF: Final = "ICD9:"
_KINDS: Final = ("name", "synonym")


@dataclass(frozen=True, slots=True)
class Option:
    """One concept a candidate state may show.

    Attributes:
        curie: The term's id.
        label: The term's name.
        score: The token-set Jaccard of its best target; None for a cross-reference option.
        kind: `name`, `synonym` or `xref`: what the score was read from.
        text: The name or synonym matched, or the xref.
    """

    curie: str
    label: str
    score: Fraction | None
    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """A coded state and its options, in rank order.

    Attributes:
        field: The column, the label's included.
        code: The state's code.
        meaning: The codebook meaning.
        query: The text matched: the meaning, or the field's description for a generic one.
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


def tokens(text: str) -> frozenset[str]:
    """`text` NFKC-normalized, casefolded, its non-alphanumeric characters spaces, as a set."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return frozenset("".join(c if c.isalnum() else " " for c in folded).split())


def candidates(
    frame: FrameSpec, ontologies: Mapping[str, Ontology], *, codes: Sequence[str] = ()
) -> tuple[Candidate, ...]:
    """Each state the judge is asked about, sorted by (field, code): one with an option.

    Raises:
        ValueError: As `build_lexical`.
    """
    return _survey(frame, ontologies, codes)[1]


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
    return _survey(frame, ontologies, codes)[0]


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
    lines.append("_ROWS: Final[tuple[tuple[str, int, str, str, str], ...]] = (\n")
    for row in rows:
        field, code = _state(row)
        values = ", ".join(_literal(text) for text in (row.concept, row.label, row.limits))
        lines.append(f"    ({_literal(field)}, {code}, {values}),\n")
    lines.extend((")\n# fmt: on\n", _FOOTER))
    return "".join(lines)


def precision_recall(
    built: Sequence[FieldMapping], hand: Sequence[FieldMapping]
) -> tuple[Fraction | None, Fraction | None]:
    """The share of `built`'s atoms `hand` holds, and of `hand`'s that `built` holds.

    An atom is (field, code, concept) for each code of a row of one `Codes` clause; any other row
    is one atom of its clauses. Each share is None when its denominator is empty.
    """
    mine, theirs = atoms(built), atoms(hand)
    common = len(mine & theirs)
    return (
        Fraction(common, len(mine)) if mine else None,
        Fraction(common, len(theirs)) if theirs else None,
    )


def atoms(rows: Sequence[FieldMapping]) -> frozenset[tuple[str, str, str]]:
    """Each row's (field, code, concept) atoms, as `precision_recall` counts them."""
    found: set[tuple[str, str, str]] = set()
    for row in rows:
        match row.when:
            case (Codes() as clause,):
                found.update((clause.field, str(code), row.concept) for code in clause.codes)
            case _:
                found.add((" & ".join(repr(clause) for clause in row.when), "", row.concept))
    return frozenset(found)


def permuted(rows: Sequence[FieldMapping], *, seed: int) -> tuple[FieldMapping, ...]:
    """`rows` with each concept and its label moved to another of its release prefix's concepts.

    The distinct concepts of each prefix, sorted, are permuted by `numpy.random.default_rng(seed)`,
    prefixes in sorted order; each row keeps its clauses, source and limits.
    """
    labels = {row.concept: row.label for row in rows}
    groups: dict[str, list[str]] = {}
    for concept in sorted(labels):
        groups.setdefault(concept.partition(":")[0], []).append(concept)
    rng = np.random.default_rng(seed)
    moved: dict[str, str] = {}
    for prefix in sorted(groups):
        order = groups[prefix]
        moved.update(zip(order, (order[i] for i in rng.permutation(len(order))), strict=True))
    return tuple(
        dataclasses.replace(row, concept=moved[row.concept], label=labels[moved[row.concept]])
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class _Target:
    words: frozenset[str]
    rank: tuple[int, int, str]
    curie: str
    label: str
    kind: str
    text: str


class _Index:
    """Every live term's name and EXACT synonyms, by token."""

    def __init__(self, ontologies: Mapping[str, Ontology]) -> None:
        self.targets: list[_Target] = []
        self.by_word: dict[str, list[int]] = {}
        for rank, prefix in enumerate(sorted(ontologies)):
            for curie, term in ontologies[prefix].terms.items():
                if term.obsolete or term.name is None or not curie.startswith(f"{prefix}:"):
                    continue
                texts = ((0, term.name), *((1, text) for text in term.exact_synonyms()))
                for kind, text in texts:
                    words = tokens(text)
                    if words:
                        self._add(
                            _Target(
                                words, (rank, kind, curie), curie, term.name, _KINDS[kind], text
                            )
                        )

    def _add(self, target: _Target) -> None:
        for word in target.words:
            self.by_word.setdefault(word, []).append(len(self.targets))
        self.targets.append(target)

    def options(self, query: frozenset[str]) -> tuple[Option, ...]:
        best: dict[str, tuple[tuple[Fraction, tuple[int, int, str]], _Target]] = {}
        for i in sorted({i for word in query for i in self.by_word.get(word, ())}):
            target = self.targets[i]
            shared = len(query & target.words)
            score = Fraction(shared, len(query) + len(target.words) - shared)
            key = (-score, target.rank)
            if target.curie not in best or key < best[target.curie][0]:
                best[target.curie] = (key, target)
        ranked = sorted(best.values(), key=itemgetter(0))[:TOP]
        return tuple(Option(t.curie, t.label, -key[0], t.kind, t.text) for key, t in ranked)


def _survey(
    frame: FrameSpec, ontologies: Mapping[str, Ontology], codes: Sequence[str]
) -> tuple[tuple[FieldMapping, ...], tuple[Candidate, ...]]:
    for field in codes:
        if field not in frame.categorical:
            msg = f"{field} is not a coded field of {frame.name}"
            raise ValueError(msg)
    if codes and XREF_PREFIX not in ontologies:
        msg = f"code fields {list(codes)} need the {XREF_PREFIX} release"
        raise ValueError(msg)
    rows, found = _code_states(frame, ontologies, codes) if codes else ([], [])
    index = _Index(ontologies)
    for field in (*frame.categorical, frame.label):
        if field in codes:
            continue
        meanings = _meanings(frame, field) or {}
        for code in sorted(meanings):
            meaning = meanings[code]
            words = tokens(meaning)
            if not words or words & NEGATIVE:
                continue
            generic = all(word.isdigit() or word in GENERIC for word in words)
            query = frame.descriptions[field] if generic else meaning
            options = index.options(tokens(query))
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
    return tuple(sorted(rows, key=_order)), tuple(sorted(found, key=lambda c: (c.field, c.code)))


def _code_states(
    frame: FrameSpec, ontologies: Mapping[str, Ontology], codes: Sequence[str]
) -> tuple[list[FieldMapping], list[Candidate]]:
    mondo = ontologies[XREF_PREFIX]
    exact, _, _ = exact_holders(mondo, _XREF)
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
            token = canonical(code_text(code)) if code != UNKNOWN else ""
            if not token or token[0] in "VE":
                continue
            holders = exact.get(token, ())
            concept = spanning_holder(mondo, holders) if holders else None
            if concept is not None:
                limits = f"ICD-9 {token} in {field}; {concept} carries its xref under {EXACT}"
                rows.append(_row(field, code, concept, _name(mondo, concept), limits))
            elif not holders and token in weaker:
                options = tuple(
                    Option(c, _name(mondo, c), None, "xref", f"{_XREF}{token}")
                    for c in sorted(weaker[token])[:TOP]
                )
                found.append(Candidate(field, code, meaning, f"{_XREF}{token}", options))
    return rows, found


def _name(ontology: Ontology, curie: str) -> str:
    name = ontology.terms[curie].name
    if name is None:
        msg = f"{curie} has no name in {ontology.name} {ontology.version}"
        raise ValueError(msg)
    return name


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


def _state(row: FieldMapping) -> tuple[str, int]:
    match row.when:
        case (Codes(field=field, codes=codes),) if len(codes) == 1:
            return field, next(iter(codes))
    msg = f"{row.concept}: an automatic row is one Codes clause of one code, not {row.when!r}"
    raise ValueError(msg)


def _order(row: FieldMapping) -> tuple[str, int, str]:
    return (*_state(row), row.concept)


def _literal(text: str) -> str:
    """`text` as a double-quoted Python string literal, a control character escaped."""
    return json.dumps(text, ensure_ascii=False)


_HEADER: Final = '''"""Automatic mapping rows that `python -m bko.ontology.automatic lexical` wrote.

Regenerate this module rather than editing it (ADR `research/automatic-mapping`). `_ROWS` holds
one (field, code, concept, label, limits) tuple per line.
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
        source="research/automatic-mapping",
        limits=limits,
    )
    for field, code, concept, label, limits in _ROWS
)
"""The rows, sorted by (field, code, concept)."""
'''


def main(argv: Sequence[str] | None = None) -> int:
    """Write `lexical <dataset>`'s module text to stdout.

    Returns:
        0 done; a `DatasetError`'s exit code when a snapshot cannot be read; 2 another I/O
        failure.
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
    except DatasetError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    rows, found = _survey(frame, ontologies, code_fields)
    text = render(rows)
    logger.info(
        "lexical dataset=%s rows=%d candidates=%d sha256=%s",
        args.dataset,
        len(rows),
        len(found),
        hashlib.sha256(text.encode()).hexdigest(),
    )
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    configure()
    sys.exit(main())
