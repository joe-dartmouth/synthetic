"""The label route's word rules and term index, for `bko.ontology.automatic`.

A codebook meaning and a term's name or EXACT synonym are compared as token sets (`tokens`). A
meaning is skipped when its first word is a `NEGATIVE` word (`negative`); a generic one is
matched by its field's description instead (`generic_query`). `Index` scores every live term of
each release by token-set Jaccard (ADR `research/automatic-mapping`, Decision 3).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from operator import itemgetter
from typing import Final

from bko.ontology.terms import Ontology


_KINDS: Final = ("name", "synonym")
_LEGEND: Final = re.compile(r"\([^()]*\)")

NEGATIVE: Final = frozenset(
    {"absent", "no", "none", "not", "norm", "normal", "false", "negative", "without"}
)
"""Words that make a meaning a negative state, which is skipped, when its first word is one."""

GENERIC: Final = frozenset(
    {"present", "yes", "true", "positive", "up", "down", "steady", "ch", "changed", "high", "low"}
)
"""Words that say nothing of a concept: a meaning of these and digits reads its description."""

STOPWORDS: Final = frozenset(
    {
        *("a", "among", "an", "and", "any", "as", "at", "by", "during", "for", "from", "in"),
        *("its", "of", "on", "or", "s", "the", "to", "with", "within"),
    }
)
"""Function words a generic state's description query drops (`generic_query`); `s` is a
possessive's tail, split off at its apostrophe."""


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


def words(text: str) -> tuple[str, ...]:
    """`text` NFKC-normalized, casefolded, NFKC-normalized again and split into words, in order.

    A word is a run of alphanumeric characters and combining marks; every other character splits.
    """
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())
    kept = (c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in folded)
    return tuple("".join(kept).split())


def tokens(text: str) -> frozenset[str]:
    """The set of `text`'s `words`."""
    return frozenset(words(text))


def negative(meaning: str) -> bool:
    """Whether `meaning` is a negative state: its first word is a `NEGATIVE` word.

    A negative word later in a meaning, as in "diabetes without complications", qualifies a
    concept rather than denying it, so it does not make the state negative.
    """
    found = words(meaning)
    return bool(found) and found[0] in NEGATIVE


def generic_query(description: str) -> tuple[str, ...]:
    """The words a generic state is matched by, read from its field's description.

    The description's parenthetical legend and every `GENERIC`, `NEGATIVE` or `STOPWORDS` word
    are dropped.
    """
    dropped = GENERIC | NEGATIVE | STOPWORDS
    return tuple(word for word in words(_LEGEND.sub(" ", description)) if word not in dropped)


@dataclass(frozen=True, slots=True)
class _Target:
    words: frozenset[str]
    rank: tuple[int, int, str]
    curie: str
    label: str
    kind: str
    text: str


class Index:
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
                    found = tokens(text)
                    if found:
                        self._add(
                            _Target(
                                found, (rank, kind, curie), curie, term.name, _KINDS[kind], text
                            )
                        )

    def _add(self, target: _Target) -> None:
        for word in target.words:
            self.by_word.setdefault(word, []).append(len(self.targets))
        self.targets.append(target)

    def options(self, query: frozenset[str], limit: int) -> tuple[Option, ...]:
        """The best `limit` options for `query`, one per term, in rank order."""
        best: dict[str, tuple[tuple[Fraction, tuple[int, int, str]], _Target]] = {}
        for i in sorted({i for word in query for i in self.by_word.get(word, ())}):
            target = self.targets[i]
            shared = len(query & target.words)
            score = Fraction(shared, len(query) + len(target.words) - shared)
            key = (-score, target.rank)
            if target.curie not in best or key < best[target.curie][0]:
                best[target.curie] = (key, target)
        ranked = sorted(best.values(), key=itemgetter(0))[:limit]
        return tuple(Option(t.curie, t.label, -key[0], t.kind, t.text) for key, t in ranked)
