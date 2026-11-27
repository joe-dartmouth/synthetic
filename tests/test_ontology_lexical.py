"""The label route's word rules and term index (`bko.ontology.lexical`), on invented releases."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import pytest

from bko.ontology.automatic import TOP, build_lexical, candidates, state
from bko.ontology.lexical import Index, generic_query, negative, tokens, words
from bko.ontology.obo import parse_obo
from bko.ontology.terms import Ontology
from tests._automatic_fixtures import FRAME
from tests._ontology_fixtures import write_obo


HEAD = "format-version: 1.2\ndata-version: releases/2026-10-06\nontology: {name}\n\n"


def _release(
    tmp_path: Path, name: str, stanzas: Sequence[tuple[str, str, tuple[str, ...]]]
) -> Ontology:
    """A release of (curie, name, EXACT synonyms) stanzas, in the order given."""
    text = HEAD.format(name=name) + "".join(
        f"[Term]\nid: {curie}\nname: {label}\n"
        + "".join(f'synonym: "{s}" EXACT []\n' for s in synonyms)
        + "\n"
        for curie, label, synonyms in stanzas
    )
    return parse_obo(write_obo(tmp_path, text, f"{name}.obo"), name=name)


def _frame(codebook: dict[str, dict[int, str]], descriptions: dict[str, str] | None = None):
    """The toy frame with `sex` and `codebook`'s fields, each described `x` unless given."""
    fields = ("sex", *codebook)
    return dataclasses.replace(
        FRAME,
        features=("age", *fields),
        categorical=fields,
        codebook={"sex": FRAME.codebook["sex"], **codebook},
        descriptions={
            "age": "age",
            "sex": "sex",
            "target": "toy",
            **dict.fromkeys(codebook, "x"),
            **(descriptions or {}),
        },
    )


def _mapped(frame, ontologies) -> dict[tuple[str, int], tuple[str, str]]:
    found = {}
    for row in build_lexical(frame, ontologies):
        found[state(row)] = (row.concept, row.limits)
    return found


def test_a_negative_meaning_that_would_match_a_term_is_skipped(tmp_path):
    ontologies = {"MONDO": _release(tmp_path, "mondo", [("MONDO:0000001", "hypertension", ())])}
    frame = _frame({"htn": {1: "no hypertension", 2: "hypertension"}})
    assert ("htn", 1) not in _mapped(frame, ontologies)
    assert ("htn", 2) in _mapped(frame, ontologies)
    assert [c.code for c in candidates(frame, ontologies)] == [2]


def test_a_negative_word_after_the_first_does_not_skip_the_state():
    assert negative("without complications")
    assert negative("Not recorded")
    assert not negative("diabetes without complications")
    assert not negative("")


def test_exactly_four_fifths_maps_and_below_does_not(tmp_path):
    term = ("MONDO:0000001", "alpha beta gamma delta", ())
    ontologies = {"MONDO": _release(tmp_path, "mondo", [term])}
    frame = _frame({"f": {1: "alpha beta gamma delta epsilon", 2: "alpha beta gamma delta e z"}})
    found = _mapped(frame, ontologies)
    assert found["f", 1][0] == "MONDO:0000001"
    assert "at token-set Jaccard 4/5" in found["f", 1][1]
    assert ("f", 2) not in found
    (best,) = {c.code: c for c in candidates(frame, ontologies)}[2].options
    assert best.score == Fraction(2, 3)


def test_a_name_beats_a_synonym_of_the_same_prefix_whatever_the_curie(tmp_path):
    ontologies = {
        "MONDO": _release(
            tmp_path,
            "mondo",
            [
                ("MONDO:0000001", "renal calculus", ("kidney stone",)),
                ("MONDO:0000002", "kidney stone", ()),
            ],
        )
    }
    concept, limits = _mapped(_frame({"s": {1: "kidney stone"}}), ontologies)["s", 1]
    assert concept == "MONDO:0000002"
    assert (
        limits
        == 's state 1 ("kidney stone") matched the name "kidney stone" at token-set Jaccard 1'
    )


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_the_smaller_curie_wins_a_tie_whatever_the_release_order(tmp_path, order):
    stanzas = [("MONDO:0000001", "kidney stone", ()), ("MONDO:0000002", "stone kidney", ())]
    ontologies = {"MONDO": _release(tmp_path, "mondo", [stanzas[i] for i in order])}
    assert _mapped(_frame({"s": {1: "kidney stone"}}), ontologies)["s", 1][0] == "MONDO:0000001"


def test_the_label_field_is_matched(tmp_path):
    ontologies = {"MONDO": _release(tmp_path, "mondo", [("MONDO:0000001", "readmitted", ())])}
    assert _mapped(_frame({"f": {1: "zzz"}}), ontologies)["target", 1][0] == "MONDO:0000001"


def test_a_meaning_reads_its_description_only_when_every_word_is_generic(tmp_path):
    ontologies = {
        "MONDO": _release(
            tmp_path,
            "mondo",
            [("MONDO:0000001", "glucose", ()), ("MONDO:0000002", "high glucose level", ())],
        )
    }
    frame = _frame({"g": {1: "high", 2: "high glucose level"}}, {"g": "glucose (High = above)"})
    found = {c.code: c for c in candidates(frame, ontologies)}
    assert found[1].query == "glucose"
    assert found[2].query == "high glucose level"
    assert _mapped(frame, ontologies)["g", 2][0] == "MONDO:0000002"


def test_a_generic_query_drops_its_legend_generic_negative_and_stop_words():
    text = "a diabetes medication's dose changed during the encounter (Ch = changed)"
    assert generic_query(text) == ("diabetes", "medication", "dose", "encounter")
    assert generic_query("pneumonia (ICD-9 486) among the three diagnoses") == (
        "pneumonia",
        "three",
        "diagnoses",
    )
    assert generic_query("no present positive") == ()


def test_a_term_whose_curie_is_another_prefix_is_no_target(tmp_path):
    ontologies = {"HP": _release(tmp_path, "hp", [("MONDO:0000001", "kidney stone", ())])}
    assert candidates(_frame({"s": {1: "kidney stone"}}), ontologies) == ()


def test_text_is_folded_by_nfkc_around_casefold_keeping_marks_in_words():
    assert (
        tokens("".join(chr(0xFF21 + ord(c) - ord("A")) for c in "HYPERTENSION"))
        == tokens("Hypertension")
        == {"hypertension"}
    )
    # U+01F0 casefolds to j and a combining caron; NFKC after casefold composes it again.
    assert words("ǰ") == ("ǰ",)
    # U+0130 casefolds to i and a combining dot above, which stays inside its word.
    assert words("İstanbul fever") == ("i̇stanbul", "fever")


def test_options_are_capped_at_top(tmp_path):
    stanzas = [(f"MONDO:{i:07d}", f"stone {i}", ()) for i in range(1, TOP + 3)]
    ontologies = {"MONDO": _release(tmp_path, "mondo", stanzas)}
    (candidate,) = candidates(_frame({"s": {1: "stone"}}), ontologies)
    assert len(candidate.options) == TOP
    assert len(Index(ontologies).options(frozenset({"stone"}), TOP + 5)) == TOP + 2
