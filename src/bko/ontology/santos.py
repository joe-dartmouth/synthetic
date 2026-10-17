"""Santos et al.'s MONDO/DO fusion example, rebuilt with `bko.bkb` and checked quantity by quantity.

Santos, Jurmain and Ragazzi (`santos-2024`, section 7) fuse 11 classes of MONDO and DO, joined by
eight bridge PTAs, at weight 1 each; instantiate the result from an anchor; and report two
contradictory inferences of probability 0.33 and a chain from sciatic neuropathy to inflammatory
disease that neither ontology holds. This module is the one home for what the paper prints: its
terms, PTAs, anchor, passes, inference probability and misprints. `reproduce` checks each against
loaded releases and returns one `Finding` per quantity. The bridge comes from MONDO's
`skos:exactMatch` annotations, where the paper queried the OxO service (ADR
`research/bko-fragment-encoding`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from bko.bkb.fusion import Normalization, fuse, is_source
from bko.bkb.inference import derivation, instantiate, probability, reach
from bko.bkb.structure import BKB
from bko.datasets.sources import DO, MONDO, Source
from bko.ontology.fragments import INDIVIDUAL, exact_match_fragment, is_a_fragment
from bko.ontology.terms import Ontology


logger = logging.getLogger(__name__)

type Status = Literal["match", "erratum", "differs", "reported"]

PAPER_RELEASES: Final[Mapping[str, Source]] = MappingProxyType({"DOID": DO, "MONDO": MONDO})
"""The pinned release for each prefix the example reads."""


@dataclass(frozen=True, slots=True)
class PaperTerm:
    """One row of the paper's Table 1.

    Attributes:
        printed: The id as Table 1 prints it.
        label: The label as Table 1 prints it.
        curie: The release's id for the class; `printed` unless Table 1 misprints it.
    """

    printed: str
    label: str
    curie: str


@dataclass(frozen=True, slots=True)
class PaperPTA:
    """One PTA of the paper's section 7: P(head | tail) = 1.

    Attributes:
        label: The paper's name for it, renumbered where the paper repeats one.
        head: The consequent class.
        tail: The antecedent class.
        fragment: The fragment holding it: `mondo`, `do` or `bridge`.
        printed: How the paper prints it, where that differs from `label`, `head` and `tail`.
    """

    label: str
    head: str
    tail: str
    fragment: str
    printed: str | None = None


@dataclass(frozen=True, slots=True)
class PaperPass:
    """What section 7.1 reports of the instantiation from the anchor, by PTA label.

    Attributes:
        item: Which quantity.
        printed: The labels the paper gives, in `PTAS` order.
        algorithm: The labels the paper's algorithm (section 5.3) adds; `printed` unless the
            narration contradicts it.
    """

    item: str
    printed: tuple[str, ...]
    algorithm: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Finding:
    """One quantity of the example, as the paper prints it and as the releases give it.

    Attributes:
        kind: `term`, `pta`, `instantiation`, `inference` or `chain`.
        item: Which one.
        paper: What the paper prints; empty where it prints nothing.
        measured: What the releases give.
        status: `match`; `erratum`, where the paper misprints or contradicts its own algorithm and
            the releases give what it means; `differs`; or `reported`, a measurement the paper
            gives no value for.
    """

    kind: str
    item: str
    paper: str
    measured: str
    status: Status


TERMS: Final[tuple[PaperTerm, ...]] = (
    PaperTerm("MONDO:0006960", "Sciatic Neuropathy", "MONDO:0006960"),
    PaperTerm("MONDO:0001543", "Lesion of Sciatic Nerve", "MONDO:0001543"),
    PaperTerm("MONDO:0001397", "Mononeuropathy", "MONDO:0001397"),
    PaperTerm("MONDO:0002121", "Mononeuritis Simplex", "MONDO:0002121"),
    PaperTerm("MONDO:0002122", "Neuritis", "MONDO:0002122"),
    PaperTerm("MONDO:0021166", "Inflammatory Disease", "MONDO:0021166"),
    PaperTerm("DOID:114466", "Sciatic Neuropathy", "DOID:11446"),
    PaperTerm("DOID:12528", "Lesion of Sciatic Nerve", "DOID:12528"),
    PaperTerm("DOID:9473", "Mononeuritis of Lower Limb", "DOID:9473"),
    PaperTerm("DOID:1188", "Mononeuropathy", "DOID:1188"),
    PaperTerm("DOID:1802", "Mononeuritis", "DOID:1802"),
)
"""Table 1, in its order; MONDO maps sciatic neuropathy to DOID:11446, which DO holds."""

PTAS: Final[tuple[PaperPTA, ...]] = (
    PaperPTA("TM1", "MONDO:0006960", "MONDO:0001543", "mondo"),
    PaperPTA("TM2", "MONDO:0001397", "MONDO:0006960", "mondo"),
    PaperPTA("TM3", "MONDO:0001397", "MONDO:0002121", "mondo"),
    PaperPTA("TM4", "MONDO:0002122", "MONDO:0002121", "mondo"),
    PaperPTA(
        "TM5",
        "MONDO:0021166",
        "MONDO:0002122",
        "mondo",
        printed="TM4: Inflammatory Disease | Mononeuritis",
    ),
    PaperPTA("TD1", "DOID:12528", "DOID:11446", "do"),
    PaperPTA("TD2", "DOID:9473", "DOID:12528", "do"),
    PaperPTA("TD3", "DOID:1802", "DOID:9473", "do"),
    PaperPTA("TD4", "DOID:1188", "DOID:1802", "do"),
    PaperPTA("TB1", "MONDO:0001543", "DOID:12528", "bridge"),
    PaperPTA("TB2", "DOID:12528", "MONDO:0001543", "bridge"),
    PaperPTA("TB3", "MONDO:0006960", "DOID:11446", "bridge"),
    PaperPTA("TB4", "DOID:11446", "MONDO:0006960", "bridge"),
    PaperPTA("TB5", "MONDO:0001397", "DOID:1188", "bridge"),
    PaperPTA("TB6", "DOID:1188", "MONDO:0001397", "bridge"),
    PaperPTA("TB7", "MONDO:0002121", "DOID:1802", "bridge"),
    PaperPTA("TB8", "DOID:1802", "MONDO:0002121", "bridge"),
)
"""The 17 PTAs of section 7; MONDO puts inflammatory disease above neuritis, not mononeuritis."""

ANCHOR: Final = ("MONDO:0001543", "DOID:11446")
"""Section 7.1's reasoning anchor: lesion of sciatic nerve in MONDO, sciatic neuropathy in DO."""

PASSES: Final[tuple[PaperPass, ...]] = (
    PaperPass("pass 1", ("TM1", "TD1"), ("TM1", "TD1", "TB2", "TB3")),
    PaperPass("pass 2", ("TM2", "TD2", "TB1", "TB2", "TB3", "TB4"), ("TM2", "TD2", "TB1", "TB4")),
    PaperPass(
        "after pass 2",
        ("TM1", "TM2", "TD1", "TD2", "TB1", "TB2", "TB3", "TB4"),
        ("TM1", "TM2", "TD1", "TD2", "TB1", "TB2", "TB3", "TB4"),
    ),
)
"""Section 7.1 narrates TB2 and TB3 in pass 2, though their antecedents are the anchor."""

INFERENCES: Final = (("a", "MONDO:0001543", "MONDO:0006960"), ("b", "DOID:11446", "DOID:12528"))
"""Section 7.2's contradictory inferences: each from its half of the anchor to the class its
ontology puts above it."""

PAPER_PROBABILITY: Final = "0.33"
"""P(a) = P(b), section 7.2."""

CHAIN: Final = ("DOID:11446", "MONDO:0021166")
"""Section 7.2's emergent chain: from DO's sciatic neuropathy to MONDO's inflammatory disease."""

MONDO_START: Final = "MONDO:0006960"
"""MONDO's sciatic neuropathy, where the chain would start in MONDO alone."""

WEIGHTS: Final = (1, 1, 1)
"""The paper's weight for each of the MONDO, DO and bridge fragments."""

_LABELS: Final = {(pta.fragment, pta.head, pta.tail): pta.label for pta in PTAS}
_ORDER: Final = {pta.label: index for index, pta in enumerate(PTAS)}


def paper_fragments(mondo: Ontology, do: Ontology) -> tuple[BKB, BKB, BKB]:
    """The MONDO, DO and bridge fragments over the example's classes the releases hold live.

    Raises:
        ScopeError: If either release holds none of them live.
    """
    mondo_scope, do_scope = _live(mondo, "MONDO"), _live(do, "DOID")
    return (
        is_a_fragment(mondo, mondo_scope, prefix="MONDO", name="mondo"),
        is_a_fragment(do, do_scope, prefix="DOID", name="do"),
        exact_match_fragment(
            mondo,
            mondo_scope,
            do,
            do_scope,
            subject_prefix="MONDO",
            target_prefix="DOID",
            name="bridge",
        ),
    )


def reproduce(mondo: Ontology, do: Ontology) -> tuple[Finding, ...]:
    """Every quantity of the example, checked against `mondo` and `do`, in the paper's order.

    Raises:
        ScopeError: If either release holds none of the example's classes live.
    """
    releases = {"MONDO": mondo, "DOID": do}
    findings = [_term(term, releases[_prefix(term.curie)]) for term in TERMS]
    fragments = paper_fragments(mondo, do)
    findings.extend(_ptas(fragments))
    fused = fuse(fragments, WEIGHTS)
    findings.extend(_instantiation(fused))
    findings.extend(_inference(item, fused, start, goal) for item, start, goal in INFERENCES)
    item, start, goal = INFERENCES[0]
    pybkb = fuse(fragments, WEIGHTS, normalization=Normalization.COMPONENT, name="pybkb")
    measured = _measured(pybkb, start, goal)
    findings.append(
        Finding("inference", f"{item} under pybkb normalization", "", measured, "reported")
    )
    findings.extend(_chains(fused, fragments, mondo))
    return tuple(findings)


def _term(term: PaperTerm, release: Ontology) -> Finding:
    found = release.terms.get(term.curie)
    if found is None or found.obsolete:
        state = "absent" if found is None else "obsolete"
        return Finding("term", term.printed, term.label, state, "differs")
    measured = f"{term.curie} {found.name}"
    agrees = (found.name or "").casefold() == term.label.casefold()
    if term.printed == term.curie:
        return Finding("term", term.printed, term.label, measured, "match" if agrees else "differs")
    misprinted = term.printed not in release.terms
    held = "absent" if misprinted else "present"
    status: Status = "erratum" if agrees and misprinted else "differs"
    return Finding("term", term.printed, term.label, f"{term.printed} {held}; {measured}", status)


def _ptas(fragments: tuple[BKB, BKB, BKB]) -> list[Finding]:
    held = {
        (fragment.name, snode.head[0], inode[0])
        for fragment in fragments
        for snode in fragment.snodes
        for inode in snode.tail
    }
    findings: list[Finding] = []
    for pta in PTAS:
        paper = f"{pta.head} | {pta.tail}" if pta.printed is None else pta.printed
        if (pta.fragment, pta.head, pta.tail) in held:
            status: Status = "match" if pta.printed is None else "erratum"
            findings.append(Finding("pta", pta.label, paper, pta.fragment, status))
        else:
            findings.append(Finding("pta", pta.label, paper, "absent", "differs"))
    expected = {(pta.fragment, pta.head, pta.tail) for pta in PTAS}
    findings.extend(
        Finding("pta", "extra", "", f"{name}: {head} | {tail}", "differs")
        for name, head, tail in sorted(held - expected)
    )
    return findings


def _instantiation(fused: BKB) -> list[Finding]:
    anchor = [(curie, INDIVIDUAL) for curie in ANCHOR if (curie, INDIVIDUAL) in fused]
    named = [_names(fused, added) for added in (instantiate(fused, anchor) if anchor else ())]
    seen = [*named, (), ()]
    measured = {
        "pass 1": seen[0],
        "pass 2": seen[1],
        "after pass 2": _ordered((*seen[0], *seen[1])),
    }
    findings = [_pass(paper, measured[paper.item]) for paper in PASSES]
    every = _ordered(tuple(label for labels in named for label in labels))
    complete = sorted(every) == sorted(pta.label for pta in PTAS)
    paper, found = f"{len(PTAS)} PTAs", f"{len(every)} PTAs"
    findings.append(
        Finding("instantiation", "fixpoint", paper, found, "match" if complete else "differs")
    )
    return findings


def _pass(paper: PaperPass, measured: tuple[str, ...]) -> Finding:
    if measured == paper.printed:
        status: Status = "match"
    elif measured == paper.algorithm:
        status = "erratum"
    else:
        status = "differs"
    shown = " ".join(measured) or "none"
    return Finding("instantiation", paper.item, " ".join(paper.printed), shown, status)


def _names(fused: BKB, indices: tuple[int, ...]) -> tuple[str, ...]:
    snodes = fused.snodes
    labels: list[str] = []
    for index in indices:
        snode = snodes[index]
        [(_, source)] = [inode for inode in snode.tail if is_source(inode[0])]
        [(tail, _)] = [inode for inode in snode.tail if not is_source(inode[0])]
        labels.append(_LABELS.get((str(source), snode.head[0], tail), "extra"))
    return _ordered(tuple(labels))


def _ordered(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(labels, key=lambda label: _ORDER.get(label, len(_ORDER))))


def _inference(item: str, fused: BKB, start: str, goal: str) -> Finding:
    measured = _measured(fused, start, goal)
    agrees = measured != "unreached" and f"{float(measured):.2f}" == PAPER_PROBABILITY
    return Finding("inference", item, PAPER_PROBABILITY, measured, "match" if agrees else "differs")


def _chains(fused: BKB, fragments: tuple[BKB, BKB, BKB], mondo: Ontology) -> list[Finding]:
    start, goal = CHAIN
    found = _derive(fused, start, goal)
    path = "absent" if found is None else " -> ".join(found[0])
    status: Status = "differs" if found is None else "match"
    findings = [Finding("chain", "fused", f"{start} -> {goal}", path, status)]
    live = [c for c, t in mondo.terms.items() if c.startswith("MONDO:") and not t.obsolete]
    release = is_a_fragment(mondo, live, prefix="MONDO", name="mondo release")
    logger.info("mondo release fragment classes=%d snodes=%d", len(live), len(release.snodes))
    mondo_fragment, do_fragment, _ = fragments
    for item, bkb, origin in (
        ("mondo fragment", mondo_fragment, MONDO_START),
        ("do fragment", do_fragment, start),
        ("mondo release", release, MONDO_START),
    ):
        if (origin, INDIVIDUAL) not in bkb:
            # Without its start class the control cannot fail, so it is not evidence.
            findings.append(Finding("chain", item, "absent", f"{origin} not in it", "differs"))
            continue
        alone = _derive(bkb, origin, goal)
        measured = "absent" if alone is None else " -> ".join(alone[0])
        findings.append(
            Finding("chain", item, "absent", measured, "match" if alone is None else "differs")
        )
    value = "absent" if found is None else repr(probability(fused, found[1]))
    findings.append(Finding("chain", "probability", "", value, "reported"))
    return findings


def _measured(bkb: BKB, start: str, goal: str) -> str:
    found = _derive(bkb, start, goal)
    return "unreached" if found is None else repr(probability(bkb, found[1]))


def _derive(bkb: BKB, start: str, goal: str) -> tuple[list[str], tuple[int, ...]] | None:
    """The classes from `start` to `goal` and the S-nodes of that inference, or None.

    Every S-node of these fragments has exactly one tail I-node off the source components.
    """
    origin, target = (start, INDIVIDUAL), (goal, INDIVIDUAL)
    if origin not in bkb:
        return None
    reached = reach(bkb, [origin])
    if target not in reached:
        return None
    snodes = bkb.snodes
    path, node = [goal], target
    while (index := reached[node]) is not None:
        [node] = [inode for inode in snodes[index].tail if not is_source(inode[0])]
        path.append(node[0])
    return path[::-1], derivation(bkb, reached, target)


def _live(release: Ontology, prefix: str) -> list[str]:
    return [
        term.curie
        for term in TERMS
        if _prefix(term.curie) == prefix
        and (found := release.terms.get(term.curie)) is not None
        and not found.obsolete
    ]


def _prefix(curie: str) -> str:
    return curie.split(":", 1)[0]
