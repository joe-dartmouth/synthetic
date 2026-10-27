---
status: accepted
topic: research
created: 2026-10-17
updated: 2026-10-27
---
# A dataset becomes a fragment of one prior per mapped concept its records show

## Context

RESEARCH, Represent, adds the user's dataset to the BKO as a fragment weighted by observed
frequency, and ADR `research/bko-formal-gap-definition` judges grounding against that fragment.
Santos, Jurmain and Ragazzi ([`santos-2024`](../../literature.md#probabilistic-ontologies-and-bkos))
give no way to turn records into a fragment: their example fuses ontologies only, and they model an
assertion as an unconditional PAA of probability 1. The ontology fragments are about one individual,
`a`, with one single-state component per class (ADR `research/bko-fragment-encoding`), and
`bko.bkb.inference` instantiates by which I-nodes S-nodes head and need, never by their
probabilities.

## Decision Drivers

* **Must ground exactly what the records show**, since the gap list is what grounding leaves out.
* **Must fuse with the ontology fragments as they are**: one individual, single-state classes.
* **Should carry the observed frequency**, as RESEARCH asks, with no free parameter.

## Decision

- **One fragment per set of records, about the ontology fragments' individual `a`**, read as a
  record drawn at random from that set: every record for the population-level definition, a cell's
  records for the per-subgroup and frequency-aware ones (ADR `research/bko-formal-gap-definition`).
- **One prior S-node per held concept.** A concept the dataset's own mapping rows name is held when
  at least one record shows it: a UCI training record for which one of its field mappings applies,
  or a Synthea patient alive on the snapshot date who has one of its codes by then. Its prior is the
  share of records that show it. A concept no record shows gets no S-node.
- **Fusion at weight 1 per fragment** under `fuse`'s default normalization, the dataset fragment
  last.
- **Grounding is instantiation from the held concepts** over the fused BKB; a concept the caller
  holds without a record, such as one a cell assumes (ADR `research/bko-formal-gap-definition`),
  joins the anchor alone. A prior grounds its concept whatever the anchor, so grounding without a
  concept some record shows leaves its prior out of the fragment. A set of records that holds no
  concept and assumes none, such as an empty cell that assumes nothing, leaves every ontology
  statement ungrounded, and nothing is instantiated.
- **Home**: `bko.gaps.observe` builds the fragments, whole and per cell; `bko.gaps.detect` fuses
  and grounds.

## Rationale

Grounding is existential over records: a concept is held when one record shows it, whether the
fragment has one individual or one per record, so a single individual grounds the same statements
without the two-individual encoding ADR `research/bko-fragment-encoding` defers. The share is the
probability that a record drawn at random shows the concept, which is the observed frequency
RESEARCH asks for. Instantiation reads structure only, so a prior of 0 would still ground its
concept, which is why an unheld concept gets no S-node. The weights move only the source priors,
and no planned unit reads a posterior (`docs/IDEAS.md`, Exact belief updating), so weight 1, as the
fusion reproduction used, adds no free parameter. A cell's fragment is the same encoding over the
cell's records, so per-subgroup grounding needs no second individual.

## Consequences

**Gained**: a gap list that changes only with which concepts the records show; one encoding for
UCI records and Synthea patients.
**Accepted**: no co-occurrence is represented, so no statement about two concepts together can be a
gap; a subgroup is represented only by its own cell's fragment; the fusion weights change no gap
list. Two stages read the shares: the frequency-aware score (ADR
`research/frequency-aware-gap-score`), and the direct sampler, which weighs a cell's prior by its
record count against its label group's (ADR `research/bkb-direct-generation`).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| One fragment per record | Needs a second individual, which ADR `research/bko-fragment-encoding` defers, and grounds the same statements | 2026-10-17 |
| Conditional S-nodes between concepts that occur together | Rows record presence only (ADR `research/ontology-mapping-design`), and every head already has a prior, so no grounding changes | 2026-10-17 |
| A prior of 0 for a concept no record shows | Instantiation ignores probabilities, so it would ground the concept | 2026-10-17 |
| A dataset weight from its record count | Changes no gap list; the direct sampler weighs a cell's prior by its record count when it reads the shares (ADR `research/bkb-direct-generation`), outside the fusion, whose weights stay 1 | 2026-10-27 |
| The converter in `bko.datasets` | `bko.ontology` imports `bko.datasets`, so a converter there that reads the mapping rows makes an import cycle | 2026-10-17 |

## Enforcement

- Owner: `src/bko/gaps/observe.py`
- Pinned by: `tests/test_gaps_observe.py`, `tests/test_gaps_detect.py`,
  `tests/test_gaps_definitions.py`

## Reversal Conditions

**One fragment per record replaces the per-set fragment if ADR `research/bko-fragment-encoding`
admits a second individual**; the check is that record's Decision.

## Related

- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [Ontology classes become single-state BKB components joined by is_a and exactMatch S-nodes](bko-fragment-encoding.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [A frequency-aware gap: a concept credibly under four-fifths of its share in the rest of its label group](frequency-aware-gap-score.md)
