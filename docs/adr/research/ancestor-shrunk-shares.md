---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# The `bko-tree` arm draws each code field as one categorical whose shares are shrunk node by node along the ontology

## Context

The `diabetes-codes` frame keeps each encounter's three diagnosis codes, and its field rows map
each code ADR `research/diabetes-diagnosis-codes` selects, in each diagnosis column, to the MONDO
concept that carries the code's xref. With a field this fine, most codes are rare in a small cell.
`bko-direct` draws each concept present at its fused share, the cell's prior shrunk toward its
label group's, and copies the three diagnosis columns from one record that shows exactly the drawn
concepts (ADR `research/bkb-direct-generation`). Its share reads the cell and the label group and
nothing between them: a code the cell never shows borrows only from the label group, never from
the codes the ontology places beside it.

A hierarchy of classes gives a share somewhere to borrow from. McCallum et al. estimate a leaf
class's word distribution by shrinking it toward the distributions of its ancestors in a class
hierarchy ([`mccallum-1998`](../../literature.md#knowledge-guided-generation)), and the
hyper-Dirichlet of Dennis puts a distribution on a tree's leaves through one conditional
distribution per internal node ([`dennis-1991`](../../literature.md#knowledge-guided-generation)).
Neither work is fetched, so both readings are this record's.

A direct rule would shrink a code's cell share toward its parent's cell share times the code's
share within the parent in the label group, recursively from the release's root. Shares computed
that way do not sum to 1 unless every leaf sits at one depth, and a chain of single children
changes them, so the rule depends on how deep the release happens to nest a code.

## Decision Drivers

* **Must sum to 1 over a field's values**, so the shares are a distribution one uniform draw can
  place.
* **Must give a code the cell never shows a share above 0 and below its parent's**, so the cell
  borrows from the ontology.
* **Must leave `bko-direct`'s draw byte for byte**, so its runs re-read under identity checks.
* **Must read a release whose classes have several parents**, as MONDO's do.

## Decision

1. **Spec field and arm.** `DrawSpec.tree`, a bool, default false, and the arm `bko-tree` at
   `tree=True`. Off its default it needs `mapped=True` and `presence` shares, and it is logged
   only off its default, as every spec field is.
2. **Tree fields.** A tree field is a column other than `age`, `sex` and the label where every
   row reading it is one single-clause `Codes(column, S)`, the sets are pairwise disjoint and at
   least two rows read it. On `diabetes-codes` these are `diag_1`, `diag_2` and `diag_3`. A tree
   field leaves the field groups and the concept draw: its concepts are drawn as its leaves.
3. **The tree.** Per tree field: the mapped concepts' ancestor closure in their release, under a
   virtual top whose children are the closure's roots and an `unmapped` leaf, which holds every
   value no row maps. A node with several parents in the closure keeps the one with the most
   mapped concepts at or below it, ties to the lowest CURIE, and the tree holds only the nodes on
   some mapped concept's path to the top. A mapped concept that is an ancestor of another gets its
   own leaf beneath itself, named the concept followed by `/self`. A record shows a leaf when its
   column holds a code of a row mapping the leaf's concept, and `unmapped` when it holds no
   mapped code.
4. **Shares.** With `n_c` and `n_g` the cell's and the label group's record counts at a node for
   that column, a node's count the sum of its leaves', and `w = GROUP_WEIGHT`,
   `r(v|u) = (n_c(v) + w * n_g(v) / n_g(u)) / (n_c(u) + w)` for a child `v` of `u`, the group
   term 0 where `n_g(u)` is 0, and `s(v) = s(u) * r(v|u)` with `s(top) = 1`. Children sum to their
   parent; a chain of single children changes nothing; an empty cell gives the label group's
   shares.
5. **The draw.** One uniform per tree field per record, in frame column order, after the concept
   draw and in the column loop, placed by cumulative share over the leaves of positive share in
   depth-first order with children by id, clipped to the last such leaf. The column's value is
   copied from a record whose column shows the drawn leaf, the cell's first, else the label
   group's, by one uniform pick as `bko-direct`'s. `Draw.concepts` holds each drawn leaf's
   concept, none for `unmapped`. A spec that also sets `min_donors` or `reuse_cap` picks the
   column as a unit of its own under the floor and the cap of ADR `research/sampler-donor-limits`.
6. **The trace.** For each drawn leaf, the trace names each level's node from the top's child down
   to the leaf, the cell's and the label group's counts at the node and at its parent, the weight,
   the conditional and the share.

## Rationale

The conditional form is the Dirichlet-tree reading of that direct rule: each node splits its
parent's share among its children, so the leaves sum to 1 by construction and a chain of single
children passes its share down whole. Each split is the cell's conditional shrunk toward the label
group's with the weight `bko-direct`'s share uses, so a cell of `GROUP_WEIGHT` records weighs as
its group does at every node. A code the cell lacks takes `w * n_g(v) / n_g(u)` of its parent's
split, so it gets a share above 0 whenever the label group shows it and below its parent's
whenever a sibling holds any record. One parent per node keeps the shares a tree's; the parent
that holds the most mapped concepts lends each code the widest neighbourhood, and the CURIE breaks
a tie the same way on every run. Drawing each tree field alone, rather than the three diagnosis
columns as one pattern, keeps each column's donors as many as the codes allow.

## Consequences

**Gained**: a free, seeded arm whose contrast with `bko-direct` and `bko-nomap` on the same
holdouts reads what the ontology's hierarchy adds to a coded field's shares.
**Accepted**: a tree field's value is copied from a donor of that column alone, so the three
diagnosis columns of one row may come from three records, and E15's contrast with `bko-direct`
mixes the shrinkage with that change; a class's other parents lend nothing; the leaves' shares
depend on the release's hierarchy, so a new release is a new arm's reading.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The direct recursion of Context | Does not sum to 1 unless leaves are padded to one depth, and depends on chain length | 2026-11-25 |
| Shrinking toward the label group only | That is `bko-direct` | 2026-11-25 |
| One categorical over the three columns' joint pattern | Donors are too sparse | 2026-11-25 |

## Enforcement

- Owner: `src/bko/generate/tree.py` and `src/bko/generate/tree_draw.py` with
  `src/bko/generate/direct.py`, the spec in `src/bko/generate/draw_spec.py`,
  `src/bko/evaluate/direct_arms.py`, `src/bko/evaluate/runs_direct.py`,
  `src/bko/evaluate/redraw.py`, `src/bko/evaluate/run_scores.py`
- Pinned by: `tests/test_generate_direct.py` (`DRAW_DIGEST`, `bko-direct`'s draw unchanged),
  `tests/test_generate_tree.py` (the hand shares, their sum, the unseen code's share, single
  children, the parent rule, the own leaf and each code's leaf, the tree fields, the refusal of
  a count off the leaves, the placement of a uniform, one uniform per tree field after the
  concept draw, the donor showing the drawn leaf, the label group's donor of a leaf the cell
  lacks, and the floor and the cap), `tests/test_evaluate_tree.py` (the arm's params, its trace
  and each level's parts, the refusal of an arm line whose `tree` is not its arm's, the refusal
  of a run with no tree field, `code-share-error` and E15's runs),
  `tests/test_generate_tree_pinned.py` (on `diabetes-codes` the tree fields are the three
  diagnosis columns, each one's shares sum to 1 in E6's cell and its label group, and E15's gap
  request asks E6's quota, run with `--releases`)

## Reversal Conditions

**`bko-tree`'s removed-cell recall is not non-inferior to `bko-direct`'s within the margin, or its
`code-share-error` is not below `bko-direct`'s,** on E15's runs; the check is E15's Result in
`docs/EVIDENCE.md`. **`tree_fields` finds no field on a frame whose rows partition a coded
field**; the arm then draws as `bko-direct` does.

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [The diabetes-codes dataset keeps each ICD-9 diagnosis code and maps it through MONDO's equivalentTo xrefs](diabetes-diagnosis-codes.md)
- [A record's explanation is the stored or re-derivable facts that determine it](record-explainability.md)
