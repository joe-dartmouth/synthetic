---
status: accepted
topic: research
created: 2026-10-16
updated: 2026-10-17
---
# The BKB layer is pybkb's core, ported under GPL-3.0 and checked against pybkb's own results

## Context

The 2-pager names the absence of a public BKB library as its principal risk (`docs/RESEARCH.md`
section 9). pybkb, by Chase Yakaboski and licensed GPL-3.0, is a public library, but it declares
Python ~3.11 and numpy 1.24.4, so it cannot join this lock (`docs/ARCHITECTURE.md`, pybkb).

pybkb's default branch, `master`, ends at commit `1e1b62a02faa0abc73db4597076c2d38c1313df6`
(2023-07-28). At that commit:

- **`BKBReasoner.update` raises `NotImplementedError`** (`pybkb/reason/bkb.py`). The one working
  inference is `BKBReasoner.heuristic_update_complete_fused`, which returns, for each state of one
  target component, the log of the summed probability of every choice of one agreeing S-node per
  I-node of the evidence and that state. It needs complete evidence and normalizes nothing.
- **pybkb stores no result of that inference, and none of collapsed fusion.** Its tests that run
  compare built, loaded and fused structures; its reasoning test imports modules that no longer
  exist.
- **`fuse` and that inference call nothing in `pybkb/utils/probability.py`**, which serves only
  `BKB.score`, the structure-learning score. `learn/` needs gobnilp and gurobi.
- **Its example `.bkb` files are LZ4-compressed pickles** that name `scipy.sparse._arrays`, a
  module scipy removed in 1.11, the release pybkb itself declares. ruff bans pickle here, and the
  lock holds no lz4, compress-pickle or numba.
- **Its `BKB` stores adjacency in scipy sparse matrices** grown by assigning their private
  `_shape`, keys an I-node by Python equality (so `1`, `1.0` and `True` are one state), orders
  fused I-nodes by hash seed, and defines `==` as a one-sided subset test.

## Decision Drivers

* **Must reuse pybkb's behavior, not re-derive it**: it is the one runnable statement of the BKB
  structure and of source-weighted fusion within reach.
* **Must run on Python 3.14 against this lock**, with no pickle in the repository.
* **Must be checked against results pybkb itself produced**, with a record of how they were made.
* **Must pass the gate**: ruff, ty, the file-size cap and the coverage floor.
* **Should port only what the pipeline reads.**

## Decision

We will **port pybkb's core into `src/bko/bkb/` under GPL-3.0, depend on pybkb nowhere, and test
the port against JSON fixtures that pybkb itself produced in a one-time run outside the lock.**

- **Taken** from pybkb at the commit above: the BKB structure (`bko.bkb.structure`: I-nodes,
  S-nodes, `find_snodes`, the mutex check), `fuse` with and without `collapse` and its
  normalization (`bko.bkb.fusion`), and `heuristic_update_complete_fused` (`bko.bkb.reason`).
  `fuse` also offers, as its default, the fusion of Santos et al.
  ([`santos-2024`](../../literature.md#probabilistic-ontologies-and-bkos), section 6.1), which
  divides each source's reliability by every fragment's and never collapses.
  `bko.bkb.fusion.Normalization` names the three ways to fuse: `FRAGMENTS`, the default, and
  pybkb's `COMPONENT` and `COMPONENT_COLLAPSED`; `fuse` takes no separate `collapse` flag.
- **Not taken**: `learn/`, `BKB.score`, `pybkb/scores.py`, `pybkb/utils/probability.py`, the
  networkx and cytoscape mixin, `BKB.union`, `get_causal_ruleset`, `re_annotate_features`,
  `snodes_by_tail`, `source_features`, `source_inodes`, `non_source_features` and
  `non_source_inodes` (`bko.bkb.fusion.is_source` replaces them), `BKBReasoner.predict`,
  `BaseReasoner`, pickle persistence (`save`, `load`, `dumps`, `loads`), `to_dict`, `json` and the
  hash helpers. No planned unit reads them.
- **The update is `bko.bkb.reason.log_joint_complete(bkb, target, evidence)`.** It returns what
  pybkb's heuristic returns, keyed by state: a natural log of joint mass, minus infinity, or None.
  It sums per I-node and then across I-nodes instead of over pybkb's Cartesian product, which is
  the same sum, cyclic choices included as in pybkb, and refuses a query whose answer depends on a
  component the evidence leaves out.
- **Names are strings.** A collapsed source state is the frozenset of its source names. The
  fixtures turn pybkb's integer names into strings with `str`, and refuse a BKB whose names then
  collide.
- **The layer imports only the standard library.** Storage is lists and dicts; logarithms come
  from `math`.
- **Equality is symmetric and structural, and a BKB is unhashable.** Two BKBs are equal when they
  hold the same I-nodes and the same multiset of S-nodes, whatever their names and order.
- **Deviations from pybkb.** `fuse` defaults to Santos et al.'s fusion and takes one
  `Normalization` instead of pybkb's `collapse` flag, so every check against pybkb's fixtures
  passes `Normalization.COMPONENT` or `Normalization.COMPONENT_COLLAPSED`. The rest are each on a
  path pybkb's fixtures never take: I-nodes keep first-seen
  order; a probability must be a number in [0, 1] (pybkb lets NaN and `True` through); a tail may
  not name its head's component or a component twice; `fuse` refuses an empty fragment list,
  mismatched lengths, a reliability that is not finite and positive, reliabilities that overflow
  when summed, a source name that is not a non-empty string, a repeated source name, a fragment
  that is already fused and a fragment holding one S-node twice; `is_mutex` returns False instead of raising; `find_snodes` with
  `prob=0.0` filters instead of raising, and with an empty `tail_subset` returns every S-node on
  the head (pybkb returns none); the update refuses an unknown, source or non-string target or
  evidence and a source I-node without exactly one prior, sums a tail's source priors in a fixed
  order, makes a state None when an S-node contradicts the
  evidence before it notices a missing component (pybkb's `KeyError` depends on tail order), and
  its log-sum stays finite where pybkb's `log(exp(a) + exp(b))` underflows. Everything else,
  including merging S-nodes only on exactly equal probabilities and summing source totals left to
  right, is pybkb's.
- **Fixtures are JSON, exported once by pybkb.** A committed exporter runs pybkb at the commit
  above under Python 3.11 with pybkb's declared versions, except scipy 1.10.1, the last release
  that reads pybkb's own pickles. It checks every input against its sha256 at that commit, which
  is the guarantee, refuses a pybkb tree holding any other file, statically scans each pickle's
  globals against an allow-list before pybkb loads it as a second check, replays
  pybkb's own test assertions, and writes the example BKBs, fused BKBs and update results in the
  format `bko.bkb.serialize` reads, with a manifest of the commit, every sha256 and the versions.
  No pickle, lz4 or `.bkb` file is committed. `docs/DEVELOPMENT.md` (BKB fixtures) holds the
  procedure.
- **What "the same result as pybkb" means**: every example file decodes to the BKB pybkb loaded;
  `fuse` with `Normalization.COMPONENT` or `Normalization.COMPONENT_COLLAPSED` on pybkb's examples
  gives pybkb's fused BKB S-node for S-node, with exact probabilities;
  and the update agrees with pybkb's within a relative and absolute tolerance of 1e-12, with None
  and minus infinity exact.
- **Attribution.** Every module and test module adapted from pybkb opens its docstring with a
  notice naming pybkb, its author, GPL version 3 and the date it was modified, and citing this
  record; `bko.bkb.errors`, `bko.bkb.serialize`, `bko.bkb.inference` and the serialize, reason,
  determinism, normalization and inference tests are this project's own and carry none. The fixtures' manifest carries a notice naming pybkb,
  its author and GPL version 3; `README.md` names what was adapted.
  pybkb's files carry no copyright line of their own, so none is retained.

## Rationale

The port gives the team the only runnable statement of BKB fusion within reach, and GPL-3.0-only
holds adapted GPL-3.0 code (ADR `tooling/gpl-3-license`). A clean-room build would start without a
specification in the repository and would still check itself against pybkb.

The update keeps pybkb's arithmetic but not its name: "heuristic" and "update" both promise
something it does not compute. Summing per I-node is the same sum as pybkb's product of choices,
because each I-node's chosen S-node brings its own source prior, and it costs one pass instead of
one term per choice. pybkb also counts choices that form a cycle; refusing them would refuse
whole queries over fragments that disagree on direction, so the port keeps pybkb's sum.

The default follows Santos et al. because the project's formalism is theirs (ADR
`research/bko-formal-gap-definition`); pybkb's two modes stay selectable, so the port still
reproduces pybkb exactly. One enum instead of a mode and a flag leaves out every-fragment weights
with collapse, which would count a fragment in several priors of one component.

A one-time export keeps pickle, lz4 and an unpickler out of the repository. A restricted unpickler
could read pybkb's files, but it would be a second parser for a format nothing here writes, behind
a ruff suppression and a new locked dependency; and only running pybkb yields its update and
collapsed-fusion results, which it never stored. The export's interpreter differs from the one ADR
`tooling/python-314-without-synthcity` rules out for synthcity: that one would run on every
pipeline run, while this one runs once, outside the lock, and the gate never touches it.

Strings remove pybkb's `True == 1 == 1.0` identity and make the JSON lossless. Nothing the port
computes needs an array, so the layer needs no extra. A symmetric `==` is what "the same BKB" means
in a fixture check; pybkb's one-sided test would pass a BKB with extra S-nodes.

## Consequences

**Gained**: pybkb's structure, fusion and update on Python 3.14 with the lock unchanged and no
optional extra; every fixture traceable to a pybkb run; no pickle in the repository.
**Accepted**: regenerating a fixture needs a Python 3.11 environment built by hand; the gate checks
the exporter's output, not pybkb's files; the update needs complete evidence and counts cyclic
choices, so no posterior over partial evidence exists (`docs/IDEAS.md`); `fuse` names a source
component from its head's component, so a caller fuses fragments, never a fused BKB. Under the
default, a component's source priors sum to less than 1 where a fragment holds no S-node on it.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Clean-room build from the papers, with pybkb as an oracle | The license allows the port, and pybkb's outputs would be the only check of a clean-room build anyway | 2026-10-16 |
| pybkb as a dependency | It declares Python ~3.11 and numpy 1.24.4, and does not resolve on this repository's interpreter | 2026-10-16 |
| Reading pybkb's pickles through a restricted unpickler and lz4 | A second parser for a format nothing here writes, behind a ruff suppression and a new locked dependency; it still yields no update or collapsed-fusion result | 2026-10-16 |
| Committing pybkb's `.bkb` files | Only a pickle reader can use them | 2026-10-16 |
| pybkb's `to_dict` JSON as the fixture format | It writes a collapsed source state as the text of a Python set, and pybkb has no reader for it | 2026-10-16 |
| Regenerating fixtures inside the gate | Needs Python 3.11 and numpy 1.24 outside the lock; the gate is offline | 2026-10-16 |
| Porting `learn/`, `BKB.score` and `pybkb/utils/probability.py` | `learn/` needs gobnilp and gurobi; the score and the probability routines serve structure learning, which neither `fuse` nor the update reads | 2026-10-16 |
| Porting `BKB.union`, `get_causal_ruleset`, `predict` and the visualization mixin | No planned unit reads them | 2026-10-16 |
| scipy or numpy storage | The adjacency holds only 0 and 1, and pybkb grows it through a private attribute | 2026-10-16 |
| Generic hashable names | Python equality merges `1`, `True` and `1.0`, and JSON has no frozenset | 2026-10-16 |
| pybkb's one-sided `==` and the hash of a mutable BKB | A subset test that ignores multiplicity; a hash that changes as the BKB grows | 2026-10-16 |
| Naming the update `update` or `heuristic_update` | It computes an unnormalized joint mass over complete evidence, which neither name says | 2026-10-16 |
| Refusing queries whose S-nodes can form a cycle | pybkb sums those choices, and a refusal would block whole queries over fragments that disagree on direction | 2026-10-16 |
| pybkb's normalization and collapse as `fuse`'s defaults | The project's formalism is Santos et al.'s, whose fusion divides by every fragment's weight and never collapses; pybkb's modes stay selectable for the fixture checks | 2026-10-17 |
| A `collapse` flag beside the normalization | Leaves every-fragment weights with collapse expressible, whose priors on one component can sum above 1 | 2026-10-17 |
| A separate function for Santos et al.'s fusion | A second entry point that differs from `fuse` only in its denominator | 2026-10-17 |
| Exact belief updating over partial evidence in this port | pybkb implements none and stores no result to check one against, and no planned unit reads a posterior; kept in `docs/IDEAS.md` | 2026-10-16 |
| A NOTICE file for the attribution | A third home for what the per-file notices and this record hold | 2026-10-16 |

## Enforcement

- Owner: `src/bko/bkb/`
- Pinned by: `tests/test_bkb_provenance.py`, `tests/test_bkb_structure.py`,
  `tests/test_bkb_serialize.py`, `tests/test_bkb_fusion.py`, `tests/test_bkb_determinism.py`,
  `tests/test_bkb_reason.py`, `tests/test_bkb_normalization.py`, `tests/test_bkb_inference.py`
- `test_bkb_provenance.py` asserts that each module adapted from pybkb carries the notice, that
  `tests/pybkb/manifest.json` names the commit above and every committed fixture with its sha256,
  and that `tests/pybkb/export.py` is the exporter that wrote them.

## Reversal Conditions

**The port gives way to a dependency if pybkb declares a Python range that contains 3.14 and a
numpy range that contains 2.x**; the check is its `pyproject.toml` on the default branch. This
record is then deleted.

**The fixtures are exported again if pybkb's `master` moves past
`1e1b62a02faa0abc73db4597076c2d38c1313df6` in `pybkb/bkb.py`, `pybkb/fusion/` or
`pybkb/reason/`**; the check is a diff of those paths between the two commits. This record then
names the new commit.

**Exact belief updating joins `bko.bkb` when a unit's acceptance reads a posterior over partial
evidence**; the check is a search of `tasks/` for one. This record is then edited.

## Related

- [GPL-3.0-only for the repository's code, copyright Dartmouth College](../tooling/gpl-3-license.md)
- [Python 3.14, without synthcity](../tooling/python-314-without-synthcity.md)
- [src layout](../tooling/src-layout.md)
- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
