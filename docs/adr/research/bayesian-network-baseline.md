---
status: accepted
topic: research
created: 2026-10-31
updated: 2026-11-08
---
# The Bayesian network baseline: GreedyBayes structure, BDeu tables and donor values, in-house

## Context

The BKO arms draw on a Bayesian knowledge base fused from ontologies and the data. A reviewer's
first question is what a Bayesian network learned from the data alone gives, with no ontology:
PrivBayes ([`zhang-2014`](../../literature.md#tabular-generators)) is the standard such generator
for tabular release, and B2 runs it as one more row-adding method beside E1's and E3's. The
baselines' contract (ADR [`research/baseline-augmentation-design`](baseline-augmentation-design.md))
fixes how any method fills a quota: draw whole records and keep a stratum's first matching rows.

pgmpy is the maintained Python library for Bayesian networks. A probe on 2026-10-31 added
`pgmpy==1.1.2` to an extra in a scratch worktree of `a58d9ab`: `uv lock` exited 0 and the lock grew
from 116 to 127 packages (`formulaic`, `hf-xet`, `huggingface-hub`, `interface-meta`, `opt-einsum`,
`patsy`, `pgmpy`, `pyparsing`, `scikit-base`, `statsmodels`, `wrapt`); `uv sync --locked` exited
0; a `HillClimbSearch` and BIC fit on two binary columns and a 10-row `forward_sample` gave shape
`(10, 2)` and imported none of `huggingface_hub`, `litellm` or `torch`. Its structure search is a
BIC hill climb, not PrivBayes' GreedyBayes.

## Decision Drivers

* **The structure search is PrivBayes'**, so B2 reads the generator the literature names.
* **The data's own cells survive**: a sampled record must name one age band, so the band edges are
  bin edges.
* **No empty conditional at n = 238**: a parent configuration no training row shows still draws.
* **No new dependency beside a small learner**, under the lock rules (ADR `tooling/uv-ruff-ty`).

## Decision

We will build **an in-house GreedyBayes learner, `bko.baselines.bayes_net`**, run as the method
`bayes-net`:

- **States.** Each measured field is cut at the 4 interior quantiles of its training rows, 5
  equal-frequency bins (`BINS`), with each age band edge added for `age`, then each cut that would
  leave a bin empty removed, so every state holds a training row and each age state's rows lie in
  one band; a coded field, the label included, keeps as states the codes its training rows hold,
  so no state lacks a donor.
- **Structure.** GreedyBayes: the first field is
  `frame.columns[derived_seed(seed, "bayes-net") % len(frame.columns)]`; each next field is the
  one, with the parent set of `min(k, placed)` fields already placed, of the largest empirical
  mutual information, ties to the first field in the frame's order and then the first parent set in
  placement order; `k` is 2 (`K`).
- **Tables.** Each field given its parents is the BDeu posterior mean at equivalent sample size 1
  (`ESS`); a parent configuration no training row shows gets the uniform prior.
- **Draw.** Fields are drawn in the learned order from `default_rng(derived_seed(seed,
  "bayes-net"))`, and each field's value is copied from a uniformly chosen training row whose state
  for that field is the drawn one; `fill` keeps each stratum's first matching rows, as for every
  generator. The draw reads the generator as a row-by-row draw would: one `random` double per row
  per node in the learned order, then one bounded integer per row per field in frame order. So
  grouping rows by parent configuration and by state leaves the rows byte-identical under the
  locked numpy.
- **Learn speedups** leave the learner's output byte-identical: the same order, edges and
  tables, so the draw is unchanged. `learn` computes each mutual information once per field and
  parent set, and counts each table with `np.bincount`.
- **Params.** The `arm` line names the request's seed, the first field, `k`, `BINS`, the
  equivalent sample size, each field's number of states and the learned edges as `[parent, child]`
  in drawing order; a `fit holdout=%d fill=%s method=bayes-net seconds=` line times the binning and the search.

## Rationale

GreedyBayes is the structure search PrivBayes runs; pgmpy's hill climb under BIC is another one,
and a B2 that ran it would read a different generator under PrivBayes' name. With two parents of
five bins each, a field has 25 parent configurations, about 9 training records apiece at 238,
which keeps the tables estimable; a larger `k` empties most of them. BDeu at a small equivalent sample size
keeps every configuration drawable without moving a well-counted one far from its counts. Copying
each value from a training row in its bin keeps every value on the archive's precision and range,
as sequential CART's donors do.

## Consequences

**Gained**: PrivBayes' structure without its noise, at no new dependency; a learned structure each
run logs; a method the contract and the harness already know how to fill and score.
**Accepted**: departures from PrivBayes: no differential-privacy noise, a fixed `k` rather than
one chosen from the privacy budget, equal-frequency bins rather than its generalization, and a BDeu
prior where it reads noisy marginals; the in-house learner is ours to keep correct.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| pgmpy | 11 new locked packages, `huggingface-hub` and the native `hf-xet` among them, for a BIC hill climb that is not PrivBayes' search (probe above) | 2026-10-31 |
| A BIC hill climb | not the structure search PrivBayes runs | 2026-10-31 |
| Exact conditionals (maximum likelihood) | at n = 238 many parent configurations are empty and draw nothing, and the rest copy their few records | 2026-10-31 |
| PrivBayes' noise | B2 asks what the network's structure gives, not what a privacy budget costs | 2026-10-31 |

## Enforcement
- Owner: `src/bko/baselines/bayes_net.py`
- Pinned by: `tests/test_baselines_bayes_net.py`

## Reversal Conditions
A study that reads the privacy cost of release reopens PrivBayes' noise; a frame large enough that
`k` = 3 keeps its configurations filled reopens `k`.

## Related
- [The baselines: each method fills a given quota of added rows](baseline-augmentation-design.md)
