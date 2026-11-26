---
id: paper-novelty
title: Define structured support deficiency and audit novelty
status: blocked
depends_on: evaluate-diabetes-control-results, evaluate-quota-sweep-results, evaluate-tableshift-results, evaluate-variant-dev-results, ontology-auto-mapping-results, paper-control-results
blocks: paper-manuscript
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Define structured support deficiency and audit novelty

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

A missing subgroup-label cell can admit several target distributions consistent with the observed data.

It depends on `paper-control-results` and on the results units of this batch that its
`depends_on` names, so its claim is framed against what those runs show.

## Scope

Define X, Y, G and external K. Exhibit two laws with identical observed training distributions, zero training mass and positive target mass at (g,y), but different conditional feature laws there. State what assumptions on transport, dependence and knowledge coverage identify or bound a target functional. Derive an uncertainty interval or sensitivity bound and give a failure example.

**Findings from planning.**

- Where the definition, the example and the bound are written, a `docs/` document or an ADR, is
  this unit's own first decision.
- The works it must register in [`docs/literature.md`](../docs/literature.md), none of them there
  now: ReTabSyn (Lin 2026), AugMask (Kim 2026), Cai et al. on temporal shift (2025) and Hasny et
  al. on missing values (2026). No identifier for any of them has been read yet, so the unit
  registers each from its primary source. CLLM is registered as `seedat-2024`.

## Acceptance

Compare CLLM (Seedat 2024), ReTabSyn (Lin 2026), AugMask (Kim 2026), Cai et al. on temporal shift (2025) and Hasny et al. on missing values (2026) against the protocol. Verify each primary source in the literature registry. Distinguish missing cells, scarce rows, missing values and temporal shift. Compare knowledge source, learned correlations, prediction target, provenance, compute and API use. Document the search scope and the closest competing claim.
