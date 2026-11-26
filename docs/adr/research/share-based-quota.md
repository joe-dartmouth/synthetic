---
status: accepted
topic: research
created: 2026-11-26
updated: 2026-11-26
---
# A flagged cell's quota can be read from its fused shares and its neighbours' label split, beside the fixed fill

## Context

The harness asks each flagged cell for `fill_to - count` rows (ADR
[`research/e1-evaluation-protocol`](e1-evaluation-protocol.md), The gap request). On E6's frame
`fill_to` is 1,269, the cell's training count before the removal (ADR
[`research/diabetes-subgroup-removal-design`](diabetes-subgroup-removal-design.md)). That constant
is fixed by the experiment, not by the cell: it says nothing of how many rows a cell needs before
its fill shows the concepts its fused shares give it, or before its label split matches its
neighbours'.

Two counts can be read from what a holdout's training split already holds. The fused BKB gives
each mapped concept a share in the cell (`bko.gaps.shares.shares`, the shares the direct sampler
draws at; ADR [`research/bkb-direct-generation`](bkb-direct-generation.md)). A concept of share p
is missing from n independent draws with chance (1 - p)^n, so the rows at which every common
concept appears follow from the shares. The cell's other label in the same age band and sex,
beside the label split of the same band's other sex, gives the size the cell would have were its
odds its neighbours'.

`bko.evaluate.quotas` holds the rewrite arm's per-call value quotas (ADR
[`research/identifier-perturbation`](identifier-perturbation.md)). It is a different thing: it
splits one call's rows across a field's values and never sets how many rows a cell is asked for.

## Decision Drivers

* **Must leave every existing run's request byte for byte**, so its folders re-read under the
  identity checks and `rows` still reproduces each store.
* **Must give a cell of no record a finite count**, since E6's empty fraction asks one.
* **Must keep the counts in the gap stage**, so a reader that rebuilds a request computes them
  from the training split alone.
* **Must read the rule's effect against the fixed fill without paying for E6's Messages arms
  again.**

## Decision

1. **Module.** `bko.gaps.quota` holds the counts and imports nothing from `bko.generate` or
   `bko.evaluate`. It reads the fused shares as the direct sampler does: the cell's records
   against its label group's, observed under the dataset's field rows.
2. **Coverage.** `coverage(share)` is the smallest n at which every concept of fused share at
   least `SHARE_FLOOR` = 0.02 appears with probability at least 0.95, by a union bound: the sum
   over those concepts of (1 - p)^n is at most `MISS_CHANCE` = 0.05. It is 0 when no concept
   qualifies. Each share, the floor and the miss chance are read as the fraction of denominator
   at most 10^9 nearest them, and the sum is compared exactly: a share is the float nearest a
   ratio of counts, so a share one ulp below 0.02 reads as 1/50, and a share of 0.95 meets the
   miss chance after one row. A NaN share is refused.
3. **Expected size.** For cell (a, b, y), `expected_size` is n(a, b, not y) times
   (N1 + 1/2) / (N0 + 1/2), N1 and N0 the counts of (a, b', y) and (a, b', not y) over every
   other value b' of the second cell axis, read by position. When both are 0, N1 and N0 are the
   split's counts of y and not y. It is computed as an exact fraction and rounded half up.
4. **Recommended.** `recommended(count, covered, expected)` is
   `max(0, max(covered, expected) - count)`.
5. **Rules.** `QuotaRule` holds seven: `fill-to`, the current rule, `fill_to - count`;
   `coverage`, `coverage-x2`, `coverage-x4` and `coverage-x8`, one, two, four and eight times
   coverage; `expected-size`; and `recommended`. Every rule but `recommended` asks
   `max(0, value - count)`. A cell asked 0 leaves the request, and a request left empty raises the
   existing "nothing to fill" `EvaluationError`.
6. **Experiment field.** `Experiment.quota_rule`, default `fill-to`, follows `fill_to`. It is
   refused beside `flagged_cell` or `synthetic_share`, whose requests are not read from the
   threshold. The `experiment` line names `quota_rule=<value>` after `fill_to=` and `cutoff=`
   only off its default, and an older line lacking the key reads at `fill-to`.
7. **Request.** `gap_request`, `score_holdout`, `run` and `shared`'s exclusion take the data's
   field rows and source name as required keywords, so every caller that builds or rebuilds a
   gap request passes both; a rule other than `fill-to` with no rows is refused.
8. **Log line.** Under a rule other than `fill-to`, each flagged cell writes
   `quota-rule holdout=%d fill=gap key=%s rule=%s count=%d coverage=%d expected=%d asked=%d`,
   `key` as the direct arms' lines name it, once per holdout scored: `score_holdout` writes the
   lines, and every reader that only rebuilds the request (`trace`, `privacy`,
   `contradictions <run>`, `shared`'s `--with` exclusion) writes none.
9. **Reading.** `python -m bko.evaluate representativeness` reads a run's `rows` stores: per
   holdout, arm and flagged cell, the largest absolute gap between the added rows' concept shares
   and the fused shares, and the cell's share of the arm's training set beside `real`'s, with the
   arm's flagged recall from the run's scores.
10. **Sweep.** E21 runs E12's `e12-thin` and `e12-empty` with `quota_rule` set to each of the six
    rules other than `fill-to`, each cut into ten slices, its privacy reference arm `real`. A run
    name holds no digit after its entry, so the runs name `coverage-x2`, `coverage-x4` and
    `coverage-x8` as `coverage-two`, `coverage-four` and `coverage-eight`. Every
    E12 arm is free, so the sweep makes no paid call. The fixed fill is read from E12's own runs
    with `compare --beside`.

## Rationale

The union bound is the smallest count that makes the stated guarantee hold whatever the
dependence between concepts, and it needs only the shares. On E6's cell at holdout 0 it is
small: the thinned cell's 19 records leave 9 concepts at or above 0.02, the least 0.042, and
coverage is 86; the empty cell reads its label group's shares, 12 concepts, the least 0.023, and
coverage is 163. The multiples step coverage up toward the fixed fill so the sweep reads recall
along that axis.

The expected size reads the cell's odds from its band's other sex, so it rests on counts the
training split holds in thousands even when the cell holds none. The half added to each count
keeps it finite when a neighbour holds no record of a label. On E6's cell at holdout 0,
n(6-7, male, 0) is 11,740 and the female cells hold 1,314 and 12,422, so the expected size is
1,242 at both fractions.

Against the fixed fill, computed on holdout 0: on `e12-thin` the cell holds 19 records and the
fixed fill asks 1,250 rows; `expected-size` and `recommended` ask 1,223, within 30 rows of it,
and `coverage`, `coverage-x2`, `coverage-x4` and `coverage-x8` ask 67, 153, 325 and 669. On
`e12-empty` the fixed fill asks 1,269; `expected-size` and `recommended` ask 1,242, within 30
rows, and the coverage rules ask 163, 326, 652 and 1,304, the last 35 rows above it. So the
expected-size rules restate the fixed fill on this cell, and the coverage rules are the ones that
test whether far fewer rows recover the cell.

Reusing E12's runs as the fixed fill holds every arm, split, seed and cutoff equal, so a paired
reading isolates the rule. E6's own runs carry paid Messages arms; E12's arms are all free.

## Consequences

* A run whose `quota_rule` is `fill-to` writes the bytes it wrote before: its request, its
  `experiment` line and every fill.
* `shared`, `rows`, `privacy`, `trace` and `contradictions <run>` rebuild a share-based request
  from the training split and the data's field rows, so each reads an E21 folder as it reads an
  E12 one.
* The sweep reads representativeness and recall per rule; whether a count makes the cell
  representative, and at what cost in recall, is E21's Result.
* The coverage rules ask far fewer rows than the fixed fill on E6's cell, so their flagged
  measures rest on fewer added rows.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| An independence product, 1 - prod(1 - (1 - p)^n) | It assumes concepts appear independently, which a record's linked codes do not; the union bound holds under any dependence (Rationale) | 2026-11-26 |
| A per-concept bound, each concept's own (1 - p)^n at most 0.05 | It bounds each concept's miss alone, so the chance that some concept is missed grows with the concept count | 2026-11-26 |
| A stopping rule inside the sampler | It would change the direct arms' draw and leave every other arm without the count; a count in the request reaches every arm | 2026-11-26 |
| Registering the fixed fill again | E12's runs are the fixed fill at the same arms, splits and seeds; `compare --beside` reads them | 2026-11-26 |

## Enforcement

- Owner: `src/bko/gaps/quota.py` with the request in `src/bko/evaluate/requests.py`, the field in
  `src/bko/evaluate/experiment.py`, the line in `src/bko/evaluate/provenance.py`, the assumed
  value in `src/bko/evaluate/run_folder.py`, the reading in
  `src/bko/evaluate/representativeness.py` and the runs in `src/bko/evaluate/runs_direct.py` and
  `src/bko/evaluate/reference.py`
- Pinned by: `tests/test_gaps_quota.py` (each count's hand fixtures, the multiples, a cell of no
  record, the module's imports), `tests/test_evaluate_quota_rule.py` (the `experiment` line,
  `test_an_older_experiment_line_reads_at_the_assumed_quota_rule`, the default request, the
  refusals, the `quota-rule` line, `test_a_coverage_rule_run_scores_end_to_end` and the
  `shared`, `trace` and `contradictions` readers), `tests/test_evaluate_representativeness.py`
  (the gap, the cell share beside `real`'s, the store and run refusals),
  `tests/test_evaluate_quota_runs.py` (E21's runs, their slices, no paid arm and the reference
  arm), `tests/test_generate_direct.py` (`DRAW_DIGEST`) and `tests/test_evaluate_direct.py`
  (`ADDED_DIGEST`), the fixed fill's draw unchanged

## Reversal Conditions

**On E21's runs, no rule but `fill-to` reads a smaller representativeness gap than the fixed fill
on E6's cell, and none reads removed-cell recall non-inferior to the fixed fill's within the
control reading rule's margin**; then the share-based rules add nothing and are removed. The
check is E21's Result in `docs/EVIDENCE.md`. **An `expected-size` or `recommended` run asks more
than 30 rows from the fixed fill on E6's cell**; the check is the `quota-rule` lines of E21's
`scores.log`.

## Related

- [E1's evaluation: repeated holdouts, one judged classifier, separation judged on paired intervals](e1-evaluation-protocol.md)
- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm](diabetes-subgroup-removal-design.md)
