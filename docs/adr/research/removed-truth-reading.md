---
status: accepted
topic: research
created: 2026-11-26
updated: 2026-11-26
---
# Each fill read against the records its removal took

## Context

A run with a removal (`Experiment.removal`) thins one cell of each holdout's training split
before any arm runs (`bko.evaluate.prepare.prepared`), and every gap-filling arm then adds rows
to that cell. Its downstream scores say whether a classifier gains from the added rows; they do
not say whether the added rows resemble the records the removal took out. `paper-control-results`
asks whether a fill's gain comes from the knowledge it encodes or from the records it copies,
and the removed records are the one truth of the thinned cell that no arm saw. A `rows` store
(ADR [`research/rebuilt-rows-store`](rebuilt-rows-store.md)) holds every fill of a recorded run,
checked against the run's log, and the privacy reading already reads a run's stores, one per
slice (ADR [`research/membership-inference-reading`](membership-inference-reading.md)).

## Decision Drivers

* **Must read the fills the run scored**, from its stores, not fills drawn again.
* **Must measure resemblance with a measure the project already reports**, so the reading sits
  beside the replica fidelity a run writes.
* **Must spend no money** and change no run of record.

## Decision

1. **The truth.** For each holdout, the removed records are the training rows the removal took,
   re-derived with `bko.evaluate.prepare.prepared` from the holdout's split; the store's `split`
   line must be the one the data prepares.
2. **The rows read.** Each arm's gap fill of the holdout, read from the store, keeps its rows in
   the removed cell (age band, sex and label). Replica fills are not read.
3. **The measures.** The fidelity measures a run applies to a replica fill
   (`bko.evaluate.run_scores`, through SDMetrics' quality report of `bko.evaluate.fidelity`):
   the column shapes and column pair trends scores, and each column's shape score, of the
   arm's rows in the cell against the removed records as the reference. A fill with fewer than
   two rows in the cell, or a holdout with fewer than two removed records, reads NaN, as a
   replica of fewer than two rows does.
4. **The arms.** Every arm of the run but `real`, whose rows are the removed records, so each
   of its scores is 1. An arm that adds no rows, such as `none`, reads NaN with its row count 0.
5. **The runs.** A run with no removal is refused: it has no removed records. The stores are
   bound to the named run, or its slices, by the privacy reading's checks (`read` of
   `bko.evaluate.privacy_stores`), and a store of another run is refused.
6. **The output.** One CSV row per holdout, arm and measure, with each column's shape on its own
   row; [`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#evaluation) holds the columns and log lines.
   The reading judges no criterion.

## Rationale

The removed records are the cell's records before the removal, less those the removal kept, so
a fill that resembles them restores what the removal took; that is the resemblance the question
asks about. Reusing the replica's fidelity measures keeps one definition of resemblance in the
project: a reader who knows what a replica's column shapes score means knows what this one
means, and SDMetrics' report already sets its thresholds so the same column pairs count every
time. The cell's age band, sex and label are near constant in both tables, so their column
shapes read near 1 for every arm; the other columns carry the contrast, which is why each
column's shape is written beside the overall scores. Reading the stores through the privacy
reading's checks scores exactly the fills the run's `arm` lines hash.

## Consequences

**Gained**: a free reading, per holdout and arm, of how closely each fill restores the records
its removal took, in the units of the replica fidelity the run already reports.
**Accepted**: a removed cell holds few records (a thin cell's removal leaves a handful), so each
holdout's score is noisy; the reading is descriptive and carries no interval. At a removal
fraction below 1 the kept records stay in the training split, and an arm that copies them can
read close to the removed records without having seen them.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A new distance between the fill and the removed records | A second definition of resemblance beside the replica fidelity the project already reports; see Rationale | 2026-11-26 |
| Scoring the fill against the retained cell | The retained records are what every arm saw, so resemblance to them measures copying, which the privacy reading already reads, not restoration | 2026-11-26 |
| Reading replica fills | A replica fill answers the whole training split's shape, not the removed cell | 2026-11-26 |

## Enforcement

- Owner: `src/bko/evaluate/prepare.py`, `src/bko/evaluate/fidelity.py`
- Pinned by: nothing -- convention only

## Reversal Conditions

**The replica's fidelity measures change**: `bko.evaluate.fidelity.MEASURES` no longer names the
column shapes and column pair trends scores. **A run removes records outside one cell**:
`Experiment.removal` names more than one cell, so the removed records are not one cell's.

## Related

- [The membership-inference reading of a stored run](membership-inference-reading.md)
- [A recorded run's rows are rebuilt once and stored](rebuilt-rows-store.md)
- [The E1 evaluation protocol](e1-evaluation-protocol.md)
