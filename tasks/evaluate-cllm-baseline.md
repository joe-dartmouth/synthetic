---
id: evaluate-cllm-baseline
title: A CLLM-style arm, the LLM without the ontology
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# A CLLM-style arm, the LLM without the ontology

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

`bko` asks the model for records guided by the ontology and the training split's bounds and
spreads, and never shows it a training record (ADR
[`research/record-generation-design`](../docs/adr/research/record-generation-design.md)).
`bko-direct` draws from the fused BKB with no model at all
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#g2-direct-generation-from-the-fused-bkb), G2). So the
arms separate the BKB from the LLM, but nothing separates the ontology from the LLM: no arm asks
the same model for records the way the literature does without an ontology. CLLM
([`seedat-2024`](../docs/literature.md#tabular-generators)) is that way: real training records as
in-context examples, then curation of the generated rows by a classifier's learning dynamics
(confidence and aleatoric uncertainty), with gains in the low-data regime and for
underrepresented subgroups as the brief cites them
([`docs/BACKGROUND.md`](../docs/BACKGROUND.md#tabular-generator-families)).
`bko` beside a CLLM-style arm at the same model and settings reads what the ontology adds over
in-context records. The curation differs from `generate-bkb-curation`'s, which scores records by
their likelihood under the fused BKB rather than by a classifier.

Its E3 reading pairs with E3's recorded scores, and E3's arms stay as its ADR fixes them, as
`evaluate-classical-baselines` says.

## Scope

**In.**

- A new ADR under `docs/adr/research/` fixing which training records each request shows (the
  cell's own, its label group's, or a sample of the split) and how many, the prompt, the
  curation classifier and its checkpoints, the confidence and uncertainty thresholds, how many
  records a cell is asked for so curation can keep its quota, and the seeding, each with its
  reason. It states that this arm departs on purpose from the driver of ADR
  `research/record-generation-design` that no training record is shown, and why that driver
  still holds for `bko`.
- The arm sends through the repository's `Send` seam (`bko.generate.messages`), so the SDK stays
  in its one module, and logs every call as the BKO arms do, so its run replays. It uses
  `SETTINGS`' model, effort and limits, so the prompt and the curation are what differ from
  `bko`.
- **Egress**: real training records leave the machine; no arm sends one today
  ([`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#generation), What leaves the machine). UCI
  Heart Disease's rows are de-identified patient records, which leave under the dataset's public
  licence (CC BY 4.0, `bko.datasets.sources`); Synthea's rows are generated, so they describe no
  person. The ADR and that list state what the arm sends, and the arm refuses a dataset held under
  a DUA unless a named permission for it is recorded
  ([`docs/RESEARCH.md`](../docs/RESEARCH.md#6-datasets)).
- E1, E3 and every run built from them keep their arms and comparators. The arm joins a new named
  run per dataset (`bko.evaluate.experiment.RUNS`), with no criterion, the arm as the candidate
  and `bko` beside it in-run, so the pair needs no reading across files: one on E1's holdouts over
  UCI and one on E3's holdouts, removal and copy rule over the Synthea frame, with `real`.
- Each run's spend estimated before it, in its Runs row, for both Messages arms: the judged run's
  tokens per record asked, plus the shown records' input tokens for the new arm, at
  `SETTINGS.model`'s `PRICES` row, under the named run's own ceiling, and launched under the
  paid-run protocol (`docs/DEVELOPMENT.md`, Evaluation). Each is also read by `compare --beside`
  the judged scores of its dataset (E1's judged run, E3's merged scores) with `--beside-arm bko`,
  to check the in-run `bko` against them.
- A new `docs/EVIDENCE.md` entry: per dataset, validity and drops before and after curation, test
  copies, distance to closest record, replica fidelity and utility and each criterion's measures,
  with no verdict; on Synthea, recall on the removed cell's held-out cases against `real`.

**Out.** Fine-tuning a model (GReaT and REaLTabFormer stay ruled out,
[`docs/IDEAS.md`](../docs/IDEAS.md#cllm-and-great-as-alternative-generators)); adding the arm to
E1's comparison set (that changes [`docs/RESEARCH.md`](../docs/RESEARCH.md#4-success-criteria)
section 4 first); changing what `bko` sends.

## Acceptance

- `tmp_path` tests on a fake client pin the prompt's shown records and the curation's kept rows
  at one seed.
- `E1`'s and `E3`'s arms and comparators are unchanged, pinned by a test.
- The arm refuses a dataset held under a DUA that names no permission for it, pinned by a test.
- ADR `research/record-generation-design` points at the new ADR where it states that no training
  record is shown, naming the arm's departure from that driver.
- Each run is recorded with its spend and replays byte for byte.
- `make check`.
