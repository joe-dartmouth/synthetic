---
id: proposal-datasets
title: UCI Heart Disease and Synthea loading
status: ready
depends_on: --
blocks: proposal-gap-detection, proposal-baselines, proposal-eval-harness
parallel_safe: true
owner: Joe
milestone: status-update-1
paid: false
adrs: --
---

# UCI Heart Disease and Synthea loading

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

Every stage after Represent reads the evaluation data. The 2-pager fixes UCI Heart Disease as the prediction task, with underrepresentation operationalized as fewer than 20 training records per age-sex subgroup-label combination, and Synthea for the gap-detection ablation ([RESEARCH section 2](../docs/RESEARCH.md#2-intended-users-and-evaluation-setting), [section 6](../docs/RESEARCH.md#6-datasets)).

## Scope

**In.**

- Load UCI Heart Disease into `src/bko/datasets/` and report the age-sex subgroup-label cells and their counts.
- Load a Synthea extract and the ablation that removes selected cases, keeping the removed cases as ground truth.

**Out.** MIMIC and the other datasets in [RESEARCH section 6](../docs/RESEARCH.md#6-datasets).

## Acceptance

- The cell counts reproduce from the raw files with one command, and the cells under the threshold are listed.
- Downloaded data lives under the gitignored `/data/`.
- `make check`.
