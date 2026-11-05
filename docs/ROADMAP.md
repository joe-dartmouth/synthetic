# Roadmap

Sources: [RESEARCH.md](RESEARCH.md), the plan of record.

The schedule is the 2-pager's timeline. It runs Oct 10 - Nov 25; the proposal is dated October 7,
2026. The dependency-ordered work units, with their status and owners, are in
[tasks/experiments-index.md](../tasks/experiments-index.md).

Under each milestone, the Work column is the 2-pager's timeline row as written. The Workstream
column names the member whose section 6 assignment covers that work; the 2-pager assigns
workstreams, not rows, so that column is derived.

## Status Update 1

**Date:** Oct 23.

| Dates | Work | Workstream |
|---|---|---|
| Oct 10 - Oct 22 | Reproduce paper's MONDO/DO fusion; build BKO layer; train baselines. | Partiban (fusion reproduction, BKO layer); Joe (baselines) |
| Oct 23 | Status Update #1: gap detection end-to-end; baselines locked. | Partiban (gap detection); Joe (baselines) |

ADR [research/bko-formal-gap-definition](adr/research/bko-formal-gap-definition.md) ties one of its
reversal conditions to this milestone.

## Status Update 2

**Window:** Oct 24 - Nov 6. The 2-pager gives no single date for Status Update #2 within it.

| Dates | Work | Workstream |
|---|---|---|
| Oct 24 - Nov 6 | LLM repair and fusion; evaluation harness; Status Update #2. | Shirley (LLM repair and fusion); Joe (evaluation harness) |

## Working Prototype

**Window:** Nov 7 - Nov 13.

| Dates | Work | Workstream |
|---|---|---|
| Nov 7 - Nov 13 | Streamlit UI; Working Prototype + Technical Brief. | Krish (Streamlit UI, pipeline integration); the Technical Brief is not assigned to a workstream |

## Final Report

**Window:** Nov 14 - Nov 25. The 2-pager gives no separate dates for Demo Day or the Final Report
within it.

| Dates | Work | Workstream |
|---|---|---|
| Nov 14 - Nov 25 | Final evaluation: Slides, Demo Day; Final Report. | Joe (experiments); Krish (demo coordination); all members (final report and presentation) |

## Workload

Each member's workstream, as the 2-pager assigns it (section 6):

| Member | Workstream |
|---|---|
| Partiban Anathurai | Ontology mapping, BKO inference, and gap detection. |
| Shirley Asangi | LLM generation, record validation, and fusion. |
| Joe Laws | Baseline models, evaluation metrics, and experiments. |
| Krish Sapru | Streamlit interface, pipeline integration, and demo coordination. |
| All members | Design reviews, results analysis, and final report/presentation. |

The subpackage each member owns is in the [package map](ARCHITECTURE.md#package-map).

## Beyond the proposal

The 2-pager schedules nothing past the Final Report. The brief (F15) proposed a stretch goal
(distill a small model on the best synthetic data, if ahead of schedule) and a Track B follow-on
(that distillation plus an agentic data-gap-filling copilot). Both are recorded in
[IDEAS.md](IDEAS.md#slm-distillation-stretch-goal) and
[IDEAS.md](IDEAS.md#track-b-agentic-gap-filling-copilot).
