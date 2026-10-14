# Evidence register

Sources: [2-pager](../papers/proposal-2-pager.pdf) sections 2, 3 and 5.

The results register: one entry per experiment, holding its configuration, its result and what the
result was read against. The criteria live in [RESEARCH.md](RESEARCH.md); an entry links the
section that holds them rather than restating them.

## How to read an entry

| Field | Meaning |
|---|---|
| Status | `not yet run`, `running` or `complete` |
| Question | What the experiment answers |
| Configuration | Everything needed to rerun it: dataset and split with the dataset snapshot's identity (download date and checksum; for Synthea its version and seed), the generator's model id, sampling parameters and prompt, baselines and controls, metrics and library versions, seeds, the commit, and how the readings [RESEARCH.md](RESEARCH.md#4-success-criteria) leaves open were fixed |
| Runs | One dated row per run: run id, date, commit, where its logs and artifacts live, and API spend for runs that call a paid API. A rerun adds a row; a superseded or failed run keeps its row with the reason |
| Result | What was measured, with units, and the command that produced it. Empty until the status is `complete` |
| Read against | The RESEARCH.md section whose criteria judge the result, and the verdict: met, not met, or a null result reported as such |

## E1. UCI Heart Disease augmentation

| Field | Entry |
|---|---|
| Status | `not yet run` |
| Question | Does BKO-guided augmentation on UCI Heart Disease meet the success criteria against the baselines and controls? |
| Configuration | Not fixed. The setting is in [RESEARCH.md](RESEARCH.md#2-intended-users-and-evaluation-setting), the baselines, controls and open readings in [RESEARCH.md](RESEARCH.md#4-success-criteria), and the metrics and libraries in [RESEARCH.md](RESEARCH.md#5-evaluation-contract) |
| Result | None |
| Read against | [RESEARCH.md: success criteria](RESEARCH.md#4-success-criteria) |

## E2. Synthea gap-detection ablation

| Field | Entry |
|---|---|
| Status | `not yet run` |
| Question | Does BKO gap detection find the cases deliberately ablated from a Synthea population, and how do BKO-detected gaps compare with count-based ones? |
| Configuration | Not fixed: which cases are ablated, and from which generated population |
| Result | None |
| Read against | [RESEARCH.md: intended users and evaluation setting](RESEARCH.md#2-intended-users-and-evaluation-setting) for the ablation's design, and [RESEARCH.md: two kinds of gap](RESEARCH.md#two-kinds-of-gap-evaluated-separately) for the separate evaluation of count-based and BKO-detected gaps. The 2-pager sets no pass threshold for this experiment |
