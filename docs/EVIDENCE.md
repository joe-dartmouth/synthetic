# Evidence register

Sources: [2-pager](../papers/proposal-2-pager.pdf) sections 2, 3 and 5; [Santos et al.](../papers/santos-2024.pdf) section 7 for R1.

The results register: one entry per experiment or reproduction, holding its configuration, its result and what the
result was read against. The criteria live in [RESEARCH.md](RESEARCH.md); an entry links the
section that holds them rather than restating them.

## How to read an entry

| Field | Meaning |
|---|---|
| Status | `not yet run`, `running` or `complete` |
| Question | What the experiment answers |
| Configuration | Everything needed to rerun it: dataset and split with the dataset snapshot's identity (download date and checksum; for Synthea its `run.json`: jar checksum, seeds, snapshot date, population size and CSV checksums), the generator's model id, sampling parameters and prompt, baselines and controls, metrics and library versions, seeds, the commit, and how the readings [RESEARCH.md](RESEARCH.md#4-success-criteria) leaves open were fixed |
| Runs | One dated row per run: run id, date, commit, where its logs and artifacts live, and API spend for runs that call a paid API. A rerun adds a row; a superseded or failed run keeps its row with the reason |
| Result | What was measured, with units, and the command that produced it. Empty until the status is `complete` |
| Read against | The RESEARCH.md section whose criteria judge the result, and the verdict: met, not met, inconclusive (its intervals decide neither way), or a null result reported as such |

## R1. MONDO and DO fusion reproduction

| Field | Entry |
|---|---|
| Status | `complete` |
| Question | Does `bko.bkb`, given MONDO and DO fragments built from the pinned releases, rebuild the fusion example of Santos et al. (section 7 of [the paper](../papers/santos-2024.pdf)), and where does it differ? |
| Configuration | Command: `uv run --no-sync python -m bko.ontology reproduce`. Code: `bko.ontology.santos` holds the paper's terms, PTAs, anchor, passes and misprints; `bko.ontology.fragments` the encoding (ADR `research/bko-fragment-encoding`). Environment: Python 3.14, pronto 2.7.3 (`uv.lock`). Releases: MONDO 2026-10-06 and DO 2026-09-30 as `bko.datasets.sources` pins them, where the paper used MONDO 2023-09-12 and DO 2023-10-21. Bridge: MONDO's `skos:exactMatch` annotations, where the paper queried the EMBL-EBI OxO service. Fusion: weight 1 per fragment, `fuse`'s default (each source prior over every fragment's weight, the paper's section 6.1). Instantiation: from the paper's anchor (MONDO's lesion of sciatic nerve, DO's sciatic neuropathy), each pass adds every rule whose antecedents hold, as the paper's algorithm states. Misprints read as the releases give them: Table 1's DOID:114466 is DOID:11446; the fifth MONDO PTA is printed as a second TM4 with antecedent Mononeuritis, and MONDO gives neuritis (MONDO:0002122) as the subclass of inflammatory disease; section 7.1 narrates two bridge rules in its second pass whose antecedents are the anchor |
| Runs | `r1-1` · 2026-10-17 · commit `a8ddddb` on `feat/mondo-do-fusion` (kept in the PR's history) · stdout and stderr in `scratchpad/reproduce.csv` and `scratchpad/reproduce.err` on the runner's machine (gitignored); the command rewrites them byte for byte · no API spend |
| Result | `make reproduce`: no row `differs` of 40. The 11 terms and 17 PTAs are found; DOID:114466 and the fifth MONDO PTA are `erratum` rows. Instantiation from the anchor adds TM1, TD1, TB2 and TB3, then TM2, TD2, TB1 and TB4, where section 7.1 narrates TM1 and TD1, then TM2, TD2 and TB1 to TB4 (`erratum`: TB2 and TB3 take the anchor as their antecedents); the same 8 after two passes, and all 17 at the fixpoint. Inferences a and b: 0.3333333333333333 each against the paper's 0.33; under pybkb's normalization a is 0.5, so the paper's figure follows from its algorithm's every-fragment denominator. The chain DOID:11446 -> DOID:12528 -> DOID:9473 -> DOID:1802 -> MONDO:0002121 -> MONDO:0002122 -> MONDO:0021166 exists in the fused BKB and in neither fragment nor the whole MONDO release; its product is 0.001371742112482853, one source prior of 1/3 per S-node, where the paper gives no figure. Beyond the command: the fused BKB is not mutually exclusive, since MONDO's two S-nodes on mononeuropathy share their source (`tests/test_ontology_santos.py`); and the chain's first two DO steps are axioms the pinned MONDO excludes: `relationship: excluded_subClassOf MONDO:0001543 {source="DOID:11446", source="https://orcid.org/0000-0001-5208-3432"} ! lesion of sciatic nerve` and `relationship: excluded_subClassOf MONDO:0004797 {source="DOID:12528", source="https://orcid.org/0000-0001-5208-3432"} ! mononeuritis of lower limb` |
| Read against | [RESEARCH.md: Detect](RESEARCH.md#detect); verdict: met |

## E1. UCI Heart Disease augmentation

| Field | Entry |
|---|---|
| Status | `not yet run` |
| Question | Does BKO-guided augmentation on UCI Heart Disease meet the success criteria against the baselines and controls? |
| Configuration | `bko.evaluate.experiment.E1` holds the values and ADR [`research/e1-evaluation-protocol`](adr/research/e1-evaluation-protocol.md) the protocol and its reasons. Data: the UCI archive `bko.datasets.sources` names, split by `bko.datasets.split` at each of `E1`'s holdout seeds. Arms: no augmentation, class weighting, SMOTE, ADASYN, CTGAN, TVAE and the BKO arm (registered by `proposal-final-evaluation`), each asked to fill every underrepresented cell up to the threshold of [RESEARCH.md](RESEARCH.md#2-intended-users-and-evaluation-setting) from the holdout's training rows; the four row-adding baselines also fill a replica of the training split's cells; a refused fill is retried with a derived seed, up to `E1`'s attempts. Classifier: `bko.evaluate.classifier`. How the criteria of [RESEARCH.md](RESEARCH.md#4-success-criteria) are read: primary, the candidate beats every other arm, however slightly, in both recall and AUC-PR over the whole test split, each paired difference's corrected interval lying above zero (`E1`'s margin is zero); secondary, the same over the flagged subgroups, the age-sex subgroups that hold an underrepresented cell; privacy, the 5th percentile of the added rows' distance to the training rows reaches `E1`'s share of the test rows' 5th percentile, its interval lying above it; completeness and validity come from the fusion and validation stages, not from this harness. A criterion is met when every interval it reads lies above its threshold, not met when one lies below it at the family-wise level (each of its k intervals read at confidence 1 - (1 - c)/k, c being `E1`'s confidence), and inconclusive otherwise. Not fixed: the BKO arm's generator model id, sampling parameters and prompt |
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
