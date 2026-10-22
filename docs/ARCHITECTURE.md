# Architecture

Sources: [2-pager](../papers/proposal-2-pager.pdf) sections 3, 4, 5 and 6; [research brief](../papers/research-brief.pdf) F16; facts verified outside the PDFs (pybkb, ofs_app, synthcity resolution).

> **Status: built.** Each stage's README says what the stage holds.

How the system fits together. The method each stage implements is in
[RESEARCH.md](RESEARCH.md#3-approach); each stage's README is the home for its role, extra, owner
and status.

## Tools

The tools the 2-pager names (section 4):

| Concern | Tool | Notes |
|---|---|---|
| Front end | Streamlit over a Python backend | |
| Ontology access | owlready2, pronto, OLS4 | The repository reads pinned OBO releases with pronto; owlready2 and OLS4 are not dependencies (ADR [`research/ontology-mapping-design`](adr/research/ontology-mapping-design.md)) |
| Record generation | Anthropic Messages API | |
| Baselines | CTGAN (Conditional Tabular GAN), TVAE (Tabular Variational AutoEncoder), SMOTE (Synthetic Minority Over-sampling Technique), ADASYN (Adaptive Synthetic Sampling) | CTGAN and TVAE come from SDV, SMOTE and ADASYN from imbalanced-learn |
| Evaluation | scikit-learn, SDMetrics, synthcity | synthcity is not a dependency: it does not resolve on Python 3.14 (ADR [tooling/python-314-without-synthcity](adr/tooling/python-314-without-synthcity.md)). Evaluation uses scikit-learn and SDMetrics ([RESEARCH.md](RESEARCH.md#5-evaluation-contract)) |
| Code | A GitHub repository | |
| Data | UCI Heart Disease and Synthea, both openly available | [RESEARCH.md](RESEARCH.md#6-datasets) |

**Principal risk.** The 2-pager states: "The principal risk is that no public BKB library exists, so
the inference layer will be built by the team." A public BKB library exists ([pybkb](#pybkb)), and
`bko.bkb` ports its core under GPL-3.0 (ADR [`research/bkb-layer-source`](adr/research/bkb-layer-source.md)).

## Pipeline

```mermaid
flowchart LR
    R["Represent<br/>bko.ontology, bko.bkb, bko.datasets"] --> D["Detect<br/>bko.gaps, bko.bkb"]
    D --> G["Generate<br/>bko.generate"]
    G --> V["Validate<br/>bko.validate"]
    V --> E["Evaluate<br/>bko.baselines, bko.evaluate"]
    E -.->|"BKO arms<br/>bko.gaps, bko.generate, bko.validate"| D
    V -.->|"Fuse<br/>bko.fusion"| R
```

| Stage | Takes | Produces |
|---|---|---|
| Represent | MONDO, HPO and the dataset | A BKO per cell and one for every record: the ontology fragments fused with a dataset fragment, weighted by observed frequency |
| Detect | The BKO | BKO-detected (cell, statement) gaps under three definitions, with count-based gaps alongside for comparison |
| Generate | A request of rows per cell, each cell's gaps, and a guidance level | Candidate records per cell, decoded and unfiltered, with a placeholder per record that did not decode, through the Anthropic Messages API |
| Validate | Candidate records | The records that pass the consistency checks and numeric range constraints |
| Fuse | The validated records and the training split | The joined records, and each gap definition's pairs before and after the join |
| Evaluate | Augmented training data, the baselines and the controls; its BKO arms run Detect, Generate and Validate on each holdout's training rows | The success criteria ([RESEARCH.md](RESEARCH.md#4-success-criteria)) |

The dotted edge from Evaluate is `bko.evaluate`'s BKO arms, which run gap detection, generation and
validation (`bko.gaps`, `bko.generate`, `bko.validate`) on each holdout's training rows. The dotted
edge from Validate is the fusion of validated records back into the BKO, after which the completeness
criterion counts the gaps that close; [BACKGROUND.md](BACKGROUND.md#fusion) explains that reading
of "after fusion". The Streamlit front end (`bko.app`) sits over every stage.

## Run records

Every experiment run, and every paid generation run outside an interactive app session, is
recorded in [EVIDENCE.md](EVIDENCE.md): its configuration, including the dataset snapshot and the
generator's model and sampling parameters, and one Runs row per run with its id, date, commit, log
location and API spend. Each run's artifacts live in its folder under `runs/`
([DEVELOPMENT.md](DEVELOPMENT.md#runs)).

## Package map

Which pipeline stage each subpackage serves. Each stage README is the one home for its role, extra,
owner and status, and ADR [`tooling/src-layout`](adr/tooling/src-layout.md) records the layout; what
each extra installs is in [DEVELOPMENT.md](DEVELOPMENT.md#environment).

| Subpackage | Stage |
|---|---|
| [`bko.datasets`](../src/bko/datasets/README.md) | Represent (loading the records) and Evaluate (training data) |
| [`bko.ontology`](../src/bko/ontology/README.md) | Represent (ontology loading, concept-to-field mapping, BKO fragments) |
| [`bko.bkb`](../src/bko/bkb/README.md) | Represent and Detect (BKB fusion, instantiation from an anchor, reasoning) |
| [`bko.gaps`](../src/bko/gaps/README.md) | Represent (the dataset fragments) and Detect |
| [`bko.generate`](../src/bko/generate/README.md) | Generate |
| [`bko.validate`](../src/bko/validate/README.md) | Validate |
| [`bko.fusion`](../src/bko/fusion/README.md) | Fuse (validated records back into the BKO, and gap closure) |
| [`bko.baselines`](../src/bko/baselines/README.md) | Evaluate (baseline augmentation) |
| [`bko.evaluate`](../src/bko/evaluate/README.md) | Evaluate (its BKO arms run Detect, Generate and Validate through `bko.gaps`, `bko.generate` and `bko.validate`) |
| [`bko.app`](../src/bko/app/README.md) | Front end over every stage |

Fusion appears in two places: `bko.fusion`, which Shirley owns, joins validated records to the
training records and reads what closes, and `bko.bkb`, which Partiban owns, holds the fusion
operation that `bko.gaps.detect.represent` calls.

## Demo-Day flow

The brief (F16) proposed this flow. Its premise: Track A requires a product, so the demo must be
clickable, not a notebook, and is buildable with a Streamlit or Gradio front end, a Python back end
and an LLM API.

1. **Upload or pick a dataset** (default: UCI Heart Disease or a Synthea extract). The user sees the
   row count and class balance.
2. **"Analyze gaps".** The app maps columns to an ontology (HPO, MONDO or SNOMED through the
   OLS/BioPortal API) and renders a visual coverage map or tree, with under-represented concepts in
   red (the brief's "Bayesian coverage score"). The brief calls this the visual wow-moment and the
   ontology differentiator.
3. **"Fill gaps".** The user picks a method from a dropdown (CLLM, GReaT, CTGAN, TVAE or SMOTE) and
   the app generates synthetic rows targeting the red regions. Ontology or SHACL validation rejects
   implausible rows live, with a rejected-row counter on screen.
4. **"Human-in-the-loop"** (an optional tab). For the thinnest gaps, the app drafts questions for a
   human expert or an LLM to supply real values.
5. **"Evaluate".** A one-click dashboard: fidelity (KS/TVD, correlation heatmaps), a
   TSTR-versus-TRTR AUC/F1 bar chart, and privacy (DCR). The headline number is the rare-class
   recall/AUC improvement over the no-augmentation baseline.
6. **Download** the augmented dataset and a one-page "synthetic data quality report".

What the judge sees: an upload, an ontology coverage tree that lights up red and green, a choice of
generation method, gaps filling while implausible rows are rejected, and a dashboard showing the
downstream detection model getting measurably better.

The 2-pager's method choices govern where they differ from the flow:

| Point | Brief F16 | Plan |
|---|---|---|
| Front end | Streamlit or Gradio | Streamlit |
| Step 1 | Upload or pick a dataset (UCI Heart Disease or a Synthea extract) | UCI Heart Disease, picked, not uploaded |
| Step 2 | HPO, MONDO or SNOMED through OLS/BioPortal; a Bayesian coverage score | MONDO and HPO through pronto, from pinned OBO releases; gaps are ungrounded BKO statements, with count-based gaps alongside |
| Step 3 | A dropdown of CLLM, GReaT, CTGAN, TVAE and SMOTE; SHACL validation | Generation through the Anthropic Messages API, compared against CTGAN, TVAE, SMOTE and ADASYN; validation against the consistency checks and range constraints; the page's scores table shows a BKO fill's generated, passed and dropped counts once the fill ends, not live |
| Step 4 | Optional human-in-the-loop tab | Not planned ([IDEAS.md](IDEAS.md#human-in-the-loop-expert-questions)) |
| Step 5 | Improvement over no augmentation | The success criteria, which add the class-weighting control and the baselines ([RESEARCH.md](RESEARCH.md#4-success-criteria)); the page shows one split's measures, with no replica fidelity or train-on-synthetic reading and no verdict; earlier runs of record show from their run folders |
| Step 6 | Download the augmented dataset and a one-page "synthetic data quality report" | The augmented training set and the scores, as CSV; no report |

## Prior code

### pybkb

[pybkb](https://github.com/di2ag/pybkb), by Chase Yakaboski, is a Python package for reasoning and
learning with Bayesian Knowledge Bases. The facts below were read from its repository through the
GitHub API on 2026-10-16.

- **License**: GPL-3.0, the repository's own license (ADR
  [`tooling/gpl-3-license`](adr/tooling/gpl-3-license.md)), so its code may be ported, and `bko.bkb`
  ports its core (ADR [`research/bkb-layer-source`](adr/research/bkb-layer-source.md)).
- **State**: the default branch, `master`, ends at commit `1e1b62a02faa0abc73db4597076c2d38c1313df6`,
  dated 2023-07-28; not on PyPI; its README says documentation is to come.
- **Pins**: Python ~3.11, torch 2.0.1, numpy 1.24.4, pandas 1.x, gurobipy 10.0.2, pygobnilp (a
  Bitbucket git dependency) and ray. It therefore cannot be a dependency on this repository's
  Python 3.14, but its core is portable.
- **`learn/`** needs gobnilp and gurobi.

The core, with line counts at that commit:

| Path | Lines | Holds |
|---|---|---|
| `pybkb/bkb.py` | 682 | The BKB structure |
| `pybkb/fusion/__init__.py` | 198 | `fuse(bkfs, reliabilities, ...)` |
| `pybkb/reason/bkb.py` | 160 | `BKBReasoner.update`, which raises `NotImplementedError`, and `heuristic_update_complete_fused`, the one working inference |
| `pybkb/utils/probability.py` | 779 | Probability routines, using numba; they serve `BKB.score` only |

[RESEARCH section 9](RESEARCH.md#9-source-reconciliation) records how this changes the 2-pager's
principal risk.

### ofs_app

[di2ag/ofs_app](https://github.com/di2ag/ofs_app) describes itself as "Ontology Fusion Service using
Bayesian Knowledge Base Fusion and Reasoning". Read through the GitHub API on 2026-10-13, it is a
TRAPI service template with no fusion code and no license, and it is not used.
