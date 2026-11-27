# Development: the environment, the gate and its hazards

How this repository is built, checked and run. [`AGENTS.md`](../AGENTS.md) keeps the working
summary; this file is the full record behind it. The repository's own checkers are described in
[`tools/README.md`](../tools/README.md).

## Environment

Requires `uv`. The interpreter version is the one in `.python-version`, and `uv` downloads it when
the machine lacks it. `make setup` installs the `dev`, `data`, `synth`, `eval`, `ontology`, `llm`,
`app` and `slm` extras from the lock, and `source .venv/bin/activate` enters the environment.

- **Runtime dependencies are optional extras, declared and locked**: `data` (numpy, pandas, scipy),
  `synth` (sdv, imbalanced-learn, scikit-learn, torch), `eval` (scikit-learn, sdmetrics, scipy),
  `ontology` (pronto), `llm` (anthropic), `app` (streamlit) and `slm` (transformers, peft,
  tokenizers, safetensors, accelerate, torch) are installed by `make setup`.
  `synth` brings `torch` to every machine, and on Linux the lock adds the NVIDIA CUDA packages and `triton`. ADR
  [`tooling/uv-ruff-ty`](adr/tooling/uv-ruff-ty.md) states when an extra joins `make setup`.
  `make sync` installs every extra.
- **Try a new extra before it joins `make setup`, the lock-faithful way**: copy the `setup` recipe
  out of the `Makefile` and append `--extra <name>`. `uv sync` is exact, so a bare
  `uv sync --extra <name>` removes the others. Never `uv pip install -e ".[...]"`: it resolves
  fresh rather than from the lock, and `make synced` refuses the result.
- **Change a dependency with `uv lock`, then `make setup`.** Both install targets run
  `uv sync --locked`, which refuses a lock that disagrees with `pyproject.toml` instead of
  rewriting it, and every gate step runs `uv run --no-sync`, so nothing but an explicit `uv lock`
  writes `uv.lock`. `tests/test_repo_gate.py` compares the two offline as well.
- **Build backends are pinned** in `[tool.uv] build-constraint-dependencies`, because uv resolves
  them outside the lock: hatchling for `bko`.
- **ManimGL lives in its own environment**, `video/.venv`, outside the lock ([Video](#video)).

## Data

The evaluation data is downloaded or generated, never committed. It lives under `data/` at the
repository root, which `.gitignore` ignores, and only
[`bko.datasets`](../src/bko/datasets/README.md) writes there, apart from the classified OBO files
`python -m bko.ontology elk` keeps under `data/elk/` ([ELK comparison](#elk-comparison)) and what
converting an OWL-only release writes beside its OWL ([Ontologies](#ontologies)): the OBO, written
as a hidden partial file renamed into place, and ROBOT's stderr, kept only when the conversion
fails. Each download (UCI Heart Disease, UCI Diabetes 130-US Hospitals, SUPPORT2, the ACS
Massachusetts file, the NHANES component files, the NHAMCS 2022 ED file, the PhysioNet 2012
files, the Synthea jar, the MONDO, HPO and DO releases, OccO's OWL file, and the OWL files and
ROBOT jar the `elk` command reads) is pinned by size and sha256 in `bko.datasets.sources`. A
download that differs from its pin is refused; the commands delete and fetch again a snapshot on
disk that fails its pin, and the UCI and Diabetes loaders refuse one. A generated population is checked against its own `run.json`: the
commands regenerate one whose CSVs or recorded settings differ.

Both commands run inside the project's virtual environment (`.venv`, built by `make setup`), write
their report to stdout as CSV and their logs to stderr. The module exits 0 when done, 1 on an
unexpected error (with its traceback), 2 when the command could not run, and 3 when the data
contradicts its pin or schema; `make` reports any failure as its own exit 2. To use another data
root or to see the module's own code, run it directly:
`uv run --no-sync python -m bko.datasets <cells|synthea> --data-dir <dir>`.

- **The cell counts**: `make cells` runs `python -m bko.datasets cells`. It fetches UCI Heart
  Disease if `data/` lacks it, verifies it, and writes the age-sex subgroup-label cell counts of
  the training split, the same bytes on every run. The underrepresented cells and the snapshot
  identity go to stderr.
- **The Synthea ablation**: `make synthea` runs `python -m bko.datasets synthea`. It needs a JDK,
  17 or newer: `JAVA_HOME` when set, else `java` on `PATH`, so `JAVA_HOME=<jdk home> make synthea`
  picks one. The first run fetches the pinned Synthea jar and generates the population
  `DEFAULT_RUN` in `bko.datasets.synthea_run` names; later runs reuse it. It removes the cases
  `DEFAULT_ABLATION` in `bko.datasets.ablation` names and writes `age_band`, `sex`, `target`,
  `before` and `after` for every cell.
- **The Synthea frame**: E3's frame (ADR
  [`research/synthea-downstream-design`](adr/research/synthea-downstream-design.md)) needs only the
  pinned population `DEFAULT_RUN` names: `score e3` checks it against its `run.json` with no Java
  and never generates it, so `make synthea` runs once first. It keeps the patients alive on the
  snapshot date and aged 40 and over, and the sha256 of its records is the `digest` of the
  `data dataset=synthea frame=synthea records=<n> digest=<sha256> agreement=none` line `score`
  logs.
- **The Diabetes frame**: `bko.datasets.diabetes` reads `diabetic_data.csv` and
  `IDS_mapping.csv` from the verified `uci-diabetes-130` snapshot without extracting it. It keeps
  each patient's first encounter (the lowest `encounter_id`), then drops encounters discharged
  expired or to hospice (`discharge_disposition_id` 11, 13, 14, 19, 20 and 21, the rows
  `IDS_mapping.csv` names Expired or Hospice), `Unknown/Invalid` gender and age brackets under 20,
  the cohort of Strack et al. 2014 (`strack-2014` in [`literature.md`](literature.md)) restricted
  to adults. The label is 1 for readmission in under 30 days (`readmitted` `<30`), 0 otherwise. A
  record holds the age decade index (2 for `[20-30)` to 9 for `[90-100)`), sex, race (`?` is code
  0, not recorded), the eight count fields, `A1Cresult` and `max_glu_serum` (`None` is a state,
  not missing), the eight drug columns most often given, `change`, `diabetesMed`, and one
  indicator per mapped ICD-9 category, set when any of `diag_1` to `diag_3` falls in it;
  `record_id` is the record's 1-based position in encounter order. `encounter_id` and
  `patient_nbr` never enter the frame. `bko.ontology.frames` holds its `FrameSpec`: age in
  decades (UCUM `10.a`), band edges 4, 6 and 8, each coded column's meanings the archive's own state
  strings, and one field row per ICD-9 and lab row of `bko.ontology.icd9`; `score` refuses records
  whose columns or codes the frame lacks. The loader refuses, as a contract error, a member over
  its size cap, a value outside the archive's schema (a whole number in other than ASCII digits or
  past int64 among them), and a cohort with no record or no record readmitted in under 30 days.
  `tests/test_datasets_diabetes_pinned.py` loads the snapshot under the data root and checks the
  cohort's size, its label count, the frame's ranges and the pinned discharges and drugs; the gate
  skips it, and `--releases` runs it, as for [the ontology release tests](#ontologies):

      uv run --no-sync pytest tests/test_datasets_diabetes_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

  It reads and never fetches, and an absent snapshot fails it; `bko.datasets.fetch.ensure`
  fetches the snapshot, as `score` does for a run over Diabetes 130.
- **The Diabetes codes frame**: the dataset `diabetes-codes` reads the same cohort and label with
  `load_diabetes(..., diagnoses=True)` and no indicators: in their place, `diag_1`, `diag_2` and
  `diag_3` hold each token as `bko.datasets.diagnoses.code_number` (`?` is 0). Its `FrameSpec`
  (`bko.ontology.code_frames.diabetes_codes_frame`) is built from the loaded records, since the
  three columns' codebook is the codes they hold; its field rows are one per mapped code of
  `bko.ontology.icd9_codes` and diagnosis column, then the lab rows (ADR
  [`research/diabetes-diagnosis-codes`](adr/research/diabetes-diagnosis-codes.md)). The runs
  `e15-thin` and `e15-empty` and their slices read it (`docs/EVIDENCE.md`, E15); the
  `contradictions` census does not.
- **The TableShift frame**: `bko.datasets.tableshift` rebuilds TableShift's Hospital Readmission
  partition of the same archive in numpy (ADR
  [`research/tableshift-comparison`](adr/research/tableshift-comparison.md)). It drops the rows
  whose race is `?`, puts `admission_source_id` 7 out of domain, and splits each domain with
  `RandomState(264738)` as TableShift's `train_test_split` calls do; `record_id` is the row's
  1-based position in the file. `bko.datasets.tableshift_frame` holds the frame `tableshift`:
  every column but the identifiers, `admission_source_id` and `readmitted`, `sex` male against
  every other gender, `race` Caucasian against every other race, the diagnoses as
  `code_number`, band edges 1, 2, 4, 6 and 8, and the label 1 for any readmission. Its codebooks
  for `weight`, `payer_code`, `medical_specialty`, the two id columns and the diagnoses are read
  from the kept rows, so the frame is built from them. The dataset `tableshift-id` loads the
  in-domain training split with the in-domain test split held, and `tableshift-ood` the same
  training split with the out-of-domain test split held; every holdout seed holds out the same
  test records (`bko.evaluate.data.held_split`), and the field rows are the lab rows alone.
  `attribution` refuses both, since it loads no records to build their frame.
  `tests/test_datasets_tableshift_pinned.py` checks the partition's sizes against TableShift's
  Table 12 and the flagged cells of the in-domain training split; the gate skips it, and
  `--releases` runs it:

      uv run --no-sync pytest tests/test_datasets_tableshift_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

- **The SUPPORT2 frame**: `bko.datasets.support2` reads `support2.csv` from the verified
  `support2` snapshot without extracting it (ADR
  [`research/support2-frame`](adr/research/support2-frame.md)). `record_id` is each row's leading
  row number; the label is `hospdead`, in-hospital death; a record whose `scoma` is missing is
  dropped and counted in the `cohort` line. `age` is floored to whole years; `race` and `income`
  hold code 0 for not recorded; each lab or vital is a coded state, 0 for not measured, then its
  band at the SIRS or SOFA cutoffs, or at the published file's tertiles for the unmapped ones.
  The columns the publisher warns against and those known only at or after the outcome never
  enter. `bko.datasets.support2_frame` builds its frame from the records, band edges 50, 65 and
  80, and `bko.ontology.support2_rows` holds its field rows; it is the first entry of
  `bko.ontology.catalog`, so `score`, `gaps support2` and `contradictions --dataset support2`
  read it through its catalog entry. `tests/test_datasets_support2_pinned.py` checks the
  archive's sha256, the cohort, the training split's death rate and the tertile cutoffs; the gate
  skips it, and `--releases` runs it:

      uv run --no-sync pytest tests/test_datasets_support2_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

  It reads and never fetches, and an absent snapshot fails it; `gaps support2` fetches it.
- **The ACS frame**: `bko.datasets.acs` reads `psam_p25.csv`, the ACS 2024 1-Year person file for
  Massachusetts, from the verified `acs-ma-2024` snapshot without extracting it (ADR
  [`research/acs-frame`](adr/research/acs-frame.md)). Every column is read as text, so `OCCP`
  keeps its leading zeros; the header must be the published one and every row as wide. The
  cohort is Folktables' ACSIncome filters in its order, `AGEP` above 16, `PINCP` above 100,
  `WKHP` above 0 and `PWGTP` at least 1, counted in the `cohort` line; `record_id` is a kept row's
  position among the kept rows; the label is unadjusted `PINCP` above 50,000. A blank in a used
  column of a kept row is refused, as is an `OCCP` code whose rows carry two `SOCP` codes.
  `bko.datasets.acs_frame` builds its frame from the records, band edges 30, 45 and 60, and
  `bko.ontology.acs_rows` holds its field rows: four disability items on HPO and each `OCCP` code
  on the Occupation Ontology. Its catalog entry holds its own release map, HP and OccO, so
  `score`, `gaps acs` and `contradictions --dataset acs` read OccO, converted from OWL with ROBOT
  (a JDK). `tests/test_datasets_acs_pinned.py` checks the archive's sha256, the cohort and the
  training split's label rate; the gate skips it, and `--releases` runs it:

      uv run --no-sync pytest tests/test_datasets_acs_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

  It reads and never fetches or converts, and an absent snapshot fails it; `gaps acs` fetches
  and converts.
- **The NHANES frame**: `bko.datasets.nhanes` reads the NHANES component files, SAS XPORT, of August
  2021 to August 2023 from their verified `nhanes-*` snapshots with `pandas.read_sas`, never writing
  them elsewhere (ADR [`research/nhanes-frame`](adr/research/nhanes-frame.md)). Each file's columns
  must be the published file's, its `SEQN` values whole and unique, and every other file's `SEQN`
  values among the demographics'; a file `pandas.read_sas` warns may be corrupted is refused.
  Demographics and medical conditions join inner on `SEQN`, `record_id`; the exam and laboratory
  files join left, so an adult who was only interviewed has every measurement not measured. Adults
  of 20 and over are kept; the label is 1 when any of the heart-failure, coronary-disease,
  heart-attack and stroke items is yes and 0 when all four are no, and any other adult is left out,
  counted in the `cohort` line with each coded field's measured records. `age` is as published,
  top-coded at 80. Blood pressure is the mean of the readings present; it, glycohemoglobin, total
  and HDL cholesterol are coded at their cited cutoffs and body mass index at the published file's
  tertiles. `bko.datasets.nhanes_frame` builds its frame from the records, band edges 40 and 60, and
  `bko.ontology.nhanes_rows` holds its field rows, so `score`, `gaps nhanes` and `contradictions
  --dataset nhanes` read it through its catalog entry. `tests/test_datasets_nhanes_pinned.py` checks
  each file's sha256, the fixture writer's layout against each file, the adult frame's, composite's
  and training split's counts ADR `research/nhanes-frame` records and the tertile cutoffs; the gate
  skips it, and `--releases` runs it:

      uv run --no-sync pytest tests/test_datasets_nhanes_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

  It reads and never fetches, and an absent snapshot fails it; `gaps nhanes` fetches them.
- **The NHAMCS frame**: `bko.datasets.nhamcs` reads `ed2022`, the fixed-width NHAMCS 2022
  emergency-department file, from the verified `nhamcs-ed-2022` snapshot without extracting it
  (ADR [`research/nhamcs-frame`](adr/research/nhamcs-frame.md)). Each item's columns are a code
  table from the Stata dictionary; a line of the wrong width is refused, naming its line, and so is
  a value outside its item's codes or a `NOCHRON` at odds with the chronic flags. `record_id` is
  the hospital code times 1000 plus the visit code. Every visit is kept; the label is `ADMITHOS`,
  admission to this hospital. `age` is as published, top-coded at 94. The five diagnosis tokens are
  kept as reversible base-37 numbers (`bko.datasets.icd10_tokens`), and each chronic flag is 0, 1,
  or 2 where the whole chronic item is blank, counted in the `cohort` line.
  `bko.datasets.nhamcs_frame` builds its frame from the records, band edges 18, 45 and 65, and
  `bko.ontology.nhamcs_rows` holds its field rows, so `score`, `gaps nhamcs` and `contradictions
  --dataset nhamcs` read it through its catalog entry. `tests/test_datasets_nhamcs_pinned.py`
  checks the archive's sha256, the fixture's real record against the file, and the records' and
  training split's counts ADR `research/nhamcs-frame` records; the gate skips it, and `--releases`
  runs it:

      uv run --no-sync pytest tests/test_datasets_nhamcs_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

  It reads and never fetches, and an absent snapshot fails it; `gaps nhamcs` fetches it.
- **The PhysioNet 2012 frame**: `bko.datasets.physionet2012` reads the PhysioNet/CinC Challenge
  2012 set's three archives, sets A and B as zips and set C as a gzipped tar, from their verified
  `physionet2012-*` snapshots without extracting them, through `bko.datasets.archive`'s `members`
  and `tar_members`, which refuse a link, a device, an absolute or parent path, a name outside the
  set, a repeated name, an oversize member or total and a damaged archive (ADR
  [`research/physionet2012-frame`](adr/research/physionet2012-frame.md)). Each stay is one record,
  its `RecordID` the id; each set's outcomes file must hold one line per stay of the set, and the
  label is in-hospital death. A stay whose age or sex is -1 is left out, and a line with an empty
  parameter name is skipped; the `cohort` line counts both. Each variable is reduced over the 48
  hours, over values at or above 0, to its minimum and maximum, coded at its SIRS or SOFA cutoff.
  `bko.datasets.physionet2012_frame` builds its frame, band edges 50, 65 and 80, and
  `bko.ontology.physionet2012_rows` holds its field rows, so `score`, `gaps physionet2012` and
  `contradictions --dataset physionet2012` read it through its catalog entry.
  `tests/test_datasets_physionet2012_pinned.py` checks the six files' sha256s, the fixture's real
  stays against the archive, and the counts ADR `research/physionet2012-frame` records; the gate
  skips it, and `--releases` runs it:

      uv run --no-sync pytest tests/test_datasets_physionet2012_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

  It reads and never fetches, and an absent snapshot fails it; `gaps physionet2012` fetches them.
- **Agreements**: each `Source` in `bko.datasets.sources` also names the data use agreement its
  download is held under (`Agreement(name, sending)`, `sending` the clause or written permission
  that allows sending its records to a third-party API), None for open data; every pinned source's
  is None. `score` carries it in `bko.evaluate.data.Data.agreement`, and every paid arm
  (`bko.evaluate.experiment.PAID_ARMS`) refuses, exit 2, data whose agreement records no
  `sending` permission, before the call log or the candidates file exists (ADR
  [`research/cllm-arm-design`](adr/research/cllm-arm-design.md), Egress).
- **The ELK pins**: `MONDO_OWL` and `HPO_OWL` pin `mondo.owl` and `hp.owl` from the same MONDO
  and HPO releases as the OBO pins, and `ROBOT_JAR` pins ROBOT v1.9.11's `robot.jar` (BSD-3), each
  open with no agreement. Only `python -m bko.ontology elk` reads them
  ([ELK comparison](#elk-comparison)).
- **Layout**: a download's snapshot is `data/<source>/<first 16 hex digits of its sha256>/`, for
  example `data/uci-heart-disease/b17cd273da9ce1ca/`, holding the download and `source.json`. A
  population is `data/synthea/<key>/`, the key naming the jar's sha256 prefix, the seeds, the size,
  the snapshot date and the state; it holds Synthea's `csv/`, its log `synthea.log` and `run.json`.
  A failed generation leaves its log at `data/synthea/<key>.log`.
- **Identity**: `source.json` holds `url`, `size`, `sha256` and `retrieved_at` (the download time,
  UTC). `run.json` holds `run` (the parameters), `config`, `jvm_options`, `environment`,
  `jar_sha256`, `java_version`, `started_at`, `finished_at` and, per CSV, `rows` and `sha256`. An
  experiment copies them into its configuration in
  [EVIDENCE.md](EVIDENCE.md#how-to-read-an-entry).
- **The gate is offline.** No test it runs reads `data/`, and each builds its input in `tmp_path`;
  the release tests read `data/` only when `--releases` is passed ([Ontologies](#ontologies), and
  The Diabetes frame above). No
  test runs Java, no gate test reaches the Anthropic API, and the one test that reaches the network
  is under [Gate hazards](#gate-hazards).

## Baselines

`bko.baselines` ([README](../src/bko/baselines/README.md)) builds each comparison method's training
set from the training split `bko.datasets` computes (ADR
[`research/baseline-augmentation-design`](adr/research/baseline-augmentation-design.md)). One
command per method, inside `.venv`:

    uv run --no-sync python -m bko.baselines <none|class-weight> [--data-dir <dir>]
    uv run --no-sync python -m bko.baselines <smote|adasyn|ctgan|tvae|tabddpm|gaussian-copula|cart|bayes-net|label-bootstrap|row-resample|marginals|eb-marginals|hb-marginals> --rows <n> [--strata cell|label] [--seed <s>] [--data-dir <dir>]

- **What it does**: it fetches UCI Heart Disease if the data root lacks it, splits it with
  `SPLIT_SEED`, water-fills `--rows` over the strata (cells unless `--strata label`), runs the
  method and writes the training set to stdout as CSV: `record_id` (negative for an added row), the
  predictors, `target` and `weight`. `--seed` defaults to `BASELINE_SEED` in
  `bko.baselines.request`.
- **What it logs**: stderr gets the split line `cells` prints, one `quota` line per stratum, a line
  with the method (and, for a row-adding method, its seed and strata), the real, synthetic and total
  row counts and the output's sha256, the fit and fill lines of CTGAN, TVAE, TabDDPM, the
  Bayesian network, the Gaussian copula and sequential CART
  (`fit holdout=<h> fill=<kind> method=<name> seconds=<s>`, each method's own name, and
  `holdout=-1 fill=none` outside a harness fill), the pooled methods'
  `pool holdout=%d fill=%s method=%s key=%s own=%d banded=%d` line per cell, `hb-marginals`
  among them, `eb-marginals`' `eb holdout=%d fill=%s kappa=%s` line and `hb-marginals`'
  `hb holdout=%d fill=%s label=%s column=%s tau2=%s tau3=%s evidence=%.6f iterations=%d` line per
  fit, one per label and modelled column, and the method's parameters (the label bootstrap's are
  each stratum's pool size).
  `row-resample`, `marginals`, `eb-marginals` and `hb-marginals` refuse `--strata label` with
  exit 2. An experiment copies them into its configuration in
  [EVIDENCE.md](EVIDENCE.md#how-to-read-an-entry); the commit's `uv.lock` gives the library
  versions.
- **Exit codes**: as in [Data](#data), with 2 also for a request a method cannot fill and 3 for a
  method whose output breaks the contract.
- **Reproducibility**: a seed gives the same bytes on one machine with the same lock; CTGAN, TVAE
  and TabDDPM promise no more, so the logged sha256 is the record. The Gaussian copula gives the
  same rows at every seed, since SDV fixes its sampling seed; sequential CART's and the Bayesian
  network's rows follow the seed alone, as the label bootstrap's do. `row-resample`, `marginals`,
  `eb-marginals` and `hb-marginals` each read one generator of their own stream at the seed, so
  their rows follow it alone; `eb-marginals`' `kappa` and `hb-marginals`' fits read the training
  split alone, and the fits give the same scales on one machine with the same lock.

## Evaluation

`bko.evaluate` ([README](../src/bko/evaluate/README.md)) scores every arm of E1 on its repeated
holdouts and judges a candidate arm against E1's criteria (ADR
[`research/e1-evaluation-protocol`](adr/research/e1-evaluation-protocol.md)). `score` spends money:
its paid arms (`bko.evaluate.experiment.PAID_ARMS`: E1's three BKO arms, B3's CLLM arms beside
`bko`, G7's `bko-curated` and the perturbation runs' `bko-direct-perturbed`) call the Anthropic
Messages API, so it needs the `llm` extra and an Anthropic credential, and every call goes to the
call log at `--log`, which must not exist; its `run` line names the run's frame under `frame`.
Whenever a paid arm runs, live, with `--replay`, with `--resume` or with `--rehearse`,
`--candidates <path>` is required and must not exist either: the candidates file holds one line
per candidate a BKO arm generated or drew, with its holdout, fill, attempt, call, cell, record or
undecoded reason, and outcome; a direct arm's lines (below) carry a null call, and
`bko-direct-perturbed`'s carry the rewrite call that returned the row, null for a row that kept
its draw. `compare`
spends nothing. Inside `.venv`, writing to the run's folder under the gitignored `runs/`
([Runs](#runs)):

    mkdir -p runs/<run-id>
    test ! -e runs/<run-id>/calls.jsonl && test ! -e runs/<run-id>/holdout-candidates.jsonl && test ! -e runs/<run-id>/scores.csv && test ! -e runs/<run-id>/scores.log && uv run --no-sync python -m bko.evaluate score --log runs/<run-id>/calls.jsonl --candidates runs/<run-id>/holdout-candidates.jsonl [--data-dir <dir>] > runs/<run-id>/scores.csv 2> runs/<run-id>/scores.log
    mkdir -p runs/<run-id>-replay && test ! -e runs/<run-id>-replay/calls.jsonl && test ! -e runs/<run-id>-replay/holdout-candidates.jsonl && test ! -e runs/<run-id>-replay/scores.csv && test ! -e runs/<run-id>-replay/scores.log && uv run --no-sync python -m bko.evaluate score --replay runs/<run-id>/calls.jsonl --log runs/<run-id>-replay/calls.jsonl --candidates runs/<run-id>-replay/holdout-candidates.jsonl [--data-dir <dir>] > runs/<run-id>-replay/scores.csv 2> runs/<run-id>-replay/scores.log
    uv run --no-sync python -m bko.evaluate compare [<run>] runs/<run-id>/scores.csv [--candidate <arm>] [--beside <scores> [--beside-arm <arm>]]

`score` and `compare` take an optional run name before their other arguments, as in
`score <run> --log ...` and `compare <run> runs/<run-id>/scores.csv`; the default is `e1`. `launch`
takes it after the run id, and there it defaults to the run id's entry. A named run is a value of
`bko.evaluate.named_runs.RUNS`. A run's folder under `runs/` is named by any
`bko.datasets.run_folders.RUN_ID`: `<entry>-<n>` with an optional suffix (ADR
[`tooling/run-folders`](adr/tooling/run-folders.md)).

The run `g2` spends nothing: in place of the BKO arms it runs the direct arm `bko-direct`
(`bko.evaluate.direct`), which draws each request's rows from the fused BKB, seeded by the
request's seed (ADR [`research/bkb-direct-generation`](adr/research/bkb-direct-generation.md)). It
makes no call log, so it refuses `--log`, `--replay`, `--resume` and `--rehearse`, requires
`--candidates`, whose lines carry a null `call`, and is judged by no criterion:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/holdout-candidates.jsonl && test ! -e runs/<run-id>/scores.csv && test ! -e runs/<run-id>/scores.log && uv run --no-sync python -m bko.evaluate score g2 --candidates runs/<run-id>/holdout-candidates.jsonl [--data-dir <dir>] > runs/<run-id>/scores.csv 2> runs/<run-id>/scores.log </dev/null

Its stderr carries, from logger `bko.evaluate.direct`, a
`direct arm=%s holdout=%d fill=%s seed=%d outcome=%s %s` line per attempt (the attempt's counts
last) and the arm's `validity` sums at the end, as a BKO arm's; and from logger
`bko.generate.direct`, for every fill, a
`direct holdout=%d fill=%s key=%s rows=%d records=%d group_records=%d%s presence=%s` line per cell
(rows, the cell's records, its label group's records, the draw spec's other non-default fields as
` name=value` tokens, how presence is drawn) and a
`fallback holdout=%d fill=%s key=%s field=%s draws=%d%s presence=%s` line per cell and field read
from the label group, with the same tokens. Each names the holdout and the fill (`gap` or
`replica`) it serves; outside a harness fill they read `holdout=-1 fill=none`. `compare g2` writes a table of its header alone,
since `G2` judges no criterion, and logs every criterion's intervals as `unjudged` lines.

E9's runs, `e9-thin` and `e9-empty` and their slices `e9-thina` to `e9-thinj` and `e9-emptya` to
`e9-emptyj`, spend nothing either: beside `none`, `real` and `bko-direct` they run the logic-only
direct arm `bko-logic` as the candidate (ADR
[`research/logic-only-ablation`](adr/research/logic-only-ablation.md), which defines E9). Like
`g2`, they refuse `--log`, `--replay`, `--resume` and `--rehearse` and require `--candidates`; the
spend ceiling they keep from E6 is inert. `bko-logic` logs the lines `bko-direct` does, its
`direct` and `fallback` lines reading `presence=logic`, and its `direct arm=` line ends in
`fallback_draws=%d`, the attempt's draws that read a field from the label group. A component of
its concepts above `bko.generate.logic.MAX_COMPONENT` exits 2 once the releases load, before any
arm fits:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/holdout-candidates.jsonl && test ! -e runs/<run-id>/scores.csv && test ! -e runs/<run-id>/scores.log && uv run --no-sync python -m bko.evaluate score e9-thina --candidates runs/<run-id>/holdout-candidates.jsonl [--data-dir <dir>] > runs/<run-id>/scores.csv 2> runs/<run-id>/scores.log </dev/null

The no-mapping direct arm `bko-nomap` draws as `bko-direct` does with an empty mapping table (ADR
[`research/no-mapping-ablation`](adr/research/no-mapping-ablation.md)); the control runs
`e12-thin`, `e12-empty` and `e13` hold it. Its `direct` and `fallback` lines read
` mapped=false presence=shares`, its `arm` line's params add `"mapped": false`, and its
`direct arm=` line ends in `fallback_draws=%d`.

The donor-limit direct arm `bko-capped` draws as `bko-direct` does under a donor floor of 2 and a
reuse cap of 1 per unit (ADR [`research/sampler-donor-limits`](adr/research/sampler-donor-limits.md));
the runs `e14-thin` and `e14-empty` hold it. Its `direct` and `fallback` lines read
` min_donors=2 reuse_cap=1 presence=shares`, its `arm` line's params add `"min_donors": 2` and
`"reuse_cap": 1`, its `direct arm=` line ends in `fallback_draws=%d`, and the sampler writes one
`limits holdout=%d fill=%s key=%s floor_redirects=%d cap_skips=%d` line per cell: the picks whose
cell held fewer matching records than the floor, at least one, and the picks whose first pool
held a donor at the cap. A fill whose pools run out is refused and retried at a derived seed.

The microaggregation direct arm `bko-micro` draws as `bko-capped` does, but writes each numeric
field other than `age` as the mean of three distinct donors that share the unit's coded values
and the side of each of its cutoffs, rounded half-even to a whole number for an integer field and
to the frame's `fraction_digits` otherwise, never -0; a NaN in an averaged field is refused.
Coded fields come from the primary donor. Each other donor of a mean counts one third of a use
against the cap, and the primary of a unit holding a coded field counts a whole use, so a donor
already in any of that unit's means is never its primary (ADR
[`research/microaggregated-donors`](adr/research/microaggregated-donors.md)). On E6's frame it
averages the eight count fields. The runs `e20-thin` and `e20-empty` hold it. Its `direct` and
`fallback` lines read ` min_donors=2 reuse_cap=1 microaggregate=3 presence=shares`, its `arm`
line's params add `"min_donors": 2`, `"reuse_cap": 1` and `"microaggregate": 3`, its
`direct arm=` line ends in `fallback_draws=%d`, and it writes `bko-capped`'s `limits` line, a cap
skip counted at the pick's own weight. An averaged pick whose cell pool is under the floor of
three, or holds no class of three free donors, reads the label group, logged on the `fallback`
line, one line per averaged field; a fill whose label group holds no class either is refused,
naming the holdout and the fill, and retried at a derived seed. After the `limits` line, the
sampler writes one
`micro-field holdout=%d fill=%s key=%s field=%s floor=%d values=%d equal=%d primary=%d under=%d classless=%d sse=%s sst=%s`
line per cell and numeric field but `age`, written for a field no draw averaged too: the
effective floor `max(min_donors, microaggregate)`; the values written; those equal to at least
one of their own three donors' values; those equal to the primary's; the averaged picks that fell
back because the cell pool was under the floor (`under`) or held no class of three
(`classless`), which together make the `fallback` line's count for the field; and the donors'
within-group and total sums of squares, `1 - sse/sst` the share of the donors' sum of squares
the means keep.

The tree-share direct arm `bko-tree` draws as `bko-direct` does but for each tree field, a column
whose rows partition its codes, which it draws as one leaf of its ontology tree at its
ancestor-shrunk share and copies from a donor of that column alone (ADR
[`research/ancestor-shrunk-shares`](adr/research/ancestor-shrunk-shares.md)); the runs `e15-thin`
and `e15-empty` hold it on `diabetes-codes`. Its `direct` and `fallback` lines read
` tree=true presence=shares`, its `arm` line's params add `"tree": true`, and its `direct arm=`
line ends in `fallback_draws=%d`. The sampler writes one
`tree holdout=%d fill=%s key=%s column=%s leaves=%d positive=%d` line per cell and tree field: the
tree's leaves and those of positive share at the cell. Building a tree logs a warning
`tree absent=%d concepts=%s` naming each mapped concept its release lacks, which hangs from the
top. Every arm of a run that holds it writes `code-share-error` at the
scope `flagged-subgroups`: the mean over the tree fields of the total variation distance between
its added rows' value shares and the removed training rows', NaN for a fill with no rows.

The conditional `is_a` direct arms `bko-hier-0`, `bko-hier-w` and `bko-hier-closed` draw as
`bko-direct` does but for each `is_a` pair of drawn concepts, whose broad concept they draw given
its narrow one at the data's conditional, at pseudo-count 0, at `GROUP_WEIGHT`, or set to 1 (ADR
[`research/bkb-direct-generation`](adr/research/bkb-direct-generation.md), Decision 12); the
runs `e16-thin` and `e16-empty` hold them. Their `direct` and `fallback` lines read
` hierarchy=0.0 presence=shares`, ` hierarchy=20.0 presence=shares` and
` closed=true presence=shares`, and their `arm` lines' params add `"hierarchy": 0.0`,
`"hierarchy": 20.0` and `"closed": true`; like every direct arm but `bko-direct`, their
`direct arm=` lines end in `fallback_draws=%d`. The sampler writes one
`hier holdout=%d fill=%s key=%s pairs=%d clamped=%d` line per cell: the pairs drawn and how many
of their complements were clamped; where one was, the line ends in `clamped_concepts=%s`, the
clamped broad concepts in pair order, comma-separated. A frame with no `is_a` pair among its drawn concepts, a broad
concept with two narrow ones, or a nested pair is refused before any draw.

The coherent free-field direct arm `bko-coherent` draws as `bko-direct` does, then redraws the
free fields by sequential CART given `age`, `sex`, the label and the mapped fields, each value
copied from a training row in the new row's leaf (ADR
[`research/free-field-synthesis`](adr/research/free-field-synthesis.md)); the runs `e17-thin`
and `e17-empty` hold it. Its `direct` and `direct arm=` lines are `bko-direct`'s, and its `arm`
line's params add `"perturb": "coherent"`. Like `bko-direct-pram`, it logs
`perturb arm=%s fields=%s stream=2` at construction, `perturb coherent holdout=%d arm=%s rows=%d`
per fill with the rows it redrew, and the `perturb shapes`, `perturb columns` and
`perturb summary` lines the perturbation runs' paragraph below lists; each of its candidates
lines' `perturb` is `changed` or `unchanged`.

The pooled direct arms `bko-donor`, `bko-pool-random` and `bko-pool-all` draw `bko-nomap`'s
record with a set of fields drawn by the pooled rule: a unit's donor is a uniform cell record with
probability n/(n+`GROUP_WEIGHT`), else a uniform record of the whole label group (ADR
[`research/matched-knowledge-controls`](adr/research/matched-knowledge-controls.md)).
`bko-donor` pools each mapping group whole, `bko-pool-random` as many fields as the mapping's
groups hold, drawn at random once per fill, and `bko-pool-all` every feature but `age` and `sex`;
the run `e19-thin` and its slices hold all three. Their `direct` and `fallback` lines read
` presence=pooled`, ` pool=random presence=pooled` and ` pool=all presence=pooled`, their `arm`
lines' params add `"presence": "pooled"` and, for the last two, `"pool": "random"` or
`"pool": "all"`, and their `direct arm=` lines end in `fallback_draws=%d`, a pooled field read
from the label group's branch of the rule counting as fallen back. `bko-pool-random` writes one
`pool-random holdout=%d fill=%s fields=%s` line per fill attempt, before its `direct` lines,
naming the fields it drew at the attempt's seed, comma-joined in column order.

The knowledge-only direct arm `bko-kb` copies no training value: presence per logic component is
uniform over what the ontology allows, each mapping group's coded fields uniform over the codebook
tuples showing the drawn pattern, and `age` uniform inside the cell's band and the training
split's age range (ADR
[`research/matched-knowledge-controls`](adr/research/matched-knowledge-controls.md)). The runs
`e19-supported-thin` and `e19-supported-empty` hold it on `diabetes-supported` (ADR
[`research/dataset-variants`](adr/research/dataset-variants.md)); its `arm` line's params add
`"presence": "logic"` and `"donors": false`, its `direct` lines end in
`records=0 group_records=0 donors=false presence=logic`, it writes no `fallback` line, and its
`direct arm=` line ends in `fallback_draws=0`. A frame with a free field, or a mapping clause on a
numeric field or on `sex`, exits 2 before any arm fits.

The retained-data methods `row-resample`, `marginals` and `eb-marginals` fill a cell by the pooled
rule inside the cell's band: a whole row from one donor, each column from its own donor, or each
column at one empirical-Bayes weight `kappa`; `hb-marginals` draws each column from a multilevel
multinomial logit fitted per column (ADR
[`research/matched-knowledge-controls`](adr/research/matched-knowledge-controls.md)). `e19-thin`
holds all four, `e19-empty` `marginals` and `hb-marginals`, and the supported runs `hb-marginals`;
[Baselines](#baselines) lists their lines.

A run with neither a BKO arm nor the direct arm, such as `b1-uci`, `b1-synthea`, `b2-uci` and
`b2-synthea`, makes no call
and writes no candidates file, so it refuses `--log`, `--candidates`, `--replay`, `--resume` and
`--rehearse`:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/scores.csv && test ! -e runs/<run-id>/scores.log && uv run --no-sync python -m bko.evaluate score <run> [--data-dir <dir>] > runs/<run-id>/scores.csv 2> runs/<run-id>/scores.log </dev/null

`compare <run> runs/<run-id>/scores.csv --candidate <arm>` reads each new method in turn, and
`--beside` reads it beside another run's arm.

`merge` joins holdout slices' scores into the scores one run writes. It refuses files whose
holdouts overlap, whose union is not the named run's holdout seeds, whose headers differ, or whose
holdouts hold different (arm, source, measure, scope) keys, as slices scored under other code or
configuration would; an arm's added-row `dcr-p5` and `dcr-median` are set aside, since `score`
writes them only on a holdout where the arm added rows. It writes the rows in the named run's seed order under one header to stdout,
each holdout's rows in its file's order. It also reads each slice's sibling `scores.log` and
refuses a slice with none, or whose `data`, `versions` or `experiment` line differs from the first
slice's once `run=`, `holdouts=`, `seeds=` and `spend_ceiling=` are set aside, as slices at another
training fraction, removal fraction, cutoff or commit would. A key an older line lacks is read at
the value its runs used, as `bko.evaluate.run_folder.ASSUMED` lists for the `experiment` line and
`agreement=none` for the `data` line:

    test ! -e runs/<run-id>/scores.csv && uv run --no-sync python -m bko.evaluate merge e3 runs/<run-id-a>/scores.csv runs/<run-id-b>/scores.csv > runs/<run-id>/scores.csv

`compare e3 runs/<run-id>/scores.csv` then reads the merged file as one run's. A merge logs to
stderr, at INFO, first the `versions` line, then one line per input as it reads it,
`merge file=%s holdouts=%s rows=%d sha256=%s` (the path, the file's lowest and highest holdout as `<first>-<last>`, or `none` for a file with no
row, its row count and the sha256 of its bytes), then for the merged scores
`merge files=%d holdouts=%d rows=%d sha256=%s`, the sha256 of the bytes it writes.

`--beside` pairs the judged arm, or `--candidate`, with the same arm, or `--beside-arm`, in another
scores file over the holdouts both hold, at the run's confidence with no verdict; a holdout one file
holds is not counted. A holdout pairs only when its held-out rows match, its flagged cells match,
and no arm in either file copied a test record there, since the test split loses every record any
arm copied. A run that keeps copies (`e3` and its slices, whose experiment sets
`decontaminate` false) scores every arm on the whole test split and writes a `copies-kept`
held-out row on each holdout, so in its reading a copy excludes no holdout where both files hold
that row; beside a file without it, a copy in either file still excludes its holdout. Held-out
rows match when both files record equal held-out summary scores on the holdout (`copies-kept` and
the fingerprint aside), and the flagged cells match unless both files hold a
`flagged-fingerprint` held-out score there and the two differ; `bko.evaluate.beside.pairable` is
the one home of the exact rule. The other
shared holdouts are counted as excluded, split into copied and mismatched; a holdout whose flagged
cells differ counts as mismatched. Fewer than two paired holdouts, or an arm either file lacks, is
refused before the table is written, so a refusal leaves stdout empty. A series either arm lacks,
or that fewer than two paired holdouts define, is skipped with a `beside skip` line.

`shared` rescores a run's folder on the test split that no arm of it or of any `--with` folder
copies, so `--beside` can pair holdouts a copy would exclude (ADR
[`research/shared-test-split-reading`](adr/research/shared-test-split-reading.md), which holds its
identity checks and what self-reproduction admits). It makes no API call. It reads the folder's
`scores.log` and `holdout-candidates.jsonl`, rebuilds each BKO arm's rows from its last returned
attempt's `kept` candidates, regenerates each baseline, and checks each row set against its logged
`arm` line; each `--with` folder's rows leave the test split too, and every holdout is scored
through `run.score_holdout`.

With `--with`, each holdout writes a `shared-split` held-out row, the first 12 hex digits of the
sha256 of its kept test record ids, and logs
`shared holdout=%d records=%d copied=%d foreign=%d fingerprint=%s`; `--beside` pairs a copied
holdout when both files hold equal `shared-split` rows there. `--with` on a run that keeps copied
test records (`decontaminate=false`) is refused. With no `--with`, it rewrites the folder's
`scores.csv`. Its stdout is a `scores.csv` and its stderr a `scores.log`, which opens with one
`shared folder=<path> role=folder|with log_sha256=... candidates_sha256=...` line per input folder
and ends with `scores rows=%d sha256=%s matches_folder=true|false`, comparing the output's sha256
with the one the folder's own `scores` line logs.

A run is read in a pair with one compared run. `<run-id>` and `<run>` are the read run's folder
and named run, `<base-id>` and `<base-run>` the compared run's, and the four output folders are
named after the read run: `<run-id>-selfcheck` (no `--with`), `<run-id>-shared` (the run with the
compared run), `<run-id>-sharedbase` (the compared run with the run) and `<run-id>-sharedjudged`
(`compare`'s reading of the two):

    mkdir -p runs/<run-id>-selfcheck && test ! -e runs/<run-id>-selfcheck/scores.csv && test ! -e runs/<run-id>-selfcheck/scores.log && uv run --no-sync python -m bko.evaluate shared <run> --folder runs/<run-id> > runs/<run-id>-selfcheck/scores.csv 2> runs/<run-id>-selfcheck/scores.log </dev/null
    mkdir -p runs/<run-id>-shared && test ! -e runs/<run-id>-shared/scores.csv && test ! -e runs/<run-id>-shared/scores.log && uv run --no-sync python -m bko.evaluate shared <run> --folder runs/<run-id> --with runs/<base-id> > runs/<run-id>-shared/scores.csv 2> runs/<run-id>-shared/scores.log </dev/null
    mkdir -p runs/<run-id>-sharedbase && test ! -e runs/<run-id>-sharedbase/scores.csv && test ! -e runs/<run-id>-sharedbase/scores.log && uv run --no-sync python -m bko.evaluate shared <base-run> --folder runs/<base-id> --with runs/<run-id> > runs/<run-id>-sharedbase/scores.csv 2> runs/<run-id>-sharedbase/scores.log </dev/null
    mkdir -p runs/<run-id>-sharedjudged && test ! -e runs/<run-id>-sharedjudged/criteria.csv && test ! -e runs/<run-id>-sharedjudged/criteria.log && uv run --no-sync python -m bko.evaluate compare <run> runs/<run-id>-shared/scores.csv --candidate bko --beside runs/<run-id>-sharedbase/scores.csv --beside-arm bko > runs/<run-id>-sharedjudged/criteria.csv 2> runs/<run-id>-sharedjudged/criteria.log </dev/null

`shared` exits 2 on a missing input or an arm with no runner, and 3 when an identity check fails
or `--with` is refused.

**`rows`** rescores a recorded run's folder at the folder's own models and stores every fill, so
the readings that need a run's added rows read a file and regenerate nothing (ADR
[`research/rebuilt-rows-store`](adr/research/rebuilt-rows-store.md)). It makes no API call. The
output folder is `runs/<run-id>-rows`, `<run-id>` and `<run>` the folder and named run read. Its
`mkdir` has no `-p`, so a second run into the same folder fails there; that is the write-once guard:

    mkdir runs/<run-id>-rows && test ! -e runs/<run-id>-rows/scores.csv && env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m bko.evaluate rows <run> --folder runs/<run-id> --rows runs/<run-id>-rows/rows.jsonl > runs/<run-id>-rows/scores.csv 2> runs/<run-id>-rows/scores.log </dev/null

It runs `shared`'s identity checks (`rescore`) and, on each holdout, the strict check 7 (ADR
[`research/shared-test-split-reading`](adr/research/shared-test-split-reading.md)); a holdout that
fails a check exits 3 before its rows are written. The store is created when the first holdout
passes, so a run refused before then leaves none; a later refusal leaves the holdouts checked
before it, which a reader refuses as incomplete (ADR
[`research/rebuilt-rows-store`](adr/research/rebuilt-rows-store.md)). Exit codes are `shared`'s;
a `--rows` path that exists, or whose directory does not, exits 2 before any file is read. The
panel's shortfall WARNING lines are not reproduced, because a stored fill does not run the
generator again.

Its `scores.log` holds `shared`'s lines and these, at INFO unless named:

- `rows folder=%s run=%s holdouts=%d started_utc=%s pid=%d`, once, before the first holdout.
- `rows differing holdout=%d key=%s ours=%s theirs=%s`, per replica `quality` or
  `column-pair-trends` row that differs on a holdout training on more than 50,000 rows, which the
  strict check 7's exception admits; `key` is the row's key as the scores CSV writes it, `ours`
  the rescore's value and `theirs` the folder's.
- `rows holdout=%d arm=%s added=%d sha256=%s matches_arm=true`, per stored fill, once its
  holdout's lines are synced to disk.
- `rows holdout=%d stored=%d differing=%d subsampled=true|false`, per holdout: the fill lines
  stored, the `rows differing` lines logged, and whether the holdout trains on more than 50,000
  rows.
- `rows done holdouts=%d stored=%d sha256=%s`, last, once the store is closed: the holdouts and
  fill lines stored, and the store's sha256.
- `rows partial path=%s holdouts=%d`, at WARNING, when the run stops after the store exists: the
  holdouts stored before the stop.

`rows.jsonl` holds a header line, then one line per fill of every arm with an `arm` line, `real`
included, gap and replica; its format is in the ADR above. By hand, the fill of holdout 0's
`ctgan` arm hashes to its `arm` line's sha256:

    jq -j 'select(.kind=="rows" and .holdout==0 and .arm=="ctgan") | .csv' runs/<run-id>-rows/rows.jsonl | shasum -a 256

**`panel`** scores a recorded run's arms under the classifier panel from the fills its `rows`
store holds, with each model's calibration and no verdict; ADR
[`research/classifier-panel`](adr/research/classifier-panel.md) holds the models, their settings,
the calibration measures, how each arm's fills are answered and the thread limits. It makes no API
call and regenerates no rows. `rows` runs first, as its stage 1 and the self-check at the commit
that reads. `<run-id>` and `<run>` are the folder and named run read:

    mkdir runs/<run-id>-panel && test ! -e runs/<run-id>-panel/scores.csv && env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m bko.evaluate panel <run> --folder runs/<run-id> --rows runs/<run-id>-rows/rows.jsonl > runs/<run-id>-panel/scores.csv 2> runs/<run-id>-panel/scores.log </dev/null

It runs `rows`' identity checks and the strict check 7, with the keys ADR
[`research/shared-test-split-reading`](adr/research/shared-test-split-reading.md) lets `panel`
admit. Its scores CSV holds the folder's rows plus each panel model's, suffixed by its
`bko.evaluate.classifier.Model` value as `hgb` suffixes its own (`recall-rf`, `auc-pr-nb`), and
`brier` and `calibration-in-the-large` for every model and scope, suffixed the same way
(`brier-hgb`), with `auc-roc` for every model where the run's `Experiment.flagged_cell` is set.
Its `experiment` line logs `models=lr,hgb,rf,et,mlp,nb`. Exit codes are `rows`'. A
run succeeded when it exits 0 and its log's last line is `panel done`; its `scores` line reads
`matches_folder=false`, since its CSV holds rows the folder's lacks. A refused run leaves the
redirect's `scores.csv`, empty or partial, which the `test ! -e` guard then refuses: rerun into a
new folder.

Its `scores.log` holds `shared`'s lines, less `shortfall` lines since a stored fill runs no
generator, `rows`' `rows differing` lines, and these, at INFO:

- `panel rows=%s sha256=%s`, first: the store's path and sha256.
- `replay holdout=%d arm=%s attempt=%d`, per attempt before a `StoredBaseline` fill's logged one,
  which the run refused; a generator's refusal, and a BKO arm's stored one, log a `retry` WARNING
  instead.
- `fit holdout=%d arm=%s model=mlp n_iter=%d converged=true|false`, per perceptron fit of an arm's
  training set: its epochs, and `false` when it stopped at its epoch count.
- `shift fold holdout=%d model=mlp fold=%d converged=true|false`, per perceptron fold fit behind
  the arm `prior-shift`'s offsets, folds counted from 0.
- `panel holdout=%d differing=%d partial=%d subsampled=true|false`, per holdout once its strict
  check 7 passes: the `rows differing` lines logged, the fills whose `arm` line logs a shortfall,
  and whether the holdout trains on more than 50,000 rows.
- `panel done holdouts=%d`, last.

A paid run that stopped resumes with `--resume STOPPED`. It answers the stopped log's answered
calls from that log, sends the rest live, and writes one new call log at `--log` and one new
candidates file at `--candidates`, which holds every holdout's candidates, the answered calls'
included. The stopped run keeps its folder; the resumed run takes a fresh run id of the same entry,
the next free `<entry>-<n>` ([Runs](#runs)), so its supervised segments' `<run-id>-resume-<k>`
folders stay run ids. A log of a concurrent run (its `run` line holds `in_flight` above 1) resumes from
its first call not answered by exactly one reply: the later calls are dropped with one warning
each and sent live again. The new run's `carried_usd` holds their spend plus whatever the stopped
log itself carried, at any in-flight, so the ceiling reads the whole run. `launch --resume` runs
this command under the paid-run protocol below; by hand it reads:

    test -e runs/<last-segment>/calls.jsonl && mkdir -p runs/<new-run-id> && test ! -e runs/<new-run-id>/calls.jsonl && test ! -e runs/<new-run-id>/holdout-candidates.jsonl && test ! -e runs/<new-run-id>/scores.csv && test ! -e runs/<new-run-id>/scores.log && uv run --no-sync python -m bko.evaluate score <run> --resume runs/<last-segment>/calls.jsonl [--ceiling <usd>] --log runs/<new-run-id>/calls.jsonl --candidates runs/<new-run-id>/holdout-candidates.jsonl > runs/<new-run-id>/scores.csv 2> runs/<new-run-id>/scores.log

`<last-segment>` is the folder of the run's last segment: a supervised run that resumed its own
stops wrote `<run-id>-resume-<k>` folders, and the stop to resume is in the highest `k`'s log.
Resuming an earlier segment's log would send again the calls a later segment paid for, so
`launch --resume` refuses a log whose sha256 another run folder's `run` line names under
`resume`.

Never resume a log that another process still writes. A run stopped by the idle limit resumes. A
run stopped by the projection or the spend ceiling resumes only with `--ceiling <usd>`, a spend
ceiling for that resume alone above the stopped run's own projection, so the calls it paid for are
recovered under a fresh run id. The named run, its `experiment` line and the code stay as
launched, so `merge` joins the resumed slice to its siblings; no commit raises the ceiling, and
raising one E6 slice's raises no other's. The call log's `end` line names the stop rule that
ended a failed run under `stop` (`ceiling`, `projection` or `idle`, else null), and for a ceiling
or projection stop the holdouts it finished under `holdouts`. The projection is the `end` line's
`usd` (the whole run's spend, carried and replayed included) over those holdouts (at least 1),
scaled to the named run's holdouts, and never below `usd`; with no holdout finished, the spend of
the holdout in progress is scaled as one. A guard holds the policy: `score --resume` refuses, exit
3, a ceiling or projection stop with no `--ceiling` or one not above that projection, naming the
logged spend, its live part, the projection and the ceiling; a ceiling or projection stop with no
`holdouts`, with a `usd` or `live_usd` that is not a finite number of at least 0, or whose `run`
line names no run; `--ceiling` for any other stop; a log whose `run` line names another run; a
rehearsal's log; and a replay's log. `score` refuses `--ceiling` with no `--resume`, exit 2.
`launch --resume` refuses those, a log whose `run` line's settings differ from the named run's, and
a log a later segment already resumed, exit 2 before it claims a folder
(`bko.generate.resume_guard.refuse_unresumable`, `bko.evaluate.resumes`). A log with no `stop`
field predates it and resumes. The resumed run's ceiling and projection read the whole run's
spend, the stopped segments' included, as any resume's do, and its call log's `resume` field (in
the `run` and `end` lines) adds `stop` and `ceiling`, the rule it resumes and the ceiling it runs
under. A later resume of that run's log, a supervisor's included, runs under the same ceiling,
read from the log's `resume` field, and records it as `ceiling` alone; a raise stays above the
stopped run's projection, but the projection rule reads the resumed run's spend after
`projection_after` holdouts, carried spend included, so a resume that carries a concurrent
segment's dropped calls can stop again on the projection while it still replays, paying nothing.
The supervisor never resumes a ceiling or projection stop itself (F1 and F2 below); it writes a
`supervise stop=` WARNING with the stopped log's spend, its projection and the command, and the
operator, whose command is the sign-off, resumes the last segment's log under a fresh run id:

    uv run --no-sync python -m bko.evaluate launch <new-run-id> <run> --resume runs/<last-segment>/calls.jsonl --ceiling <usd above projected_usd>

For an E6 slice, `<run>` is the slice's own name (`e6-thinc`), `<new-run-id>` the next free
`e6-<n>`, and `<usd>` that slice's raised ceiling; the other 19 slices keep theirs.

`--rehearse` answers every call with canned records in the call's cell, inside every holdout's
training range, at e1-4's tokens per record priced at the named run's model's
rates, so the stop rules and spend lines run for free (a local model's rates are zero, so its
rehearsal checks wiring and folder roles alone); its spend is a wiring check, not an estimate; the
call log's `run` line holds `"rehearsal": true`, and a resume refuses it. `bko.evaluate.rehearse` holds the canned send and the
tokens per record. A replay of a rehearsal's log works, and the replay's own `run` line keeps
`"rehearsal": true`. A rehearsal
starts with `launch <run-id> [<run>] --rehearse`, which writes into the `rehearsal` folder inside
the run's folder, so the run's own folder stays free to launch. A rehearsal's `scores.csv` carries
no marker: its folder (`rehearsal/`), its call log and the WARNING line `score` writes to stderr
before any other run line
(`rehearsal canned replies: spend, projection and stop lines are wiring checks, not paid figures`)
do, and `compare` reads its scores like any scores file.

The `test ! -e` guards matter: the shell empties `scores.csv` and `scores.log` before `score` can
refuse an existing log or candidates file, and `scores.csv` is the only copy `compare` reads. A
judged run keeps `compare`'s stdout and stderr in its folder:

    test ! -e runs/<run-id>/criteria.csv && test ! -e runs/<run-id>/criteria.log && uv run --no-sync python -m bko.evaluate compare runs/<run-id>/scores.csv > runs/<run-id>/criteria.csv 2> runs/<run-id>/criteria.log

**The paid-run protocol.** A new paid run starts with `launch`, which a Runs row cites instead of
the `score --log` command above; a resume and a rehearsal start with it too:

    uv run --no-sync python -m bko.evaluate launch <run-id> [<run>] [--runs-dir <dir>] [--data-dir <dir>] [--in-flight N] [--retries R] [--dry-run]
    uv run --no-sync python -m bko.evaluate launch <new-run-id> [<run>] --resume runs/<last-segment>/calls.jsonl [--ceiling <usd>] [--runs-dir <dir>] [--data-dir <dir>] [--in-flight N]
    uv run --no-sync python -m bko.evaluate launch <run-id> [<run>] --rehearse [--runs-dir <dir>] [--data-dir <dir>]

`<run>` defaults to the run id's entry and must share it: a run registered as `g3-cell` launches
into `runs/g3-1/` as `launch g3-1 g3-cell`. `--runs-dir <dir>` sets the run folders' root, resolved
to an absolute path; it defaults to `runs/` beside the code, and an empty value is refused. A run
launched from a frozen worktree, with `PYTHONPATH=<worktree>/src`, passes the main checkout's
`runs/` as `--runs-dir`, because the worktree holds no `runs/`. It refuses a run id that is not a
run of record (`RUN_ID`) of `<run>`'s entry, a run with no BKO arm, a folder that already holds
`calls.jsonl`, `scores.csv`, `scores.log` or `holdout-candidates.jsonl`, and a machine that reports
battery power or, on macOS, whose power `pmset -g batt` cannot read (elsewhere it cannot check and
records `power=unchecked`). Otherwise it creates `scores.csv` and `scores.log` exclusively, so a
second launch into the folder refuses, appends the launch line (`run=`, `dry_run=`, `rehearse=`,
`resume=`), for a raised ceiling a `ceiling=%.4f stop=%s usd=%.4f live_usd=%.4f holdouts=%d
projected_usd=%.4f` line (the ceiling it accepted and the stopped log's spend and projection), the
power reading, the keep-awake and the command to `launch.log`, and starts `supervise <run-id> <run>
--retries R --runs-dir <runs>` in its own session, held awake by `caffeinate -i -s` on macOS, with
stdout and stderr in those two files; `--data-dir` is forwarded as an absolute path, and
`--in-flight` when given. `supervise` is internal to `launch`: it runs `score <run> --log
<folder>/calls.jsonl --candidates <folder>/holdout-candidates.jsonl` as its child, with the same
stdout and stderr (supervision, below). It appends the process id to `launch.log` as `pid=` and
prints the command that watches the folder (`tail -f <runs>/<run-id>/scores.log`, its path absolute;
for `--rehearse`, `tail -f <runs>/<run-id>/rehearsal/scores.log`). When the process cannot start it
appends `error=<type>: <message>` and then removes the two files; it also removes them when
`launch.log`'s first lines cannot be written, and a failed claim writes nothing, so a failed launch
leaves the folder free. A started process whose `pid=` line cannot be written is still reported
started, with a WARNING. `launch.log` is appended to, never truncated, so a failed launch's `error=`
line survives a retry. `--resume <calls.jsonl>` resumes a stopped run into a new folder with the
same power check and keep-awake: its first segment is `score <run> --resume <calls.jsonl> --log ...
--candidates ...` with the stopped log's path made absolute, and `--ceiling` forwarded to that
segment alone, and refuses a stopped log that is not a file or lies inside the new folder, and the
logs the resume guard above refuses. `launch` refuses, before it reads the power, a run id whose
first resume folder name, `<run-id>-resume-1`, is not a `RUN_ID` (such as `e1-4-resume`, whose
`e1-4-resume-resume-1` is not), unless `--retries 0` or `--rehearse` asks for no resume.
`--rehearse` supervises `score <run> --log ... --candidates ... --rehearse` with the same checks
inside `<run-id>/rehearsal/`; a folder that holds only `rehearsal/` is still free for the run.
`--resume` and `--rehearse` exclude each other. `--dry-run` writes `launch.log` and starts nothing,
so the folder stays free to launch. The power is read at launch and again before each resume, which
refuses on battery; between those reads a machine unplugged is not caught, and `caffeinate -s` holds
it awake on mains power only. The run itself stops, exit 2 with the call log ending `failed`, on any
of three rules, each logged as `stop rule=<name>` with the measure that fired: the spend ceiling
(`E1.spend_ceiling`, logged as `stop rule=ceiling` with the run's spend before the call it refuses),
the projection (the run's whole spend after `projection_after` holdouts, scaled to every holdout,
reaching the ceiling; a resume counts the holdouts it replays and the spend it carries and replays,
so it projects as the fresh run would; a rehearsal is projected, a replay is not, and `stop rules
projecting=` says which applies), and the idle limit (`idle_limit` seconds for one call to get a
reply, its retries included, timed by `asyncio.timeout` around the awaited send; on expiry no later
call is sent, and the calls beside it land under their own idle limits, so their replies and spend
are recorded). `bko.evaluate.watch` holds the projection and `bko.generate.flight` the ceiling and
the idle limit. `--in-flight N` sets the calls in flight, `bko.generate.flight.IN_FLIGHT` by default
(ADR [`research/concurrent-call-waves`](adr/research/concurrent-call-waves.md)); a stop lands at the
call it lands at with one call in flight. For a run with a call log, a `scores.log` with no
`run status=` line, which the call log writes as it closes, belongs to a process that is still
running or was killed. Copy the call log
off the runner before any replay.

**Supervision.** `launch` supervises its run (ADR
[`research/concurrent-call-waves`](adr/research/concurrent-call-waves.md)): `supervise` runs each
segment's `score` as a child process in its own process group, under its own `caffeinate -i -s` on
macOS so the machine stays awake while the segment runs even if the supervisor dies, and, when the
child exits, reads the call log's `end` line and the child's exit code. The line says under
`retryable` whether the API caused the stop, and under `rate_limited` whether any retry in the
segment saw a 429. Each stop branch is handled as follows:

| Branch | Signal | Supervisor |
|---|---|---|
| F1 projection | `stop` `projection` | no resume; a `supervise stop=` WARNING, then the operator's `launch --resume --ceiling` |
| F2 ceiling | `stop` `ceiling` | no resume; a `supervise stop=` WARNING, then the operator's `launch --resume --ceiling` |
| F3 idle | `stop` `idle`, `retryable` true | resume |
| F4 API failure | `status` `failed`, `retryable` true (a connection error, a timeout, a 408, 409 or 429, or a status of 500 or above) | resume |
| F5 deterministic | `status` `failed`, `retryable` false (`retryable` must be the JSON `true`) | no resume |
| F5 crash | no `end` line, the child exited on its own (exit code 0 or above) | no resume |
| F5 no log | no call log (`absent`), or one whose first line is not a call log's (`unreadable`, with a `supervise log=%s unreadable error=%s: %s` WARNING) | no resume |
| F6 killed or slept | no `end` line, a signal killed the child (a negative exit code: `SIGKILL`, an out-of-memory kill, a sleep) | resume |
| F7 operator | `status` `interrupted`, or the operator's signal or `STOP` file (below) | no resume |

Resume `k` (from 1) waits `min(60 x 2^(k-1), 900)` seconds, then checks the folder
`<run-id>-resume-<k>` with `launch`'s own argument, power and folder checks and the previous
segment's call log with the resume guard above (`launch --resume`'s), claims the folder only once
every check passed, and runs `score <run> --resume <the previous segment's calls.jsonl>` there,
with that folder's `scores.csv` and `scores.log`, at half the previous segment's in-flight (at
least 1) when that segment was rate limited, else at the same. `--retries R` caps the resumes, at
most 20; `--retries 0` runs one segment, and `launch --help` holds its default. A rehearsal is
never resumed, and neither is a run id whose `-resume-<k>` folder name is not a `RUN_ID`. The last
segment's call log holds every call, the replayed ones included, so it replays the whole run, and
a Runs row lists every segment's folder.

**Stopping a supervised run.** Three commands stop it, each with its own effect:

- `kill -TERM <pid>`, the `pid=` line of `<run-id>/launch.log` (`supervise pid=` repeats it):
  `caffeinate -i -s` runs the supervisor under the pid it was started with, so the signal reaches
  the supervisor itself. `SIGINT` and `SIGHUP` do the same. The supervisor forwards one `SIGINT` to
  the running segment's child, waits for it with no kill timeout, so the call log ends
  `interrupted`, and resumes nothing; during a backoff it claims nothing. It exits 130 when the
  child ends on the interrupt.
- `touch <runs>/<run-id>/STOP`: the running segment goes on to its own end, and the supervisor
  resumes nothing after it; a backoff in progress ends within 0.2 s. This covers `kill -9` of the
  child alone, which leaves no `end` line and reads as an out-of-memory kill, so it would otherwise
  be resumed.
- `kill -TERM <child_pid>`, a segment's `child_pid=` line: `score` turns `SIGTERM` and `SIGHUP`
  into an interrupt, as `SIGINT` is, so the call log ends `interrupted` (F7) and the supervisor
  resumes nothing.

**What the supervisor writes.** Every line goes to `<run-id>/launch.log` (for a rehearsal,
`<run-id>/rehearsal/launch.log`) behind a UTC time, `%Y-%m-%dT%H:%M:%SZ` and a space, and to the
supervisor's stderr, which is segment 0's `scores.log`. Each resume's own `score` writes to its
folder's `scores.log`, which the `next=` line's `watch=` names. The lines, in order:

- `supervise pid=%d`: the supervisor's process id, once.
- `supervise segment=%d child_pid=%d`: each segment's child, once it starts.
- `supervise stop=operator by=%s`: the operator's stop, a WARNING; `by` is `SIGTERM`, `SIGINT`,
  `SIGHUP` or `STOP` for the file.
- `supervise segment=%d folder=%s code=%d status=%s stop=%s retryable=%s rate_limited=%s reason=%s`:
  each segment's outcome. `folder` is the segment's absolute folder; `code` the child's exit code,
  negative for a signal's; `status` the `end` line's (`complete`, `failed`, `interrupted`), or
  `none` for no `end` line, `absent` for no call log, `unreadable` for a log whose first line is not
  a call log's; `stop` the `end` line's (`projection`, `ceiling`, `idle`) or `none`; `retryable`
  and `rate_limited` `true` or `false`, as the supervisor reads them. When a resume is scheduled the
  line ends `reason=retryable backoff_s=%d`.
- `supervise stop=%s usd=%.4f holdouts=%s projected_usd=%s resume=launch <fresh-run-id> <run>
  --resume <folder>/calls.jsonl --runs-dir <runs> [--data-dir <dir>] [--in-flight N] --ceiling
  <USD above projected_usd>`, one line: after a ceiling or projection stop's segment line, a
  WARNING with the `end` line's spend and holdouts, the projection over the named run's holdouts
  (`none` when the `end` line records no holdouts), and the command that resumes it under the
  run's runs root, its data root when set and the stopped segment's in-flight when set, each path
  quoted for a POSIX shell.
- `supervise next=%s in_flight=%s watch=tail -f <runs>/<next>/scores.log`: the resume folder, once
  claimed (no line names a resume that was not claimed); `in_flight` is a number or `default`.
- `supervise resume=%s refused=%s`: a resume a check refused, an ERROR, with the refusal's message.
- `supervise error=%s: %s`: a segment that could not start, an ERROR, with the error's type and
  message; a resume folder's claimed files are removed first.
- `supervise done segments=%d last=%s outcome=%s exit=%d`: the last line: the segments run, the
  last one's folder name, the reason and the exit code.

`reason` and `outcome` are one of `bko.evaluate.supervise.REASONS`: `complete`; `stop` (F1, F2);
`retryable` (a resume is scheduled); `exhausted` (the resumes ran out); `unresumable` (F5 and no
log); `crash` (F5 crash); `rehearsal`; `name` (the run id's resume folder name is not a `RUN_ID`);
`operator` (F7); and, on the `done` line alone, `refused` and `error`. A resume folder's own
`launch.log` gets `launched <time> run=<run> segment=<k> of=<run-id> resume=<previous calls.jsonl>`,
`power=mains` (or `power=unchecked` off macOS), `awake=caffeinate` (or `awake=none`) and
`command=<the segment's command>`. `supervise` exits with the last segment's exit code, 128 plus
the signal's number for a child a signal killed, or 2 when a resume is refused, a segment cannot
start or the supervisor cannot run.

- **What `score` does**: the named run's dataset (`bko.evaluate.named_run.NamedRun.dataset`)
  decides what it loads, as one `bko.evaluate.data.Data` every stage reads; for UCI it fetches UCI
  Heart Disease if the data root lacks it. Then for each of `E1`'s holdout seeds it splits the
  data with that seed and gives every arm the split's training rows and the request that fills
  each cell under the threshold up to the fill target (`Experiment.threshold` and
  `Experiment.fill_to`, both 20 in `E1`), or only the cell `Experiment.flagged_cell` names when
  it is set. Under an `Experiment.quota_rule` other than `fill-to`, each flagged cell is asked
  its rule's count less its training records instead (`bko.gaps.quota`; ADR
  [`research/share-based-quota`](adr/research/share-based-quota.md)), and a cell asked 0 leaves
  the request. Under `Experiment.synthetic_share` the request asks each cell for the training rows
  the share drops there (`bko.datasets.split.dropped`), every arm fits on the whole training
  split, the real rows a classifier fits on leave the dropped rows out, `real` adds them back, and
  every distance reads the whole split (ADR
  [`research/synthetic-share-ablation`](adr/research/synthetic-share-ablation.md)). A classifier
  fit on the real rows, weighted as the arm says, and the added rows is scored on the test split,
  then a gradient boosting model fit on the same rows on one OpenMP thread; its rows follow the
  arm's logistic regression rows as `recall-hgb`, `precision-hgb` and `auc-pr-hgb`, so
  `grep -v -- '-hgb,'` leaves the regression's rows alone. A run whose `Experiment.flagged_cell`
  is set also writes each model's ROC AUC, `auc-roc` and `auc-roc-hgb`, after its AUC-PR on each
  scope, empty where the scope holds one label; no other run writes it. The replica
  arms also fill the training split's own cell counts, or, when `Experiment.replica_rows` is set, that
  many rows split across the cells in proportion to those counts; those rows alone train a
  classifier and are compared with the training split for fidelity. A replica holding rows of fewer
  than two labels, or none, scores its classifier measures undefined, a replica of no rows its
  fidelity too, and the run goes on. It writes one row per holdout, arm, source, measure
  and scope to stdout as CSV; a measure a scope cannot define is empty. Each BKO arm reads the
  split's training rows' gaps under its definition (a CLLM arm reads none and shows training
  records instead, ADR [`research/cllm-arm-design`](adr/research/cllm-arm-design.md)), asks the API for each requested cell's
  shortfall, in the gap request and the replica request alike, validates every reply with
  `bko.validate`, drops what `bko.baselines.records.check` would refuse, and keeps the rest. An
  attempt that cannot fill the request inside `E1`'s budget is refused and retried like a
  baseline's; a curating arm instead gives each cell its own budget and returns a `partial` attempt
  (What it logs, below). Every arm fills before any is scored. A test record that some arm's rows
  copy in every column leaves the test split each arm's utility is scored on. The arm
  `prior-shift` adds no rows and scores `none`'s fit with each requested subgroup's test logits
  moved to its prevalence after the gap fill (ADR
  [`research/prior-shift-comparator`](adr/research/prior-shift-comparator.md)). With `--replay`,
  every BKO call is answered from that call log instead of the API. A call that differs from the
  log's, or a logged call left unplayed, stops the run. A replay reproduces the scores, and their
  sha256, only at the commit that scored the run: every scores file holds a `flagged-fingerprint`
  held-out row per holdout, and a run that keeps copies a `copies-kept` row per holdout, so a run
  scored at a commit without those rows does not reproduce its scores' sha256 when replayed at a
  later commit. Replay each run at its own commit (its `versions` line names it), or at a replay
  commit whose `git diff --stat <run commit> <replay commit> -- src` lists only the reply-text
  reader's files, `src/bko/generate/messages.py` and `src/bko/generate/local.py`, named by a pushed
  `replay/<name>` tag; `g6-2`'s to `g6-4`'s call logs replay at `replay/g6-local`, and the rerun's
  local runs, `g6-8` to `g6-12`, replay at their own commit, `run/rerun-1` (ADR
  [`research/call-log-replay`](adr/research/call-log-replay.md)). `b3-1`'s to `b3-3`'s call logs
  replay at their run commits only: their `cllm` calls asked at a budget of 9 with shortfall-sized
  asks, which later code does not ask; `b3-4`'s and `b3-5`'s replay at `run/rerun-1` (ADR
  [`research/cllm-arm-design`](adr/research/cllm-arm-design.md)).
  With `--resume`, the stopped
  log's answered calls come first, answered from it, then live calls. With `--rehearse`,
  `bko.evaluate.rehearse.CannedSend` answers every call, and the projection and the ceiling read
  its billed spend as live.
- **What the candidates file holds**: `bko.evaluate.holdout_candidates` is its format's home. A
  header line (`kind` `header`, `schema` 4, the run name, the `frame`'s name, the sha256 of the
  run's records under `data`, each release's sha256, and the generator settings, which a run with
  no paid arm omits), then one line per candidate (`kind` `candidate`): `holdout`, `arm`,
  `fill` (`gap` or `replica`), `attempt` within the fill from 1, `returned` (true for a returned or
  partial attempt, false for a refused or failed one), `call`, the `cell` asked, `position` in
  the call's reply, the decoded `candidate` or the `undecoded` reason, `outcome` (`kept`,
  `rejected`, `dropped-range`, `dropped-cell`, `dropped-concepts`, a prescribing arm's alone,
  `dropped-curation`, a curating arm's alone, `dropped-quota`, or `dropped-likelihood`, a ranking
  arm's alone) and the `failed` checks of a rejected one, and, on the lines of a scoring arm, the
  `score` its curator read: `confidence` and `aleatoric` on each `cllm` line that passed
  validation (`bko.evaluate.curation`), and `likelihood` on each `bko-curated` line it ranked, the
  record's BKB log-probability at its call cell's shares, null for minus infinity (ADR
  [`research/bkb-candidate-curation`](adr/research/bkb-candidate-curation.md), Arm), and, on the
  lines of a perturbation arm, the `perturb` result of its step: `perturbed`, `unchanged`, or
  `refused_<reason>` for `bko-direct-perturbed`, as the `perturb holdout=` line counts it, and
  `changed` or `unchanged` for `bko-direct-pram`. `read` reads schemas 1 to 4; only 3 and 4 hold
  `score`, and only 4 holds `perturb`. On schema 3 and schema 4 alike, it refuses a `score` whose
  names are not its arm's: `likelihood` for `bko-curated`, and `aleatoric` and `confidence` for
  `cllm`. It refuses a `perturb` on any other arm than the two perturbation arms, and a result
  outside its arm's set (`PERTURBS`). A replay writes the schema its commit writes, so byte
  equality holds against a run of the same schema. Lines are written when the harness drains a
  holdout: after every arm's gap fill, then after every replica fill. A stopped run's file holds
  what was drained before the stop; the stopped holdout's fills not yet drained are missing, even
  those that finished, and their candidates live only in `calls.jsonl`. Each arm, holdout and
  fill's lines count what its `candidates` or `replica-candidates` rows count;
  `bko.evaluate.holdout_candidates.read` reads the file.
- **What `compare` does**: it reads that CSV and writes one row per paired contrast as CSV: the
  criterion, the scope, the criterion's verdict (`met`, `not met` or `inconclusive`), the arm, the
  arm it is set against, the measure, and the contrast's mean, interval, the upper bound at the
  family-wise level that `not met` reads (`refuting_high`), and holdout counts. It refuses a scores
  file whose holdouts are not `E1`'s, and one where a series it judges misses a holdout or holds an
  empty value. `--candidate` judges another row-adding arm, a dry run whose verdict is not E1's. It
  also judges validity from the `candidates` and `replica-candidates` rows, the gap and replica
  fills' counts: the candidate's `passed` summed over `generated` summed over both fills, met at
  `E1.validity` or above. It logs each arm's mean recall and AUC-PR over the holdouts, both
  models', and each replica arm's mean replica recall, AUC-PR and fidelity, with no verdict. When
  the scores hold the boosting model's rows, it logs each primary and secondary contrast of its
  recall and AUC-PR, read as the criterion reads the regression's, with no verdict, after the
  table is written; a pair of arms either of which holds none of its rows is skipped, and the
  criterion's family still counts it. The table reads the regression only, so it is the same with
  or without them. A `panel` scores file holds each panel model's rows and the calibration
  measures too: `compare` logs a `mean` line per arm, measure and scope for each, and each panel
  model's contrasts as it logs the boosting model's, `model=` naming the model. A scores file is
  compared at the commit that scored it: a replica arm's validity needs its `replica-candidates`
  rows.
- **What it logs**: every `python -m bko.*` command's stderr line starts with its local time and
  UTC offset, `YYYY-MM-DD HH:MM:SS,mmm+HHMM` or `-HHMM` (two whitespace-free tokens), then its
  level and logger name, and Python warnings arrive as log lines; `bko.core.logs.configure` sets
  this up. `bko.evaluate.run_folder.LINE` reads such a line and a line stamped without the
  offset. For a Diabetes 130 run, `score` first logs the cohort line as it loads the records,
  `cohort encounters=%d repeat_encounters=%d excluded_discharges=%d unknown_gender=%d under_20=%d records=%d readmitted_under_30=%d`, which
  accounts for every encounter: `encounters` = `repeat_encounters` + `excluded_discharges` +
  `unknown_gender` + `under_20` + `records` ([Data](#data), The Diabetes frame). `score` logs a
  `data` line once it has loaded the named run's data,
  `data dataset=%s frame=%s records=%d digest=%s agreement=%s` (the dataset, its frame, its record
  count, the sha256 of its records (with its held test ids for a TableShift dataset,
  `bko.evaluate.data.Data.digest`) that the candidates file's header names under `data`, and the
  data use agreement the records are held under: `none` for open data, else the agreement's name
  in double quotes ([Data](#data), Agreements)), then an `experiment` line,
  `experiment run=%s frame=%s holdouts=%d seeds=%d-%d
  removal=%s training_fraction=%s decontaminate=%s arms=%s replica=%s threshold=%d fill_to=%d
  budget=%d validity=%s spend_ceiling=%.2f criteria=%s models=%s` (`models=` the downstream
  models the command scores, which `panel` alone extends past `lr,hgb`; `seeds=` the first and last
  holdout seed, `removal=` the removed cell as `<band>/<sex>/<label>@<fraction>`, the fraction in
  (0, 1] and 1 removing every row of the cell, or `none`, `training_fraction=` an exact fraction
  such as `1/4`, or `1`, `decontaminate=` `true` or `false`; ` flagged_cell=<band>/<sex>/<label>`
  follows `removal=` only when the run's `Experiment.flagged_cell` names the one cell its gap
  request asks for, flagged whatever its count and restored to `fill_to`;
  ` synthetic_share=<numerator>/<denominator>` follows `training_fraction=` only when the run's
  `Experiment.synthetic_share` replaces that share of each cell's training rows;
  ` replica_rows=<rows>` follows `replica=` only when the run's `Experiment.replica_rows` caps the
  replica fill near that many rows in all: each cell asks its `kept_count` at the share
  `replica_rows` over the training rows, so the rounded cells may sum to a few rows off it;
  `cutoff=<probability>` follows `fill_to=` only when the run's `Experiment.cutoff` is not E1's
  0.5, the probability at which utility calls a record positive; `quota_rule=<rule>` follows
  `fill_to=` and `cutoff=` only when the run's `Experiment.quota_rule` is not `fill-to`, the rule
  that asks each flagged cell for a count read from the cell (ADR
  [`research/share-based-quota`](adr/research/share-based-quota.md)); `spend_ceiling=` is the named
  run's own, a resume under `--ceiling` included), and a
  `versions` line (the checkout's commit and whether its tree is dirty, `unknown` without git,
  then Python, the platform, the evaluation stack, the SDK and the ontology parser), with `--resume`
  a `resume` line (`log=`, `sha256=`, `answered=`, `dropped=`, `carried_usd=`) before any call,
  and for a resumed ceiling or projection stop a
  `resume ceiling=%.2f inherited=true` line after it for a resume under the raised ceiling its
  stopped log ran under, or a
  `resume stop=%s usd=%.4f holdouts=%d projected_usd=%.4f ceiling=%.2f` line after it (the rule,
  the stopped log's spend, its finished holdouts, its projection and the ceiling the resume runs
  under), then
  per holdout, for a run with a training fraction below 1 (the `e5-*` runs) a
  `subsample holdout=%d fraction=%s train=%d kept=%d ratio=%.6f` line (the training rows before
  and kept, and the held-out ratio the holdout's intervals read),
  the split line `cells` prints, for a run with a removal (`e3` and its slices) a
  `removal holdout=%d cell=%s fraction=%s train=%d removed=%d kept=%d` line (the training rows
  before the removal, those it removed and those it kept), for a run with a synthetic share (the
  `e18-*` runs) a `share holdout=%d share=%s kept=%d dropped=%d dropped_sha256=%s` line (the
  share, the training rows kept and dropped, and the sha256 of the dropped record ids, sorted and
  joined by commas), for a run whose `Experiment.quota_rule` is not `fill-to` (the `e21-*` runs)
  one
  `quota-rule holdout=%d fill=gap key=%s rule=%s count=%d coverage=%d expected=%d asked=%d`
  line per flagged cell (`key` the cell as `<band>/<sex>/<label>`, as the direct arms' lines
  name it; the cell's training records, its coverage, its expected size and the rows its rule
  asks, a cell asked 0 leaving the request; a holdout scored through
  `bko.evaluate.run.score_holdout` writes them once, so `scores.log` and the logs of `shared`,
  `rows` and `panel`, which rescore each holdout, hold them, while `trace`, `privacy`,
  `contradictions <run>` and `shared`'s `--with` exclusion rebuild the request without them),
  one `quota` line per stratum, a
  `flagged holdout=%d cells=%s` line naming the flagged cells as `<band>/<sex>/<label>`,
  comma-separated, one `arm` line per fill
  (a replica fill is named `<arm>-replica`) with the attempt, the seed, the added rows' sha256 and
  the method's parameters (the reference arm `real` logs attempt 1, the gap request's seed and
  `params={}`), a `retry` warning for each refused attempt before the last, a
  `shortfall holdout=%d arm=%s attempt=%d cell=%s short=%d quota=%d` warning per short cell
  when a rejection sampler (CTGAN, TVAE, TabDDPM, the Gaussian copula, CART or the Bayesian network)
  still lacks rows of a gap fill at `MAX_DRAWN_ROWS` drawn on the last attempt, or SMOTE or ADASYN
  meets a cell of fewer than 2 training records, where it adds no row; on a gap fill, any arm's
  shortfall that falls on such cells alone fills partially on the attempt that meets it, with no
  retry, and one that also falls on another cell is retried (its `arm` line then names the cell's
  missing rows under `short` in the params, and the arm scores the rows it kept; a short replica
  fill still refuses,
  and the BKO arms' own `shortfall arm=` line below is a different line), and a
  `holdout` line with the progress and seconds, then the CSV's row count and sha256 and a
  `candidates path=%s lines=%d sha256=%s` line for the candidates file it wrote, read in chunks
  after the scores are rendered (a file it cannot read gets a `candidates path=%s unread error=%s:
  %s` WARNING, and the scores still go to stdout). It also logs
  per BKO attempt a `guided` line, `guided holdout=%d fill=%s arm=%s seed=%d outcome=%s calls=%s asked=%d
  generated=%d passed=%d dropped_range=%d dropped_cell=%d dropped_quota=%d kept=%d usd=%.4f`
  (`calls=` is the attempt's call range `<first>-<last>`, or `none` for an attempt refused before
  any call, and `usd=` the run's spend so far); for the prescribing arm `bko-skeleton` it reads
  `guided holdout=%d fill=%s arm=%s seed=%d outcome=%s calls=%s asked=%d generated=%d passed=%d
  dropped_range=%d dropped_cell=%d dropped_concepts=%d dropped_quota=%d kept=%d requeued=%d usd=%.4f`, where
  `requeued=` counts the attempt's lines returned to their cell's queue; for the curating arm `cllm`
  `dropped_curation=%d` stands before `dropped_quota=`, and each attempt first logs its curator's
  fit as `curator train=%d labels=%s stages=%d confidence_at_most=%s aleatoric_at_most=%s` (the
  training rows, their count per label as `<label>:<count>` pairs, the stages read and both
  thresholds). A ranking arm also logs, once its attempt ranks, one `ranked holdout=%d fill=%s
  arm=%s seed=%d cell=%s rows=%d kept=%d tied=%d minus_infinity=%d` line per cell it ranked, in
  the quota's order; a cell with no valid row is not ranked and logs no line. `tied=` counts the
  cell's rows whose score equals the last kept row's, when that score is finite and a dropped row
  shares it, else 0, and `minus_infinity=` counts the rows the sampler cannot draw, so a cut at
  minus infinity counts its rows there alone. Its `arm` line's params carry `over_ask` when it is
  above 1. A curating arm gives each cell its own budget, its budget times the cell's quota:
  once a cell's records asked could not cover its rows still short, the cell is given up with a
  `shortfall arm=%s seed=%d cell=%s short=%d quota=%d asked=%d limit=%d` line (the cell's
  quota, its records asked and its limit) and the other cells go on; an attempt that leaves a
  cell short returns the rows it kept, with outcome `partial`, instead of refusing, its params
  carry `short` (each such cell's missing rows), and its rows are checked against its quota less
  each given-up cell's missing rows. `bko-skeleton` also logs, after each wave,
  `prescribed holdout=%d fill=%s arm=%s seed=%d wave=%d asked=%d kept=%d dropped_concepts=%d
  requeued=%d`
  (the wave's number within the attempt from 1, and the wave's records asked, kept, dropped for
  their concepts and lines re-queued), so an arm starving on concept drops shows before its
  attempt ends, and, as the `guided` and `prescribed` lines do, a baseline method's `fit` and
  `filled` lines carry `holdout=%d fill=%s` (`holdout=-1 fill=none` outside a harness fill), and, from logger `bko.generate.direct`, the `direct holdout=` and `fallback` lines of
  each attempt's draw of its lines, as `g2` logs them. It logs a `spend` line before each BKO call
  with the run's spend so far and the ceiling, a
  `ceiling wait call=%d reserved_usd=%.4f in_flight=%d usd=%.4f ceiling=%.2f` line when a call waits
  for the calls in flight because their reservations reach the ceiling, a `halt call=%d unsent=%d`
  line when a wave sends no more calls after a failure, naming the first call not sent and the
  planned calls left, a `stop rule=idle call=%d limit_seconds=%.1f` WARNING naming the lowest call
  the idle rule stopped, `bko.generate`'s `log` line (with `in_flight=`) and per-call lines, as
  [Generation](#generation) states them (each landed call's `generate` line ends with its seconds
  from send to landing and the calls in flight when it landed),
  `bko.validate`'s line per batch it checks, `validated generated=%d passed=%d rejected=%d %s
  frame=%s` (the `%s` before `frame=` is each failed check's `<check>=<count>`, space-separated,
  or `none`), or for a batch with no candidates the WARNING `validated no candidates frame=%s, so
  validity is undefined`, a
  `retry call= attempt= status= retry_after=` line for each 429 or retryable status with an attempt
  left, the SDK's own retry records, which name their call as [Generation](#generation) states,
  the `run status=` line (`stop=`, `retryable=`, `rate_limited=` and `holdouts=` as the call log's
  `end` line holds them, then `run=`, `level=`, `in_flight=`) and the call log's totals line (with
  `live_usd=`, `resume=` and `rehearsal=`), a `test holdout=%d records=%d copied=%d kept=%s` line
  per holdout, `copied=` counting the test records some arm's added or replica row equals in every
  frame column, which leave the test split unless the run keeps copies (`e3` and its slices,
  `kept=true`), a `training holdout=%d arm=%s fill=%s rows=%d copies=%d` line per arm and fill
  after it, `copies=` counting the arm's added rows that equal some training row in every frame
  column, a `shift holdout=%d model=%s subgroup=%s records=%d target=%.6f offset=%.6f
  negatives=%d positives=%d folds=%d` line per model and requested subgroup of the arm
  `prior-shift`, `records=` its training records, `offset=` the logit offset its test records get,
  `negatives=` and `positives=` the target's counts after the fill, and `folds=` the out-of-fold
  fits behind the offset, with a `shift bound holdout=%d model=%s subgroup=%s offset=%.6f
  target=%.6f achieved=%.6f` WARNING when no offset reaches the target (a target of 0 or 1, or a
  moved mean more than 1e-6 from it), a `shift fold holdout=%d model=%s fold=%d
  converged=true|false` line per perceptron fold fit of that arm, folds counted from 0, an
  `undefined holdout=%d arm=%s rows=%d labels=%d held=%s` WARNING for each replica of
  fewer than two labels (`held=` the labels it holds, or `none`), whose classifier measures score
  undefined, as its fidelity does for a replica of fewer than two rows, then after the last
  holdout an `undefined replicas arm=%s holdouts=%d of=%d` WARNING per arm with such a replica,
  and at the end of the run, a stopped run included, one `validity` line per Messages arm
  (a direct arm's and the perturbed arm's end before `usd=`),
  `validity arm=%s generated=%d passed=%d dropped_range=%d dropped_cell=%d dropped_quota=%d
  usd=%.4f`, with `dropped_concepts=%d` before `dropped_quota=` for a prescribing arm and
  `dropped_curation=%d` there for a curating arm, and `dropped_likelihood=%d` after
  `dropped_quota=` for a ranking arm: its counts
  over both fills and its own spend in USD, the sum of each of its attempts' change in the call
  log's spend. A concept-drop rate reads `dropped_concepts` over `passed` from that line, since
  only a candidate that passed validation can be dropped for its concepts. For `cllm`, the rows
  that reach its curator are `passed` less `dropped_range` and `dropped_cell`, and those its
  curator keeps are those less `dropped_curation`. Its keep share in one fill and cell asked reads
  its candidates-file lines of that `fill` and `cell`: the `kept` and `dropped-quota` lines over
  those lines and the `dropped-curation` lines together. A resume logs a WARNING
  for a cut final line it drops from the stopped log and one for each call it drops: the unanswered
  last call, or for a concurrent log every call from the first unanswered one. It logs a `resume
  live from call` line before its first live call. A fill that stops the run logs a `refused` or
  `contract` error naming the holdout and arm. `compare` logs the `versions` line, then a
  `compare candidate=%s dry_run=%s scores_sha256=%s run=%s level=%s training_fraction=%s` line
  with the candidate, whether it is a dry run, the scores file's sha256, the run's name, its
  guidance level (`none` for a run with no paid arm) and its training fraction,
  then one `criterion` line per criterion with its verdict, a `criterion=validity` line, a
  `validity` line per other BKO arm, a `candidates` line per BKO arm and fill (`fill=gap`,
  `fill=replica`) with its counts and rate, a `copies` line per arm, a `mean` line per arm, measure
  and scope, a `contrast model=hgb` line per contrast of the boosting model's measures when the
  scores hold them (`criterion=`, `scope=`, `arm=`, `against=`, `measure=`, `mean=`, `low=`,
  `high=`, `refuting_high=`, `holdouts=`, `differing=`) or, once per pair of arms either of which
  lacks them, a `contrast skip model=hgb` line (`arm=`, `against=`, `reason=absent`), then the
  same for each `panel` model whose rows the scores hold (`model=rf`, `et`, `mlp`, `nb`), for a
  run that judges no criterion (`g2`) an `unjudged` line per interval of every criterion
  `CRITERION_NAMES` holds, read with no verdict (the same keys as a `contrast` line, no `model=`;
  its table is then the header alone) or, for a contrast some holdout leaves undefined (as when
  decontamination leaves a holdout no positive test record in the flagged cells), an
  `unjudged skip` line (`criterion=`, `scope=`, `arm=`, `against=`, `measure=`,
  `reason=undefined`, `defined=`, `holdouts=`), then, only when the scores hold a `panel` model's
  rows, each other model's whose rows they hold, the boosting model first, as
  `unjudged model=<model>` lines of its separation criteria, `unjudged model=<model> skip` lines
  for an undefined contrast, and one `unjudged model=<model> skip` line (`arm=`, `against=`,
  `reason=absent`) per pair of arms either of which lacks the model's rows (privacy reads no
  model, so only the regression's lines hold it), and a `replica` line per replica arm, measure
  and scope. With `--beside`, it
  then logs a `beside flagged holdout=%d ours=%d theirs=%d` line for each shared holdout no copy
  excludes whose two `flagged-fingerprint` scores differ (each read as an integer), then a `beside` counts line (`scores_sha256=`,
  `copies_excluded=`, `shared=`, `paired=`, `excluded=`, `copied=`, `mismatched=`, `arm=`,
  `against=`, `run=`, `level=`; `copies_excluded=false` when a copy can exclude no shared holdout,
  since the run keeps copies and both files hold `copies-kept` on each), then per series a
  `beside` line (`arm=`,
  `against=`, `run=`, `level=`, `source=`, `measure=`, `scope=`, `mean=`, `low=`, `high=`,
  `holdouts=`, `differing=`, `undefined=`) or a `beside skip` line (`arm=`, `against=`, `run=`,
  `level=`, `source=`, `measure=`, `scope=`, `reason=absent` or `reason=undefined`, `defined=`); a
  refused reading logs its `beside flagged` lines and the counts line alone. An experiment copies them into its Runs row in
  [EVIDENCE.md](EVIDENCE.md#how-to-read-an-entry).
- **Where the configuration lives**: `bko.evaluate.named_runs.RUNS`. It holds `E1` as `e1`, `G2`
  as `g2` ([EVIDENCE.md](EVIDENCE.md#g2-direct-generation-from-the-fused-bkb)), each
  guidance-level run as `g3-<level>` ([EVIDENCE.md](EVIDENCE.md#g3-guidance-level)), each
  effort-sweep run as `g4-low` and `g4-medium` ([EVIDENCE.md](EVIDENCE.md#g4-effort-at-summary)),
  and the skeleton run as `g5`, whose `bko-skeleton` arm prescribes each record's concepts beside
  an in-run `bko` ([EVIDENCE.md](EVIDENCE.md#g5-bkb-drawn-concepts-llm-filled-fields)). It holds
  E3 over the Synthea frame as `e3`, and its two holdout slices as `e3-a` (holdouts 0 to 49) and
  `e3-b` (50 to 99), each with half of `e3`'s spend ceiling (ADR
  [`research/synthea-downstream-design`](adr/research/synthea-downstream-design.md);
  [EVIDENCE.md](EVIDENCE.md#e3-synthea-downstream-augmentation)). It holds B1's runs, `b1-uci`
  (E1's methods beside the Gaussian copula and sequential CART) and `b1-synthea` (E3's methods,
  both and `real` under E3's removal), each with no BKO arm and no criterion
  ([EVIDENCE.md](EVIDENCE.md#b1-classical-baselines)), and B3's, `b3-uci` (E1's methods, `bko`
  and the CLLM arms `cllm` and `cllm-uncurated`) and `b3-synthea` (E3's methods, those three and
  `real` under E3's removal), each with `cllm` as the candidate and no criterion (ADR
  [`research/cllm-arm-design`](adr/research/cllm-arm-design.md);
  [EVIDENCE.md](EVIDENCE.md#b3-cllm-arms)), and B2's, `b2-uci` (E1's methods and the Bayesian
  network `bayes-net` as the candidate) and `b2-synthea` (E3's methods, `bayes-net` and `real`
  under E3's removal), each with no BKO arm and no criterion (ADR
  [`research/bayesian-network-baseline`](adr/research/bayesian-network-baseline.md);
  [EVIDENCE.md](EVIDENCE.md#b2-bayesian-network-baseline)). It holds E4's dose runs, `e4-thirty`, `e4-forty` and
  `e4-eighty` (`E4_TARGETS`: E1's methods and `bko` filling each flagged cell to 30, 40 or 80 at
  E1's threshold, a $10.00 ceiling and no criterion;
  [EVIDENCE.md](EVIDENCE.md#e4-fill-above-the-threshold)), and E5's scarcity runs, `e5-quarter`,
  `e5-half` and `e5-threequarters` (`E5_FRACTIONS`: the same methods and ceiling, each holdout
  keeping a quarter, half or three quarters of each cell's training rows;
  [EVIDENCE.md](EVIDENCE.md#e5-augmentation-against-training-set-size)). It holds G6's generator
  runs, `g6-sonnet`, `g6-gemma`, `g6-nemotron` and `g6-olmo` (`G6_MODELS`: `G6_BASE`, E1's methods
  and `bko` with no criterion, at each run's model and ceiling), and its thinking-cost runs,
  `g6-gemmathink` and `g6-nemotronthink` (`G6_THINKING`: `G6_BASE` at each `-thinking` model id on
  holdouts 0 to 4, a $1.00 ceiling and an idle limit of 1700 s, a liveness guard below
  `bko.generate.local.TIMEOUT_SECONDS`, not a measurement; ADR
  [`research/local-generator-backend`](adr/research/local-generator-backend.md);
  [EVIDENCE.md](EVIDENCE.md#g6-generator-model)). It holds E6's runs over the Diabetes frame,
  `e6-thin` (`E6_THIN`: every arm but the curated `cllm`, 0.985 of `(6-7, male, 1)`'s training rows removed) and
  `e6-empty` (`E6_EMPTY`: all of them removed, no replica), each at `E6_SETTINGS`, and each
  run's ten holdout slices, `e6-thina` to `e6-thinj` and `e6-emptya` to `e6-emptyj`, each the
  next 10 holdouts at a tenth of its run's ceiling (ADR
  [`research/diabetes-subgroup-removal-design`](adr/research/diabetes-subgroup-removal-design.md);
  [EVIDENCE.md](EVIDENCE.md#e6-diabetes-130-subgroup-removal)). It also holds G7's curated run,
  `e6-curated` (`E6_CURATED` at `E6_SETTINGS`, unsliced, a $100 ceiling; ADR
  [`research/bkb-candidate-curation`](adr/research/bkb-candidate-curation.md);
  [EVIDENCE.md](EVIDENCE.md#g7-bkb-likelihood-curation)), and E9's logic-only runs, `e9-thin` and
  `e9-empty` (`E9_THIN` and `E9_EMPTY`: `E6_THIN` and `E6_EMPTY` with `none`, `real`, `bko-direct`
  and the candidate `bko-logic`, no replica) and their slices `e9-thina` to `e9-thinj` and
  `e9-emptya` to `e9-emptyj`, sliced as E6's are, the ceiling inert (ADR
  [`research/logic-only-ablation`](adr/research/logic-only-ablation.md)). It holds the perturbation
  pilot's runs, `e10-pilot-thin` and `e10-pilot-empty` (`E10_PILOT_THIN` and `E10_PILOT_EMPTY`:
  `E9_THIN` and `E9_EMPTY` with `none`, `real`, `bko-direct`, the candidate `bko-direct-perturbed`
  and `bko-direct-pram`, holdouts 0 and 1, a $5.00 ceiling; ADR
  [`research/identifier-perturbation`](adr/research/identifier-perturbation.md)), and its
  PRAM-only full runs, `e10-pram-thin` and `e10-pram-empty` (`E10_PRAM_THIN` and
  `E10_PRAM_EMPTY`: `E9_THIN` and `E9_EMPTY` with `none`, `real`, `bko-direct` and the candidate
  `bko-direct-pram`, the ceiling inert) and their slices `e10-pram-thina` to `e10-pram-thinj` and
  `e10-pram-emptya` to `e10-pram-emptyj`, sliced as E6's are. It holds the control runs,
  `e12-thin` and `e12-empty` (`bko.evaluate.runs_direct.E12_THIN` and `E12_EMPTY`: `E6_THIN` and
  `E6_EMPTY` with `CONTROL_ARMS`, the comparison set, `real`, `bayes-net`, `bko-direct`,
  `bko-nomap`, `label-bootstrap` and `prior-shift`, the candidate `bko-direct` and no replica) and
  their slices `e12-thina` to `e12-thinj` and `e12-emptya` to `e12-emptyj`, sliced as E6's are,
  and `e13` (`E13`: `E3` with those arms and `bko-logic`, the candidate `bko-direct`, no replica and
  a $1.00 ceiling that no arm's call reaches). `docs/EVIDENCE.md`, E12 and E13, describes them.
  `RUNS` holds the donor-limit runs, `e14-thin` and `e14-empty`
  (`bko.evaluate.runs_direct.E14_THIN` and `E14_EMPTY`: `E9_THIN` and `E9_EMPTY` with `none`,
  `real`, `bko-direct` and the candidate `bko-capped`, no replica, the ceiling inert) and their
  slices `e14-thina` to `e14-thinj` and `e14-emptya` to `e14-emptyj`, sliced as E6's are;
  `docs/EVIDENCE.md`, E14, describes them. `RUNS` holds the tree-share runs, `e15-thin` and
  `e15-empty` on `diabetes-codes` (`bko.evaluate.runs_direct.E15_THIN` and `E15_EMPTY`: `E9_THIN`
  and `E9_EMPTY` with `none`, `real`, `bko-direct`, `bko-nomap` and the candidate `bko-tree`, no
  replica, the ceiling inert) and their slices `e15-thina` to `e15-thinj` and `e15-emptya` to
  `e15-emptyj`, sliced as E6's are; `docs/EVIDENCE.md`, E15, describes them. `RUNS` holds the
  conditional `is_a` runs, `e16-thin` and `e16-empty` (`bko.evaluate.runs_direct.E16_THIN` and
  `E16_EMPTY`: `E9_THIN` and `E9_EMPTY` with `none`, `real`, `bko-direct`, `bko-hier-0`,
  `bko-hier-closed` and the candidate `bko-hier-w`, no replica, the ceiling inert) and their
  slices `e16-thina` to `e16-thinj` and `e16-emptya` to `e16-emptyj`, sliced as E6's are;
  `docs/EVIDENCE.md`, E16, describes them. `RUNS` holds the coherent free-field runs, `e17-thin`
  and `e17-empty` (`bko.evaluate.runs_direct.E17_THIN` and `E17_EMPTY`: `E9_THIN` and
  `E9_EMPTY` with `none`, `real`, `bko-direct` and the candidate `bko-coherent`, no replica, the
  ceiling inert) and their slices `e17-thina` to `e17-thinj` and `e17-emptya` to `e17-emptyj`,
  sliced as E6's are; `docs/EVIDENCE.md`, E17, describes them. `RUNS` holds P3's runs,
  `p3-fifteen`, `p3-thirty`, `p3-fifty`, `p3-eighty` and `p3-ninetyfive`
  (`bko.evaluate.runs_published.P3`: `E6_THIN` with `none`, `real`, `smote`, `ctgan`, `cart`,
  `bko-nomap`, `label-bootstrap`, `prior-shift` and the candidate `bko-direct`, removing 0.15,
  0.30, 0.50, 0.80 and 0.95 of `E6_CELL`'s training rows, that cell flagged by name and filled to
  `juwara_total`, no replica, the ceiling inert) and their slices `p3-fifteena` to
  `p3-ninetyfivej`, sliced as E6's are; `docs/EVIDENCE.md`, P3, describes them (ADR
  [`research/covariate-bias-comparison`](adr/research/covariate-bias-comparison.md)). `RUNS`
  holds E18's runs, `e18-two`, `e18-ten`, `e18-quarter`, `e18-half` and `e18-all`
  (`bko.evaluate.runs_direct.E18`: `E6_THIN` with `real`, `ctgan`, `tvae`, `cart`, `bayes-net`,
  `tabddpm`, `label-bootstrap` and the candidate `bko-direct`, no removal, 1/50, 1/10, 1/4, 1/2
  and 1 of each cell's training rows replaced by each arm's fill, `E6_CELL` flagged by name, no
  replica, the ceiling inert) and their slices `e18-twoa` to `e18-allj`, sliced as E6's are;
  `docs/EVIDENCE.md`, E18, describes them (ADR
  [`research/synthetic-share-ablation`](adr/research/synthetic-share-ablation.md)). `RUNS` holds
  the matched knowledge controls, `e19-thin` and `e19-empty` on `diabetes` and
  `e19-supported-thin` and `e19-supported-empty` on `diabetes-supported`
  (`bko.evaluate.runs_direct.E19_THIN`, `E19_EMPTY`, `E19_SUPPORTED_THIN` and
  `E19_SUPPORTED_EMPTY`: E12's runs with the arms ADR
  [`research/matched-knowledge-controls`](adr/research/matched-knowledge-controls.md) lists per
  family, the candidate `bko-direct`, no replica, the ceiling inert) and their slices
  `e19-thina` to `e19-supported-emptyj`, sliced as E6's are; `docs/EVIDENCE.md`, E19, describes
  them. `RUNS` holds the microaggregation runs, `e20-thin` and `e20-empty`
  (`bko.evaluate.runs_direct.E20_THIN` and `E20_EMPTY`: `E9_THIN` and `E9_EMPTY` with `none`,
  `real`, `bko-direct`, `bko-capped` and the candidate `bko-micro`, comparators `bko-direct`,
  `bko-capped` and `none`, no replica, the ceiling inert, `REFERENCE` `bko-capped`) and their
  slices `e20-thina` to `e20-thinj` and `e20-emptya` to `e20-emptyj`, sliced as E6's are;
  `docs/EVIDENCE.md`, E20, describes them. `RUNS` holds the quota sweep, `e21-<word>-thin` and
  `e21-<word>-empty` for each word `coverage`, `coverage-two`, `coverage-four`,
  `coverage-eight`, `expected-size` and `recommended`, naming the rules `coverage`,
  `coverage-x2`, `coverage-x4`, `coverage-x8`, `expected-size` and `recommended`
  (`bko.evaluate.runs_direct.E21` and `E21_NAMES`: `E12_THIN` and `E12_EMPTY` with `quota_rule`
  set to the rule, the ceiling inert, `REFERENCE` `real`) and their slices `e21-<word>-thina`
  to `e21-<word>-emptyj`, sliced as E6's are; `docs/EVIDENCE.md`, E21,
  describes them. `RUNS` also holds P1's runs,
  `p1-id` on `tableshift-id` and `p1-ood` on `tableshift-ood` (`bko.evaluate.runs_published.P1`: E1
  on ten holdouts with the comparison set, the classical synthesizers, the Bayesian network,
  TabDDPM, the label bootstrap and the candidate `bko-direct`, a replica for every row-adding arm
  and the candidate, no criterion and test copies kept, every holdout on TableShift's fixed test
  split; ADR [`research/tableshift-comparison`](adr/research/tableshift-comparison.md)). It holds
  P2's run, `p2` (`bko.evaluate.runs_published.P2`: E1 on UCI Heart Disease with P1's arms, the
  candidate `bko-direct`, a replica for every row-adding arm and the candidate and no criterion),
  whose replica fills `syntheval` scores (ADR
  [`research/syntheval-reading`](adr/research/syntheval-reading.md)). No flag changes a value;
  `--log`, `--candidates`, `--replay`, `--resume` and `--rehearse` choose where calls and candidates
  are recorded and where calls are answered from, and `--in-flight` how many calls overlap, never a
  value a call sends. The rate budget is `bko.generate.flight.IN_FLIGHT` per run.
- **Exit codes**: as in [Data](#data), with 2 also for a request with nothing to fill, a fit that
  does not converge, a method that refuses a request on every attempt (but a generator short at
  its row limit, which fills partially) or scores that cannot be
  judged, and 3 for a method's output or a scores file that breaks its contract; 2 also for an
  existing call log or candidates file, data held under an agreement that grants no permission to
  send its records ([Data](#data), Agreements), BKO arms with no `--log` or `--candidates`, a
  direct arm with no `--candidates`, a direct arm that reads the mapping table
  and whose field rows nest in a chain (`bko.gaps.shares.nested`), a logic-only arm with a component above
  `bko.generate.logic.MAX_COMPONENT` concepts, `--log`, `--replay`, `--resume` or `--rehearse`
  with no BKO arm to build, `--candidates` with no BKO or direct arm, one path for
  both files, more than one of `--replay`, `--resume` and `--rehearse`, a call that gets no
  response, a failed gap query, a run whose spend reached `E1.spend_ceiling` and `candidates` or
  `replica-candidates` rows whose validity cannot be read; 3 also for a replay that does not match
  its log, and a stopped log that breaks the replay contract: a replay's log, a rehearsal's log, or
  a resume that stopped while it replayed. `compare`
  writes its table before it reads validity, so it keeps the table when it exits 2 on validity.
- **Reproducibility**: a holdout seed gives the same split and requests, and the baselines keep
  the promise of [Baselines](#baselines). A BKO arm's rows do not repeat, since the API takes no
  seed; the paid run's call log is their record (ADR
  [`research/call-log-replay`](adr/research/call-log-replay.md)), and `--replay` rewrites the
  scores and the candidates file byte for byte from it. A resumed run's own log is its record: its
  paid spend is `live_usd`, while `usd` counts the whole run and is what the ceiling reads.

**Likelihood.** `likelihood` reads a run's candidates file and asks whether a candidate's
log-probability under the BKB's concept draw (`bko.gaps.likelihood.Likelihood`) separates fidelity,
the question G7's curated arm rests on
([EVIDENCE.md](EVIDENCE.md#g7-bkb-likelihood-curation)). `--candidates` repeats: a sliced run's
files are read as one run's, and the command refuses files whose holdouts overlap or leave a holdout
of the named run out, whose headers differ in any key but `run`, or whose `run` is not a registered
slice of the named run (its configuration but for its holdouts and spend ceiling, over holdouts
among the named run's). It makes no API call:

    test ! -e runs/<run-id>/likelihood.csv && mkdir -p runs/<run-id> && uv run --no-sync python -m bko.evaluate likelihood [<run>] --candidates runs/<source-run-id>/holdout-candidates.jsonl [--candidates <more>]... [--data-dir <dir>] > runs/<run-id>/likelihood.csv 2> runs/<run-id>/likelihood.log </dev/null

Per holdout, it scores the `bko` arm's replica records that were kept or dropped by the quota,
splits each cell's records at that cell's median score (ties in `sha256` order of holdout, call and
position), and scores the pooled top halves and the pooled bottom halves against the training rows.
The CSV holds one row per holdout and cell: the score's spread, its median, the share of records
at the minimum and at the median, and the two halves' sizes (equal; an odd cell's middle record sits
in neither). The log holds the `experiment` and `versions` lines, a `candidates` line with the
file's path, schema and sha256, a `likelihood pool` line counting the file's lines and the scored
ones, each holdout's `split` line, one `likelihood measure=` line per fidelity measure with the
top-minus-bottom corrected interval over the holdouts at the largest held-out ratio, one `likelihood concept=` line per concept with its share in the top half, the
bottom half and the training rows, and `gate separated=`, true when every measure's interval lies
above 0. A sliced run's files log a `candidates` line each and one `likelihood pool` line over
them all. A single file's `run` must be the named run, while a sliced run's files carry their
slices' runs; a schema-2, schema-3 or schema-4 file's data digest must match the
loaded data, and a schema-1 file's `uci` digest must match UCI Heart Disease's, for a run that
scores it.

**`privacy`** reads a run's stored fills for membership inference, a field-group match attack,
donor reuse and NNDR beside distance to closest record
(ADR [`research/membership-inference-reading`](adr/research/membership-inference-reading.md)).
`<run>` is the named run whose slices' `rows` stores are read (`e6-thin`); `rows` runs first, one
store per slice. `--rows` repeats, once per store, and the command refuses stores whose holdouts
overlap or leave a holdout of the named run out. In zsh, `$(printf -- '--rows %s ' <paths>)`
expands a list of store paths, or a brace range of them, into the repeated flags. It makes no API
call:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/privacy.csv && env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m bko.evaluate privacy <run> [--runs-dir DIR] $(printf -- '--rows %s ' runs/<store-id>-rows/rows.jsonl <more>...) > runs/<run-id>/privacy.csv 2> runs/<run-id>/privacy.log </dev/null

Per holdout and arm, it reads the attack in each scope the ADR fixes (AUC, and TPR at FPR 0.01 and
0.001), NNDR, and DCR recomputed under the same distance; an arm with no rows on a holdout is NaN
and counted, and a TPR below 1 / (the scope's non-members) is NaN. The `privacy.csv` holds the
measures `mi-auc`, `mi-tpr01` and `mi-tpr001`, suffixed by scope (`-outside`, `-cell` and `-label`;
`mi-tpr001` has no `-cell`), for the added rows, and `nndr-p5`, `nndr-median`, `nndr-ties`,
`nndr-exact`, `dcr-p5` and `dcr-median` for the added rows and, as the baseline, for the held-out
rows. It writes, for the added rows of every arm, the match attack's `match-auc`, `match-tpr01` and
`match-tpr001`, suffixed by scope as the `mi-` measures are, and the donor reuse measures
`donor-max` and `donor-p95`, NaN for an arm that is neither direct nor perturbation or a fill with
no rows; a fill with rows but no donation reads `donor-max` 0 and `donor-p95` NaN, and a
`bko-micro` fill counts a donation for every id an averaged field names. For a run with a
direct or perturbation arm it loads the pinned releases of the dataset's mapped concepts, as `trace`
does, fetching an absent one into the data root, and re-draws each such arm's gap fill at its `arm`
line's seed to read each row's donors; a perturbation fill's candidate lines are paired with their
draws as `trace` pairs them, and a `bko-coherent` fill's change is re-run so each free field donates
from its leaf donor. Its `privacy.log` opens with the `versions commit=` line, then the data's
fetch, cohort and `data` lines, for a run with a direct or perturbation arm the releases' fetch
lines, then `privacy run=%s stores=%d holdouts=%d reference=%s cell=%s root=%s`, `root` the runs
root read, one `privacy store=%s run=%s folder=%s log_sha256=%s holdouts=%s sha256=%s` line per
store, the sha256 of the bytes read, and `privacy read stores=%d holdouts=%d`. Per holdout it logs,
for an arm with rows whose match attack is undefined in a scope, a WARNING `privacy holdout=%d
arm=%s scope=match-%s undefined members=%d nonmembers=%d` line (INFO for a cell scope with no
member); for each direct or perturbation arm `privacy donor holdout=%d arm=%s donors=%d max=%.0f
p95=%.2f seconds=%.1f`, `max` and `p95` the values the CSV holds and `seconds` the re-draw's; then,
per arm, `privacy holdout=%d arm=%s added=%d` (WARNING for an arm that adds no row) and a WARNING
`privacy holdout=%d arm=%s scope=%s undefined ...` line for an arm with rows whose distance attack
is undefined in a scope (INFO for a cell scope with no member); and last `privacy holdout=%d
done=%d/%d seconds=%.1f elapsed=%.1f`. Then, per scope, arm and measure, one `privacy arm=` line
with the mean, the contrast against the run family's reference arm, set in `REFERENCE` of
`bko.evaluate.reference` (a slice reads its run's), its corrected interval, its family, the holdouts
read, the holdouts `undefined` (NaN contrasts), and `leaks_more=`: `true` when an attack statistic's
interval lies above 0 or a DCR or NNDR quantile's below 0, `undefined` with fewer than two finite
contrasts. Each scope's attack statistics, each scope's match statistics (`scope=match-outside`,
`match-cell`, `match-label`) and the DCR and NNDR quantiles are separate families at family-wise
0.95, sized by their pairs of arm and measure with a defined interval; the cell scope's lines are
descriptive, and where the cell keeps no training record (f = 1.0) its scope reads family=0 and
every value NaN. `mi-tpr001` and `match-tpr001` have no line, nor do the donor measures. The last
line is `privacy rows=%d sha256=%s readings=%d undefined=%d seconds=%.1f`, `undefined` counting the
readings with no interval. `leaks_more` is a reading rule and no criterion.

Each store's folder is the header's folder name directly under the runs root, `--runs-dir DIR`
(resolved to an absolute path; an empty value is refused) or else `runs/` beside the code; a name
that resolves elsewhere, such as a symbolic link out of the root, is refused. A reading run from a
frozen worktree passes the main checkout's `runs/` as `--runs-dir`. `--data-dir` sets the data root.
The command exits 2 for a run with no reference arm, a run lacking it, or a run that removes no
cell, and, for a run with a direct or perturbation arm, when a mapped concept has no pinned release
(`GapError`, a `DatasetError`) or a release cannot be fetched; 3 when a fetched release contradicts
its pin, for a store whose run is neither the named run nor a slice of it, whose header does not
match its folder's `scores.log`, whose data digest differs, whose arm does not hash to the `arm`
line's sha256, whose holdouts overlap another store's or leave one of the run's out, or whose
`split` line the data does not reproduce, and for a direct fill whose re-draw does not hash to its
`arm` line's sha256, or a perturbation fill whose candidate lines do not rebuild its rows or are not
one per draw, or whose lines or stored rows are not their re-draw outside the perturbed fields.

**`truth`** reads each arm's gap fill of a run with a removal against the records the removal
took (ADR [`research/removed-truth-reading`](adr/research/removed-truth-reading.md)). It takes
the run's `rows` stores as `privacy` does, one `--rows` per slice, binds them by the same checks,
and makes no API call:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/truth.csv && test ! -e runs/<run-id>/truth.log && env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m bko.evaluate truth <run> [--runs-dir DIR] $(printf -- '--rows %s ' runs/<store-id>-rows/rows.jsonl <more>...) > runs/<run-id>/truth.csv 2> runs/<run-id>/truth.log </dev/null

Per holdout it re-derives the removed records with `bko.evaluate.prepare.prepared`, keeps each
arm's added rows in the removed cell, and scores them against the removed records with SDMetrics'
quality report, as a replica's fidelity is scored. `real` is left out: its rows are the removed
records. The `truth.csv` columns are `holdout,arm,rows,measure,column,value`: `rows` the arm's
added rows in the cell, `measure` `column-shapes` or `column-pair-trends` with `column` empty, then
`column-shape` once per frame column with `column` the column's name; `value` is empty (NaN) for an
arm with fewer than two rows in the cell, a holdout with fewer than two removed records, or a
column the quality report's details lack. Its `truth.log` opens with the `versions commit=` line,
then the data's fetch, cohort and `data` lines, then `truth run=%s stores=%d holdouts=%d cell=%s
fraction=%s min_rows=%d root=%s`, one `truth store=%s run=%s folder=%s log_sha256=%s holdouts=%s
sha256=%s` line per store, and per holdout the `removal` line, `truth holdout=%d removed=%d`, one
`truth holdout=%d arm=%s rows=%d column_shapes=%.4f pair_trends=%.4f` line per arm and
`truth holdout=%d done=%d/%d seconds=%.1f elapsed=%.1f`. An arm line whose blank values come from
too few rows ends with its cause, ` short=added` (the arm's rows in the cell), ` short=removed`
(the removed records) or ` short=added,removed`, and logs at INFO; a NaN score with enough rows on
both sides logs at WARNING. The last line is `truth lines=%d sha256=%s seconds=%.1f`, the CSV's
rows and sha256. `--runs-dir` and `--data-dir` read as `privacy`'s do. It exits 2 for a run that
removes no cell, and 3 for a store that fails `privacy`'s store checks, a store of another run or
whose `split` line the data does not reproduce included.

**`representativeness`** reads each arm's gap fill of a run against the fused shares of the
cells it fills (ADR [`research/share-based-quota`](adr/research/share-based-quota.md), E21's
reading). It takes the run's `rows` stores as `privacy` does, one `--rows` per slice, binds them
by the same checks, and makes no API call:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/representativeness.csv && test ! -e runs/<run-id>/representativeness.log && uv run --no-sync python -m bko.evaluate representativeness <run> [--runs-dir DIR] $(printf -- '--rows %s ' runs/<store-id>-rows/rows.jsonl <more>...) > runs/<run-id>/representativeness.csv 2> runs/<run-id>/representativeness.log </dev/null

Per holdout it re-derives the training split and its flagged cells, reads each cell's fused
concept shares as the direct sampler draws them (`bko.gaps.quota.fused_shares`), and keeps each
arm's added rows in the cell. The `representativeness.csv` columns are
`holdout,arm,cell,rows,gap,concept,cell_share,real_share,recall`, one row per holdout, arm and
flagged cell: `rows` the arm's added rows in the cell; `gap` the largest absolute difference
between the share of those rows showing a mapped concept and its fused share, and `concept` that
concept, both empty for an arm with no row in the cell; `cell_share` the cell's share of the
arm's training set, the training split plus the fill; `real_share` the same for `real`, whose
fill is the removed records; and `recall` the arm's regression recall on the flagged subgroups of
its augmented training set, read from the store's folder's `scores.csv`, which on E6's cell is
removed-cell recall; it reads that file only once its sha256 is the one the folder's `scores`
line logs, as `rows` does. Its `representativeness.log` opens with the `versions commit=` line,
then the data's fetch, cohort and `data` lines, then `representativeness run=%s stores=%d
holdouts=%d root=%s`, one `representativeness store=%s run=%s folder=%s log_sha256=%s
holdouts=%s sha256=%s` line per store, and per holdout, the first time it reads a store,
`representativeness scores folder=%s sha256=%s`, the sha256 of its folder's `scores.csv`, then
one `representativeness holdout=%d arm=%s key=%s rows=%d gap=%.4f concept=%s cell_share=%.6f
real_share=%.6f recall=%.4f` line per arm and flagged cell and `representativeness holdout=%d
done=%d/%d seconds=%.1f elapsed=%.1f`. The last line is `representativeness lines=%d sha256=%s
seconds=%.1f`, the CSV's rows and sha256. `--runs-dir` and `--data-dir` read as `privacy`'s do.
It exits 2 for a run without `real` or a store whose folder holds no `scores.csv`, and 3 for a
store that fails `privacy`'s store checks or whose folder's `scores.csv` is not the one its
`scores` line logs.

**SynthEval.** `syntheval` scores each replica fill of a run's `rows` store under the SynthEval
metric set of Lautrup et al.'s code (`bko.evaluate.syntheval_inputs.METRICS`; ADR
[`research/syntheval-reading`](adr/research/syntheval-reading.md)), each fill against its
holdout's training split as the real data and the test split the run scored as the holdout data:
for a run that decontaminates, the split without the records any stored fill of the holdout
copies. It needs `make syntheval-env` ([Benchmark environments](#benchmark-environments)), runs
`rows` first, and makes no API call:

    mkdir -p runs/<run-id>-syntheval && test ! -e runs/<run-id>-syntheval/syntheval.csv && test ! -e runs/<run-id>-syntheval/syntheval.log && uv run --no-sync python -m bko.evaluate syntheval <run> --folder runs/<run-id> --rows runs/<run-id>-rows/rows.jsonl > runs/<run-id>-syntheval/syntheval.csv 2> runs/<run-id>-syntheval/syntheval.log </dev/null

For each fill it writes the real, test and synthetic tables and the metric set into a temporary
directory, runs `score.py` under the `.venv` of `--bench-dir DIR` (default: this checkout's
`bench/syntheval/`), seeded with the fill's holdout seed, and reads back the metric name to value
object it writes. The script gets `PATH`, `HOME`, the locale variables and
`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`, nothing else, and stops at
`bko.evaluate.syntheval_reading.TIMEOUT_SECONDS`. A reading run from a frozen worktree under
`PYTHONPATH` passes `--bench-dir <repo>/bench/syntheval`, since the worktree holds no
environment. The `syntheval.csv` holds one row per holdout, replica fill (`<arm>-replica`) and
metric SynthEval returns, `holdout,arm,metric,value`, a value SynthEval could not compute or
returned non-finite empty. Its `syntheval.log` opens with the `versions commit=` line and
`syntheval store=%s sha256=%s`, then logs per fill `syntheval start holdout=%d arm=%s`, one
`syntheval input holdout=%d arm=%s file=%s sha256=%s` line per input file and
`syntheval holdout=%d arm=%s rows=%d metrics=%d`, and closes with `syntheval rows=%d sha256=%s`,
the CSV's rows and sha256. It exits 2 when the environment is absent, naming
`make syntheval-env`, when the script exits non-zero, with the last lines of its stdout and
stderr, runs past its timeout, or writes no metrics; 3 for a folder or store that fails `rows`'
identity checks, or a holdout or replica arm the folder and store do not hold.

**The perturbation runs.** `e10-pilot-thin` and `e10-pilot-empty` run two holdouts per removal
fraction, each at a $5.00 ceiling, and judge no criterion (ADR
[`research/identifier-perturbation`](adr/research/identifier-perturbation.md)). Their paid arm
is `bko-direct-perturbed`, which sends the 19 unmapped fields of rows of open Diabetes 130 data to
the Messages API ([Generation](#generation), What leaves the machine); `bko-direct-pram` is free.
Each arm keeps its own rows on every holdout; each holdout logs the ADR's distribution reading of
them, which changes no row.

Each pilot is a paid run, so it starts with `launch` (the paid-run protocol above); the dry
run writes `launch.log` and starts nothing:

    uv run --no-sync python -m bko.evaluate launch <run-id> e10-pilot-thin --dry-run
    uv run --no-sync python -m bko.evaluate launch <run-id> e10-pilot-thin
    uv run --no-sync python -m bko.evaluate launch <run-id> e10-pilot-empty --dry-run
    uv run --no-sync python -m bko.evaluate launch <run-id> e10-pilot-empty

`<run-id>` is a run of record under entry `e10`: one id for a run's dry run and its launch, another
for the second run. A recorded pilot's rows store is read through `privacy` against `bko-direct`
(`privacy` above).

The PRAM-only full runs, `e10-pram-thin` and `e10-pram-empty`, hold no paid arm, so they spend
nothing and need no credential. `launch` refuses a run with no paid arm, so each slice runs
through `score` directly, as E9's do (the guarded command in E9's paragraph above, with
`e10-pram-thina` for the run name), refusing `--log`, `--replay`, `--resume` and `--rehearse` and
requiring `--candidates`. Run all ten slices of a run at one commit: `merge` refuses slices whose
`versions` lines differ. A slice with no call log writes no `run status=` line, so its completion
is the last `holdout=... done=N/N` line of `scores.log` plus a non-empty `scores.csv`. `score` run
directly has no keep-awake or battery guard, so keep the machine on AC power and awake.

`merge` joins each run's ten slices, and `rows`, `panel` and `privacy` read them against
`bko-direct`.

Both arms, and `bko-coherent`, log to `scores.log` (`bko.evaluate.perturb`, `bko.evaluate.pram`,
`bko.evaluate.perturbed`, `bko.evaluate.coherent`):

- `perturb arm=%s fields=%s` at construction, ending `theta=%s` for PRAM,
  `per_call=%d model=%s` for the perturbed arm and `stream=%d` for `bko-coherent`.
- `perturb pram holdout=%d arm=%s rows=%d`: the rows PRAM read; `bko-coherent` logs
  `perturb coherent holdout=%d arm=%s rows=%d` in its place.
- `perturb call=%d holdout=%d rows=%d returned=%d accepted=%d refused=%d reason=%s stop_reason=%s
  checks=%s quota_off=%d` per rewrite call: `returned` the items its reply decoded to, `reason`
  the first refusal (`empty`, `short`, `undecodable` or `order` for the whole call; `keys`,
  `validate` or `bounds` for a row; `none`), `checks` the `bko.validate` checks its `validate`
  rows failed, `quota_off` how far the call's final rows fall short of its stated counts, summed
  over the fields and their categories (`bko.evaluate.quotas`). It reads each row as the merge
  left it, so a refused row's or call's originals count against the quotas.
- `perturb holdout=%d rows=%d perturbed=%d unchanged=%d refused_empty=%d refused_short=%d
  refused_undecodable=%d refused_order=%d refused_keys=%d refused_validate=%d refused_bounds=%d`
  per holdout: each drawn row's merge outcome.
- `perturb shapes holdout=%d arm=%s shapes=%.6f unperturbed=%.6f difference=%.6f
  pair_trends=%.6f unperturbed_pair_trends=%.6f incoherent=%s unperturbed_incoherent=%s
  rows_changed=%d values_changed=%d` per holdout: SDMetrics' column shapes of the arm's rows and
  of the drawn rows against the holdout's label-1 training rows over the perturbed fields,
  `difference` the first less the second, both column pair trends, the share of each set's rows
  breaking any of the ADR's three Diabetes rules (`none` off Diabetes), `rows_changed` the lines
  whose `perturb` is `perturbed` or `changed`, and `values_changed` the values that differ from
  the draw. A score SDMetrics cannot read logs `nan`. A fill that drew no row
  logs `perturb shapes holdout=%d arm=%s empty=true` in its place.
- `perturb columns holdout=%d arm=%s kind=%s <field>=%.6f ...` twice per holdout, `kind`
  `perturbed` then `unperturbed`: each perturbed field's column shape, in field order. A fill that
  drew no row logs neither line.
- `perturb unscored holdout=%d field=%s`, a WARNING: a field SDMetrics' column-shapes details
  lack, whose column shape logs `nan`.
- `perturb summary arm=%s holdouts=%d read=%d empty=%d rows=%d rows_changed=%d values_changed=%d
  mean_shapes=%s mean_unperturbed=%s` at the end of the run, a stopped run included: the holdouts
  read and with no row drawn, and the mean column shapes of the arm's and the drawn rows over the
  read holdouts, at six places, `nan` when a read holdout's shape is `nan` and `none` when no
  holdout was read; the perturbed arm's adds its run's merge outcomes as the holdout line names
  them.

A rewrite call's `generate` line carries `rewrite=true` in place of `level=`, and its `short`
WARNING ends `rewrite=true`. Each arm's candidates-file lines are `bko-direct`'s, with a row a
rewrite call returned carrying that call and its position there, and each line's `perturb` says
what the step did to it.

**`trace`** explains one arm's fill of one holdout: for each record, the stored or re-derivable facts
that determine it (ADR [`research/record-explainability`](adr/research/record-explainability.md)).
It makes no API call. `--arm` takes the `arm` line's arm, `<arm>-replica` for a replica fill,
`--record N` writes the record at 0-based position N alone, and `--data-dir` sets the data root.
It reads the data the run used from the data root, and for a direct or perturbation arm the
pinned ontology releases too, fetching what is absent ([Data](#data), [Ontologies](#ontologies)).
Every arm of `DIRECT_ARMS` (`bko-direct`, `bko-logic`, `bko-direct-pram`, `bko-nomap`, `bko-capped`,
`bko-micro`, `bko-tree`, `bko-hier-0`, `bko-hier-w`, `bko-hier-closed`, `bko-coherent`, `bko-donor`,
`bko-pool-random`, `bko-pool-all`, `bko-kb`) re-draws at its `arm` line's seed with that arm's
draw spec and no extra random draw; a `bko-logic` or `bko-kb` record's concepts are its closed
draw, not share priors, a `bko-nomap` record names no concept, nor does a pooled direct arm's,
and a `bko-tree` record's concepts end in one entry per tree field: its `field`, `leaf`, `concept`
and `label`, and `parts`, each level's node from the top's child down to the leaf with both counts
at the node and its parent, the weight, the conditional and the share. A `bko-hier` record's
broad concepts add `hierarchy` beside their share facts: the narrow concept `given`, the weighted
counts `both` and `narrow`, the `pseudo`-count (null when closed), the `conditional`, the
`complement` (null when the narrow concept's share is within one float of 1) and whether it was
`clamped`. A `bko-micro` record adds `averaged`, each averaged field's three donor training record
ids, the primary's first, its `sources` entry the primary's; a record with no averaged field writes
no `averaged` key. `bko-direct-pram`, `bko-direct-perturbed` and `bko-coherent` rebuild their rows
from their candidate lines and re-draw underneath, and each of their records gets a `perturb`
object: `result`, the line's schema-4 `perturb` value (on a schema-3 line, `perturbed` or
`unchanged` by comparison with the draw when the line names a call, `kept` when it names none, and
PRAM's `changed` or `unchanged`); `fields`, the fields that differ from the draw; and `drawn`, the
draw's values for them. `bko-coherent` then re-runs its change at the fill's seed, which reads its
own stream, refuses unless the changed rows are the lines' records, and names each free field's leaf
donor in `sources`. The other arm kinds read their facts from the run folder, a Messages arm's from
its `calls.jsonl`. It writes `trace.jsonl`, one sorted-key JSON line per record (one per fill for a
baseline), which holds source training record ids; keep the file in its run folder. `<run-id>` is
the folder read, `<run>` the named run, `<h>` the holdout and `<arm>` the arm:

    mkdir -p runs/<run-id>-trace && test ! -e runs/<run-id>-trace/trace.jsonl && test ! -e runs/<run-id>-trace/trace.log && uv run --no-sync python -m bko.evaluate trace <run> --folder runs/<run-id> --holdout <h> --arm <arm> > runs/<run-id>-trace/trace.jsonl 2> runs/<run-id>-trace/trace.log </dev/null

- **Log lines**: `trace start` names the folder, run, holdout, arm, fill and seed before any
  re-draw; for a direct or perturbation arm, `trace arm` names the `arm` line's sha256 and the
  re-drawn or rebuilt one, the arm's own check and `matches=`, at WARNING when either check
  fails. A direct arm's check is `kept_lines=`, whether its re-drawn kept records are the
  candidates file's kept lines; a perturbation arm's is `unperturbed=`, whether its returned
  attempt holds one line per draw and every column no perturbation changes is its draw's.
  `bko-coherent`'s is `coherent=`, whether that holds and its change re-run gives every line's
  record; when the unperturbed check fails, the `trace coherent` WARNING says the re-run was
  skipped (`rerun=skipped failed=unperturbed`) and the check stays `unperturbed=`. For a
  Messages arm, the `trace partial` WARNING counts the records left `untraced`, names the calls
  the log lacks (`missing=`) and each reason; and `trace rows=` closes with the output's sha256,
  the record written (`record=`) and the untraced count.
- **Exit codes**: 0 the trace is written; 1 an unexpected error, with its traceback; 2 a holdout,
  arm or record the folder does not hold, or a `calls.jsonl` line that is not one the call log
  writes, named by path and line number; 3 a folder that fails its identity checks: a holdout's
  `split` line the data does not reproduce, a direct arm's re-draw whose rows do not hash to the
  `arm` line's sha256, or whose kept records are not the candidates file's kept lines, a direct
  fill whose `arm` line's params are not a JSON object or whose logged spec fields (`presence`,
  `mapped`, `min_donors`, `reuse_cap`, `tree`, `hierarchy`, `closed`, `pool`, `donors`,
  `microaggregate`) differ
  from its arm's, and
  a perturbation fill whose rebuilt rows miss its `arm` sha256, whose returned attempt is not one
  line per draw, or whose unperturbed columns differ from the re-draw, or a `bko-coherent` fill
  whose change re-run does not give its lines' records. A `bko-logic` trace over an ontology whose logic
  components cannot be enumerated, a `bko-kb` trace over a frame `bko-kb` refuses, and a
  perturbation trace over a frame whose unmapped fields are not its pinned ones, exit 2.

**`attribution`** reads `trace.jsonl` files and counts, per fill, what each record field can be
attributed to (ADR [`research/record-explainability`](adr/research/record-explainability.md),
decision 7): `knowledge`, a field a mapping row of `<run>`'s dataset reads under an arm whose draw
reads the mapping table; `donor`, a field with a source training record id, a field
`bko-coherent` changed among them; `donor-mean`, a field a `bko-micro` record's `averaged`
names, ranked after `knowledge` and before `donor`; `prompt`, a Messages record's field or a
rewritten one; `method-seed`, any other, a PRAM-changed field and every field of an `untraced`
record among them. A pooled field of `bko-donor`, `bko-pool-random` or
`bko-pool-all` is `donor`, and a `bko-kb` record's mapped fields are `knowledge` and its `age`
`method-seed`. It reads an arm's kind from each line, reads no data and
makes no API call. Each record arm's fill gets one row per class, zeros included, in the unit
`field`, but a `donor-mean` row only for an arm whose spec sets `microaggregate`, so every other
fill's rows keep their bytes, and a fill naming an averaged field under any other arm exits 2; a
baseline's fill gets one `method-seed` row in the unit `fill`, since its trace holds no
record. `<run>` names the dataset whose field rows and frame columns are read, and `--trace`
takes one or more files:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/attribution.csv && test ! -e runs/<run-id>/attribution.log && uv run --no-sync python -m bko.evaluate attribution <run> --trace runs/<trace-id>/trace.jsonl <more>... > runs/<run-id>/attribution.csv 2> runs/<run-id>/attribution.log </dev/null

- **Log lines**: `versions commit=` first; `attribution spec=` once per fill of a perturbation
  arm outside the direct arms (`bko-direct-perturbed`) names every field of the `bko-direct`
  spec its unchanged fields are read over, so it gains each field the spec gains; `attribution trace=` per file names its sha256, its lines and its
  `untraced` records; `attribution rows=` closes with the CSV's data rows, its sha256 and the file
  count.
- **Exit codes**: 0 the CSV is written; 1 an unexpected error, with its traceback; 2 a file that
  cannot be read or is not UTF-8; a line that is not a JSON object or not the shape `trace`
  writes, each named by its file and line number; an arm name other than lowercase letters, digits
  and hyphens; a holdout, arm and record id that two lines across the files name; a record whose
  fields are not `<run>`'s frame columns; a fill whose lines are of two kinds; a `perturb` key
  under an arm that is not a perturbation arm; or a direct line whose arm is not a direct arm.

## BKB fixtures

`bko.bkb` ([README](../src/bko/bkb/README.md)) is tested against what pybkb computes on its own
example files (ADR [`research/bkb-layer-source`](adr/research/bkb-layer-source.md)). pybkb does not
resolve here, so its results are committed as JSON under `tests/pybkb/` rather than recomputed,
and the gate never runs pybkb.

- **What is committed**: `tests/pybkb/export.py`, which writes the fixtures; the requirements it
  runs under, with hashes (`tests/pybkb/oracle-requirements.txt`); one document per pybkb example
  and per fused BKB, in the format `bko.bkb.serialize` reads; `queries.json`, pybkb's update
  results; `manifest.json`, with the pybkb commit, the sha256 of every input, every output and the
  exporter, the versions it ran under and pybkb's own test checks; and `run.json`, the hash seed.
- **Regenerating**: make a private scratch directory with `O=$(mktemp -d)`, fetch every path
  `FILES` in `export.py` names, at the commit it names, into `$O/src` and nothing else (the
  exporter refuses any other file there), then from the repository root build a throwaway Python
  3.11 environment outside the repository and run the exporter in it. That environment is not the
  project's, so `uv pip install` there leaves `uv.lock` and `.venv` untouched:

      UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR=$O/python UV_CACHE_DIR=$O/cache \
        uv venv --no-config --python 3.11 $O/venv
      UV_CACHE_DIR=$O/cache uv pip install --no-config --python $O/venv/bin/python \
        --require-hashes --only-binary :all: -r tests/pybkb/oracle-requirements.txt
      env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONHASHSEED=0 \
        $O/venv/bin/python -P -s -B tests/pybkb/export.py $O/src $O/out/seed-0

  Hash seeds 0, 1 and 2, each into its own `out/seed-N`, must give the same files, `run.json`
  aside. Copy the seed-0 output's `*.json` into `tests/pybkb/`, stage it, and delete
  `$O`. `-P` keeps the script's directory off `sys.path`; `-I` would ignore
  `PYTHONHASHSEED`.
- **What holds the fixtures to the run**: the tests ADR
  [`research/bkb-layer-source`](adr/research/bkb-layer-source.md) lists under Enforcement.

## Ontologies

`bko.ontology` ([README](../src/bko/ontology/README.md)) reads the MONDO and HPO releases
`bko.datasets.sources` pins, and holds the table that maps their concepts to UCI fields, to
Synthea condition codes, and to Diabetes 130's ICD-9 diagnosis categories and lab states, with the
codes and categories it excludes (ADR
[`research/ontology-mapping-design`](adr/research/ontology-mapping-design.md)). The table's check,
inside `.venv`:

    uv run --no-sync python -m bko.ontology check [--data-dir <dir>]

- **What it does**: it fetches the release of each prefix its rows name into the data root if
  absent, from `bko.ontology.mapping.RELEASES` or, for a prefix those lack, a catalog entry's
  release map, verifies each against its pin, converts an OWL-only one to OBO, which needs a JDK
  until the converted file exists (OWL-only releases below), parses them with pronto (and each
  xref's qualifiers with fastobo), and checks every row:
  a concept row's concept exists and is not obsolete, its label is the release's name, a Synthea row's SNOMED-CT code is a `skos:exactMatch`
  of its concept and of no other live concept, an excluded Synthea code is a `skos:exactMatch` of no
  live MONDO concept (the check reads MONDO only for it), an ICD-9 row's concept carries its
  `xref: ICD9:<code>` (any qualifier) and the live concepts carrying that xref are exactly the row's
  declared holders, a generated code row likewise but counting only xrefs qualified
  `source="MONDO:equivalentTo"` (a row of `bko.ontology.icd9_codes` or
  `bko.ontology.icd10cm_codes`), an excluded ICD-9 category is an xref of no live MONDO concept, an
  occupation row's concept carries exactly its SOC id as its `OCCO:00000008` literal, and an
  out-of-scope concept is not an `is_a` ancestor of a mapped one. stdout gets a CSV with the columns
  `table` (`field`, `code`, `out-of-scope`, `excluded-code`, `icd9`, `icd9-code`, `icd10cm-code`,
  `excluded-icd9` or `occupation-code`), `subject` (the dataset side, such as `cp in 1|2|3`,
  `A1Cresult in >7|>8`, `SCTID:<code>`, `ICD9:<code>`, `ICD10CM:<code>` or `OCCP:<code>`),
  `concept` (empty for an excluded code or category), `label` and `status` (`ok`, or how the
  release disagrees; for an `excluded-code` or `excluded-icd9` row, `ok` means the code still has
  no live holder), one row per checked row, the same bytes on every run over the same releases.
- **What it logs**: stderr gets the parser's version, each release's snapshot identity (as in
  [Data](#data)), the `data-version` its header names, its term counts, one warning per
  disagreeing row, and a warning for each OWL-only release left unconverted for want of a JDK.
- **Exit codes**: 0 every row agrees; 1 an unexpected error, with its traceback; 2 the command
  could not run; 3 a release contradicts its pin or is not OBO the loader reads (stdout is
  empty); 4 a row disagrees with its release (the report is written first).
- **Re-pinning**: edit the release's `Source` (URL, size, sha256), run this command, review each
  row it reports, then run the release tests below.
- **The gate does not run it.** `make check` tests the loader on OBO text built in `tmp_path` and
  the table's invariants; a row naming a concept the release lacks passes `make check` and fails
  this command, so run it after editing the table or re-pinning a release.
- **The release tests**: `tests/test_ontology_pinned_releases.py` loads MONDO, HPO, DO and the
  converted OccO from the data root and asserts that every row agrees and that the
  [fusion reproduction](#fusion-reproduction) differs nowhere. The gate skips them, and no `-m`
  expression runs them; `--releases` does:

      uv run --no-sync pytest tests/test_ontology_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

  They read and never fetch or convert, so run `check` and `reproduce` first: `check` fetches
  MONDO, HPO and OccO and converts OccO, `reproduce` fetches MONDO and DO. An absent release fails
  them. Run them after a re-pin, a change to `bko.ontology.obo`, or a change to pronto or fastobo
  in `uv.lock`, from the checkout whose `data/` holds the releases.
- **OWL-only releases**: a release pinned as OWL, a `Source` whose `filename` ends `.owl`, is
  converted to OBO when a command first loads it: the pinned ROBOT jar's `convert`, run with no
  reasoning, writes a hidden partial file beside the OWL in its snapshot directory, renamed to
  `<owl stem>.robot-<jar sha256[:16]>.obo` once ROBOT exits 0. Later loads reuse that OBO, and a
  re-pinned jar converts again. Each load logs the OBO's size and sha256. A failed conversion
  leaves no OBO and keeps ROBOT's stderr at the OBO's path with `.stderr` added; its message
  quotes the last lines. Conversion needs a JDK, found as
  [ELK comparison](#elk-comparison) finds it. Exit 2 means no JDK or a ROBOT failure; exit 3 an
  OWL file that names an import, refused before ROBOT runs, or a converted file the loader
  refuses. `check` alone goes on without a JDK: it leaves the unconverted release out with a
  warning, so its rows report `no release for prefix OCCO` and exit 4 while every other row still
  checks. OccO, which ACS's occupation rows name, is the one OWL-only release; `check`, the
  `occupations` command and every command that reads the `acs` dataset convert it (ADR
  [`research/ontology-mapping-design`](adr/research/ontology-mapping-design.md)).
- **The generated code rows**: `bko.ontology.icd9_codes` maps each exact diagnosis code of the
  `diabetes-codes` cohort to the one MONDO concept that carries its xref under
  `source="MONDO:equivalentTo"` (ADR
  [`research/diabetes-diagnosis-codes`](adr/research/diabetes-diagnosis-codes.md)). It is
  written, never edited. A shell redirect truncates its target before the command runs, and every
  `bko.ontology` import reads the module, so write to a scratch file and move it only on exit 0:

      mkdir -p scratchpad && uv run --no-sync python -m bko.ontology codes > scratchpad/icd9_codes.py && mv scratchpad/icd9_codes.py src/bko/ontology/icd9_codes.py

  then `make format`. The command fetches MONDO and the archive into the data root if absent,
  writes nothing to stdout until the module text is complete, and logs one
  `codes unmapped reason=... codes=N` line per reason and a closing `codes rows=N sha256=...`
  line. `tests/test_ontology_icd9_codes_pinned.py` regenerates the rows from the pinned release
  and cohort under `--releases` and compares them with the committed ones; run it with the
  release tests after a re-pin.
- **The generated ICD-10-CM rows**: `bko.ontology.icd10cm_codes` maps the ICD-10-CM id each
  four-character diagnosis token of the `nhamcs` records reads as to the one MONDO concept that
  carries its xref under `source="MONDO:equivalentTo"` (ADR
  [`research/nhamcs-frame`](adr/research/nhamcs-frame.md)). It is written, never edited, through a
  scratch file as above:

      mkdir -p scratchpad && uv run --no-sync python -m bko.ontology codes --system icd10cm > scratchpad/icd10cm_codes.py && mv scratchpad/icd10cm_codes.py src/bko/ontology/icd10cm_codes.py

  then `make format`. The command fetches MONDO and the NHAMCS archive into the data root if
  absent, writes nothing to stdout until the module text is complete, and logs one
  `codes unmapped system=icd10cm reason=... codes=N` line per reason and a closing
  `codes system=icd10cm rows=N tokens=N sha256=...` line.
  `tests/test_ontology_icd10cm_codes_pinned.py` regenerates the rows under `--releases` and
  compares them with the committed ones.
- **The generated occupation rows**: `bko.ontology.occupation_codes_low` and
  `bko.ontology.occupation_codes_high` map each `OCCP` code of the ACS cohort to the Occupation
  Ontology class its `SOCP` code reaches by SOC level, split at SOC major group 31 so each module
  stays under the size cap (ADR [`research/acs-frame`](adr/research/acs-frame.md)). They are
  written, never edited, each through a scratch file as above:

      mkdir -p scratchpad && uv run --no-sync python -m bko.ontology occupations --part low > scratchpad/occupation_codes_low.py && mv scratchpad/occupation_codes_low.py src/bko/ontology/occupation_codes_low.py
      mkdir -p scratchpad && uv run --no-sync python -m bko.ontology occupations --part high > scratchpad/occupation_codes_high.py && mv scratchpad/occupation_codes_high.py src/bko/ontology/occupation_codes_high.py

  The command fetches OccO's OWL file and the ACS archive into the data root if absent, converts
  the OWL to OBO with ROBOT (see OWL-only releases above; it needs a JDK), writes nothing to
  stdout until the module text is complete, and logs one `occupations unmapped parts=all
  reason=... codes=N` line per reason, counted over every code whichever part runs, and a closing
  `occupations part=... rows=N of=M codes=C holders=H sha256=...` line: `C` the `OCCP` codes read
  and `H` the live classes carrying a SOC id.
  `tests/test_ontology_occupation_codes_pinned.py` regenerates both parts from the pinned release
  and archive under `--releases` and compares them with the committed ones:

      uv run --no-sync pytest tests/test_ontology_occupation_codes_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null

  It reads and never fetches or converts, so run the command first.

### ELK comparison

`python -m bko.ontology elk` classifies the pinned MONDO and HPO OWL releases with ELK and compares
each classification with the hierarchy `bko.ontology` loads from the OBO release of the same
version (ADR [`research/ontology-mapping-design`](adr/research/ontology-mapping-design.md), ELK
comparison); it is E9's hierarchy comparison (ADR
[`research/logic-only-ablation`](adr/research/logic-only-ablation.md)). A run of record writes its
folder ([Runs](#runs)):

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/hierarchy.csv && test ! -e runs/<run-id>/hierarchy.log && uv run --no-sync python -m bko.ontology elk > runs/<run-id>/hierarchy.csv 2> runs/<run-id>/hierarchy.log </dev/null

- **What it needs**: a JDK, found as `make synthea` finds one ([Data](#data)) and resolved before
  anything is fetched, and a machine with the 16 GB Java heap `bko.ontology.elk.ELK_MEMORY_GB`
  sets free. It takes `--data-dir <dir>` as `check` does.
- **What it does**: it fetches the OBO and OWL releases and the ROBOT jar into the data root if
  absent and verifies each against its pin, and refuses an OWL file whose bytes name an import
  (`bko.ontology.elk.IMPORTS`) before Java runs; the sha256 pin guarantees the bytes, and the
  import check is a tripwire on them. For each release it runs ROBOT as a subprocess in the OWL
  file's snapshot directory, `reason --reasoner ELK --equivalent-classes-allowed all` then
  `convert --check false` to OBO, under `bko.ontology.elk.ELK_TIMEOUT_SECONDS`; the classified
  OBO file is kept under `data/elk/`, with ROBOT's stderr written beside it as it runs, and a
  stale pair is removed first. The comparison reads transitive ancestor sets over every concept
  `bko.ontology.mapping.MAPPED` names and the ancestors either side gives it, so a link one side
  states directly and the other reaches through a chain is no difference; ROBOT's
  `equivalent_to:` lines read as `is_a` both ways.
- **What it writes**: stdout gets a CSV with the columns `release`, `child`, `ancestor` and `side`
  (`loaded` or `elk`), one row per ancestor pair only that side holds, the loaded side's rows
  first; a header alone means the two agree in scope. One missing direct link adds an ancestor
  pair for every class below it and every ancestor above it, so the row count is not a count of
  links; the `minimal_` counts below are.
- **What it logs**: the `elk tool` line (ROBOT's URL and sha256, the Java version, the heap in
  GB and the timeout in seconds); per release an `elk classify` line before ROBOT runs (the OWL
  file's sha256, the heap, the timeout), an `elk classified` line (seconds, the classified file's
  path, bytes and sha256) and an `elk compare` line: `loaded_terms=` and `elk_classes=` (each
  side's classes), `concepts=` (the scope), `absent_loaded=` and `absent_elk=` (classes in scope
  a side has no term for), `only_loaded=` and `only_elk=` (ancestor pairs) and `minimal_loaded=`
  and `minimal_elk=` (pairs no other pair on that side implies); then
  `elk differing ancestor_pairs=%d sha256=%s`, the CSV's sha256.
- **HPO caveat**: HPO's build makes `hp.owl` as a merge of the editors' file and `hp.obo` from its
  simple non-classified release, with no reasoner (ADR
  [`research/ontology-mapping-design`](adr/research/ontology-mapping-design.md), Context), so an
  HPO pair that differs can come from that build difference as well as from classification.
- **Exit codes**: 0 the report is written, whatever it lists; 1 an unexpected error, with its
  traceback; 2 the command could not run, including no JDK, ROBOT failing, timing out or writing
  no file, whose message names the stderr file and quotes its last lines; 3 a release contradicts
  its pin, is not OBO the loader reads, or is OWL that names an import, or ROBOT's OBO has an
  `id:`, `is_a:` or `equivalent_to:` line with no value (stdout is empty).
- **The gate never runs Java**: `make check` tests the comparison and ROBOT's argv with a fake
  runner and canned OBO text.

## Fusion reproduction

`bko.ontology.santos` rebuilds the MONDO/DO fusion example of Santos et al. (section 7 of
[the paper](https://doi.org/10.1371/journal.pone.0296864)) from the pinned MONDO and DO releases, with
the fragment encoding of ADR `research/bko-fragment-encoding`. One command, inside `.venv`:

    uv run --no-sync python -m bko.ontology reproduce [--data-dir <dir>]

`make reproduce` runs the same command with the default data root.

- **What it does**: it fetches MONDO and DO into the data root if absent, verifies each against
  its pin and parses it, builds the MONDO, DO and bridge fragments over the paper's terms, fuses
  them at weight 1, instantiates the result from the paper's anchor, and checks each quantity the
  paper reports. stdout gets a CSV with the columns `kind` (`term`, `pta`, `instantiation`,
  `inference` or `chain`), `item`, `paper` (what the paper prints, empty where it prints nothing),
  `measured` and `status`: `match`; `erratum`, where the paper misprints or contradicts its own
  algorithm and the releases give what it means; `differs`; or `reported`, a measurement the paper
  gives no value for. The same bytes on every run over the same releases.
- **What it logs**: stderr gets the parser's version, each release's snapshot identity, the size of
  the whole-MONDO fragment, each fused BKB's size and name (`fused` at the paper's weights, `pybkb`
  under pybkb's normalization), one warning per `differs` row, and a summary line.
- **Exit codes**: 0 the report is written, whatever its statuses; 1 an unexpected error, with its
  traceback, including a fusion the converter should never have built; 2 the command could not
  run; 3 a release contradicts its pin or is not OBO the loader reads. The verdict is in the rows;
  [RESEARCH.md](RESEARCH.md#detect) reads it.
- **Re-pinning**: edit the `MONDO` or `DO` `Source`, run this command and `check`, and review each
  row that is not `match`; `tests/_ontology_fixtures.py` names the releases its abridged text
  comes from.
- **The gate does not run it.** `make check` runs the same reproduction on abridged release text
  built in `tmp_path` (`tests/test_ontology_santos.py`), so a release that drifts from the paper
  passes `make check` and shows here. The release tests ([Ontologies](#ontologies)) assert it
  differs nowhere.

`bko.bkb.fusion.check_fused` checks that a fused BKB is valid; its docstring and ADR
`research/data-contradiction-handling` (Decision 3) state what it requires.

## Gap detection

`bko.gaps` ([README](../src/bko/gaps/README.md)) lists a dataset's gaps as (cell,
statement) pairs under each definition of ADR `research/bko-formal-gap-definition`
(`per-subgroup`, `population-level` and `frequency-aware`, in that order), and as the age-sex
subgroup-label cells holding fewer records than `UNDERREPRESENTED_BELOW` (`count`). The
commands, inside `.venv`:

    uv run --no-sync python -m bko.gaps uci [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps support2 [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps acs [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps nhanes [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps nhamcs [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps physionet2012 [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps synthea [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps first-flag [--data-dir <dir>]

- **What `uci` does**: it fetches UCI Heart Disease and the releases the field mappings name into
  the data root if absent, splits the records as [Data](#data) says, and lists the training
  split's gaps. stdout gets a CSV whose column `detector` names a definition or `count`. A
  definition's row fills the statement's `source`, `antecedent`, `antecedent_label`, `consequent`
  and `consequent_label` and its cell's `age_band`, `sex` and `target`; a population-level gap
  holds in every cell, so it has a row per cell. A `count` row fills the cell and its `count`.
- **What `support2`, `acs`, `nhanes`, `nhamcs` and `physionet2012` do**: each catalogued dataset
  (`bko.ontology.catalog`) has its own command, `support2`, `acs`, `nhanes`, `nhamcs` and
  `physionet2012` among them: it fetches the dataset's files
  and the releases its field rows name into the data root if absent, splits the records at
  `SPLIT_SEED` under the frame's age bands, and lists the training split's gaps in `uci`'s
  columns, logging as `uci` does (ADR `research/support2-frame`; ACS's releases are HP and OccO,
  ADR `research/acs-frame`).
  Before the listing's lines it logs the loader's `cohort` line, one
  `data dataset=<name> frame=<name> records=<n> digest=<sha256> agreement=<a>` line for the
  loaded records, the line `score` logs for them, and one `split seed=<s> train=<n> test=<n>
  fingerprint=<sha256>` line, the line `uci_split` logs.
- **What `synthea` does**: it generates the population `bko.gaps.experiment.E2_RUN` names if absent
  ([Data](#data)), fetches the releases the code mappings name, observes every mapped condition
  code, and for each ablation in `E2_ABLATIONS`, each removing all of a spec's cases or a seeded
  fraction of them (`cell-50` is half of the cell's cases), lists every definition's gaps and the
  count-based ones before and after the removal. stdout gets a CSV with the column `ablation`, then `uci`'s columns without
  `count`, then `before`, `after` and `truth`: whether the row is a gap before the removal, after it, and
  among the gaps the removed patients account for. Every cell uses the ablation's age bands
  (`AblationSpec.edges`). A definition's cells carry no label, so its rows leave `target` blank,
  and a (cell, statement) pair is listed when it is a gap before or after the removal or in the
  truth; a `count` row's cell is listed when it is a gap before or after the removal or its count
  fell. [EVIDENCE.md](EVIDENCE.md#e2-synthea-gap-detection-ablation) (E2) holds the reading.
- **What `first-flag` does**: it reads `synthea`'s population and releases, and for each E2
  ablation at fraction 1, removes its cases one at a time in its seed's draw order and records, per
  detector and truth item, the removed count at which the item is first listed and the cell's own
  removed count then; frequency-aware can unlist, so its reversals are counted. The draw order is
  the one a fraction ablation takes its first cases from (`bko.datasets.ablation.removal_order`),
  so an item a per-subgroup, population-level or count row lists first at removed count `r` is a
  gap after the fraction ablation removing `ceil(f * n)` of the `n` cases (the product taken
  exactly, `bko.datasets.ablation.AblationSpec`) exactly when `r` is at most that count. The truth items are the full removal's truth, as `synthea` lists it. stdout
  gets a CSV with `synthea`'s columns up to `target`, then `cases` (the ablation's cases in the
  item's cell), `truth_from` (the removed count from which the item is in the truth),
  `first_flag` (the removed count at which the detector first lists it, 0 when it is listed before
  any removal, blank when never), `cell_removed` (the cell's removed cases at `first_flag`, blank
  when never), `reversals` (how often the detector stops listing it as cases go) and
  `flagged_at_all`.
- **What it logs**: stderr gets the `versions` line (commit, dirty tree and library versions), the
  data root and the pandas and scipy versions, each input's identity (the split or the population,
  the parser and each release), each ontology fragment's size, how many records show each mapped
  concept, each cell's record count, held-concept count and assumed concepts, each concept the
  frequency-aware definition flags with its counts and posterior mass, each fused BKB's size and
  name, for `uci` each definition's gap and cell counts and the count-based cells with the
  threshold, for `synthea` each ablation, each definition's gap counts before and after the removal
  and its listed rows, and each detector's recovery (truth, eligible, recovered, outside the truth,
  new outside the truth, and vanished) and its sensitivity and specificity (`n/a` with no
  denominator), with a warning when per-subgroup or the count-based detector puts a gap new outside
  the truth or loses one, which removing whole patients cannot cause, and last the row count and
  sha256 of stdout.
- **What the sweep logs**: `synthea` and `uci` also read each grounding count and level of
  `bko.gaps.sweep` ([D1](EVIDENCE.md#d1-graded-grounding-and-the-frequency-aware-level)), to
  stderr only, so stdout is the same bytes as without it. A setting is per-subgroup or
  population-level at each count of `GROUNDING_COUNTS` and the default level, or frequency-aware
  at each level of `CREDIBILITY_LEVELS` and count 1. After each ablation's `detection` lines,
  `synthea` logs one `sweep ablation=<name> detector=<d> grounding=<k> credibility=<level>
  sensitivity=<s> specificity=<s>` line per setting, read as the `detection` lines are. After the
  last ablation it logs one `chosen level=<level> ablation=<name> floor=<s> sensitivity=<s>
  specificity=<s> tied=<levels>` line, the outcome of the rule ADR
  `research/frequency-aware-gap-score` states. `floor` is the count-based detector's sensitivity
  on `cell-50`; `sensitivity` and `specificity` are the chosen level's; `tied` lists every
  qualifying level sharing its specificity. `level=none` says no level was chosen, and the other
  fields say why: `ablation=none` when the run has no `cell-50`; `ablation=cell-50 floor=n/a` when
  the count-based detector has no eligible cell there; a floor with `level=none` when no level
  reaches it. After the definitions' `bko` lines, `uci` logs one `sweep dataset=<name>
  detector=<d> grounding=<k> credibility=<level> gaps=<n> cells=<c>` line per setting, `<n>`
  being the setting's gap count and `<c>` the cells holding them.
- **What `first-flag` logs**: the `versions` line, the data root, versions, input identities and
  observed concepts as `synthea` logs them, then a `first-flag skip ablation=<name> fraction=<f>`
  line for each partial ablation it passes over; for each whole ablation a header, `first-flag
  ablation=<name> cases=<n> truth per-subgroup=<k> population-level=<k> frequency-aware=<k>
  count=<k>`, giving its truth item counts, so an ablation with no truth still logs; then one line
  per detector and truth cell, `first-flag ablation=<name> detector=<d> cell=<key> cases=<n>
  items=<k> flagged=<k> first=<r> last=<r>`, `first` and `last` being the earliest and latest first
  flag among the cell's items and `n/a` when none is flagged; and last the `gaps rows=<k>
  sha256=<digest>` line. Each field is `key=value`, split on the first `=`. The scan's per-state
  fusions log no `fused` line: it holds `bko.bkb.fusion`'s logger at WARNING while it runs and
  restores it after, so `synthea`, `uci` and the fusion reproduction keep theirs.
- **Exit codes**: 0 the report is written; 1 an unexpected error, with its traceback; 2 the
  command could not run, including a mapped concept its release lacks and a BKB the fragments
  cannot ground, fuse or query (`bko.bkb.errors.BKBError`); 3 the data or a release contradicts
  its pin or schema, including a release naming an `is_a` parent it lacks and a malformed BKB
  document (`DocumentFormatError`). Each refusal is logged, with no traceback.
- **Reproducibility**: the same bytes on every run over the same archive, population and releases.
- **The gate does not run it.** `make check` runs it over an archive, a population and releases
  built in `tmp_path` (`tests/test_gaps_cli.py`, `tests/test_gaps_support2.py`,
  `tests/test_gaps_acs.py`, `tests/test_gaps_nhanes.py`, `tests/test_gaps_nhamcs.py`,
  `tests/test_gaps_physionet2012.py`).

**`contradictions`** lists where a mapped frame's data contradicts its ontology, in the kinds ADR
`research/data-contradiction-handling` defines (Decision 1): `narrow-without-broad`,
`child-share-above-parent`, `empty-cell`, `all-show` and `code-overlap`. One command, inside
`.venv`:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/contradictions.csv && uv run --no-sync python -m bko.evaluate contradictions [--dataset NAME]... > runs/<run-id>/contradictions.csv 2> runs/<run-id>/contradictions.log </dev/null

- **Which frames**: UCI Heart Disease's, Synthea's and Diabetes 130's, in that order, with no
  `--dataset`. A repeatable `--dataset NAME` reads the named frames instead, in the order given:
  `uci`, `synthea`, `diabetes` or a catalogued dataset, `support2`, `acs`, `nhanes`, `nhamcs` or
  `physionet2012`; a name given twice is read once, at its first place. `--dataset` and `--rows`
  exclude each other, and `--dataset` takes no run.
- **What it needs**: for the default frames, the pinned Synthea population in the data root,
  which `make synthea` generates and the command never does. It fetches UCI Heart Disease,
  Diabetes 130, a named catalogued dataset's files and the pinned releases its rows name (MONDO
  and HPO, and OccO for `acs`, converted with ROBOT) into the data root if absent.
- **What it does**: it reads each frame's training split at holdout seed 0 with no removal and
  examines every cell. Each CSV row is one finding, so a cell appears only where it holds one, in
  the columns `kind,frame,cell,concept,parent,records,shown,with_parent,without_parent,`
  `child_share,parent_share,field,codes`. It makes no API call.
- **What it logs**: stderr gets the `versions` line (commit, dirty tree and library versions),
  `census data_dir=<dir>`, each dataset's `data` line and each release's identity. Then, per
  frame in frame order: one `census data` line with the training records' count and digest; one
  `census rows` line with the field rows' count, their digest and the `is_a` pair count; one
  `census cells` line with the grid's cell count, the cells holding a record and the concepts
  that follow the label, which `all-show` leaves out; one `census pair frame=<f> concept=<c>
  parent=<p> shown=<n> without=<n> cells=<k>` line per `is_a` pair, counting over the whole
  split the records that show the narrow concept, those of them without the broad one, and the
  cells holding the pair's `narrow-without-broad` finding; and one `census kind` line per kind
  with its count, 0 included. Last comes one `census rows` line with stdout's row count and
  sha256.
- **Reproducibility**: the same bytes on every run over the same data and releases.
- **The gate does not run it.** `make check` runs it over records, rows and releases built in
  memory and in `tmp_path` (`tests/test_contradictions_census.py`,
  `tests/test_contradictions_dataset.py`).

**`contradictions <run> --rows`** reads the second basis of ADR
`research/data-contradiction-handling` (Decision 2): each fill's added rows in a run's `rows`
stores, gap and replica fills alike. `<run>` is the named run whose slices' stores are read
(`e6-thin`); `rows` runs first, one store per slice, and `--rows` repeats once per store, as for
`privacy` ([Evaluation](#evaluation)). It makes no API call:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/contradictions.csv && test ! -e runs/<run-id>/contradictions.log && uv run --no-sync python -m bko.evaluate contradictions <run> [--runs-dir DIR] $(printf -- '--rows %s ' runs/<store-id>-rows/rows.jsonl <more>...) > runs/<run-id>/contradictions.csv 2> runs/<run-id>/contradictions.log </dev/null

- **What it needs**: the stores and their folders, which sit directly under the runs root
  (`--runs-dir DIR`, else `runs/` beside the code), and the dataset, which it fetches into the
  data root if absent, with the pinned releases of its mapped concepts.
- **What it does**: per holdout and fill, in the columns `holdout,arm,rows,kind,count`, one row
  per kind: the five kinds counted in the fill's added rows, as the ADR defines each on this
  basis, then `drug-without-med`, `change-without-med` and `dose-without-change`, the rows breaking
  each Diabetes coherence rule of `bko.evaluate.perturb.rule_breaks`, blank off the Diabetes
  columns. `empty-cell` counts the cells of the fill's request, rebuilt from the holdout's
  training split, that hold no added row: the gap request for a gap fill, the replica request
  for a replica fill. It is blank for a fill that answers no request: `none`, `real`,
  `prior-shift` and the weighting arms. A fill that answers one and has no rows counts every
  cell of its request as empty and 0 on every other kind, its rules still blank off the
  Diabetes columns.
- **What it logs**: stderr gets the `versions` line, the run's dataset's `data` line and each
  release's identity, then `added run=%s stores=%d holdouts=%d root=%s`, one `added store=%s
  run=%s folder=%s log_sha256=%s holdouts=%s sha256=%s` line per store, the sha256 of the bytes
  read, one `added holdout=%d arm=%s rows=0 asked=%d` line per fill that answers a request with
  no row, one `added holdout=%d done=%d/%d seconds=%.1f` line per holdout and, last, `added
  rows=%d sha256=%s stores=%d holdouts=%d` with stdout's row count and sha256.
- **Exit codes**: 0 the CSV is written; 1 an unexpected error, with its traceback; 2 for a run
  or `--runs-dir` without `--rows`, an unknown run, a mapped concept with no pinned release, or
  a holdout whose split flags no cell for a gap fill that answers the gap request; 3 when a
  release contradicts its pin, or a store or its folder fails the identity checks `privacy`
  applies to it, a fill whose rows do not hash to its `arm` line's sha256 included.
- **The gate does not run it.** `make check` runs it over run folders and stores built in
  `tmp_path`, UCI- and Diabetes-shaped (`tests/test_contradictions_added.py`,
  `tests/test_contradictions_added_diabetes.py`).

**Share-based quota counts.** `bko.gaps.quota` holds the counts a run under a share-based
`Experiment.quota_rule` asks each flagged cell for, `coverage`, `expected_size` and
`recommended`; ADR [`research/share-based-quota`](adr/research/share-based-quota.md), Decision,
defines each. It has no command of its own: `score` reads it through the gap request, and every
reader that rebuilds that request reads it the same way.

## Generation

`bko.generate` ([README](../src/bko/generate/README.md)) asks the Anthropic Messages API for
candidate records that fill a gap request, E1's count-based one or the completeness request (ADR
[`research/record-generation-design`](adr/research/record-generation-design.md)), or draws them
from the fused BKB with no call (the direct sampler, below). The paid command, inside `.venv`,
with `make setup`'s `llm` extra and an Anthropic credential the SDK can resolve:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/calls.jsonl && test ! -e runs/<run-id>/candidates.jsonl && test ! -e runs/<run-id>/candidates.log && uv run --no-sync python -m bko.generate <definition> [--guidance <level>] [--request <e1|completeness>] --log runs/<run-id>/calls.jsonl [--data-dir <dir>] > runs/<run-id>/candidates.jsonl 2> runs/<run-id>/candidates.log

- **What it does**: it fetches UCI Heart Disease and the releases the field mappings name into the
  data root (`--data-dir`, default `data/`) if absent, splits the records as [Data](#data) says,
  and reads the training split's gaps under `<definition>`: `per-subgroup`, `population-level` or
  `frequency-aware`. With `--request e1`, the default, it asks each cell holding fewer training
  records than `UNDERREPRESENTED_BELOW` for its shortfall; with `--request completeness` it asks
  each cell holding a per-subgroup gap for `UNDERREPRESENTED_BELOW` rows, and exits 2 for any
  `<definition>` but `per-subgroup` ([Gap closure](#gap-closure)). It asks in calls of at most
  `Settings.per_call` records and writes a header line to stdout first (schema 2: the request,
  generator `messages` with a null seed, definition, guidance level, the UCI archive's and
  releases' pinned sha256, the split's fingerprint and each call's cell and records asked; ADR
  [`research/gap-closure-design`](adr/research/gap-closure-design.md)), then one JSON line per
  asked record: `call`, `cell`, `position`, and the decoded `candidate` or the `undecoded` reason.
  Each call's lines are written and flushed as it returns, so a failed or interrupted run's stdout
  holds every batch that returned before the failure. Candidates are unfiltered and unrepaired;
  `bko.validate` checks them.
- **What leaves the machine**: the frame's study and cohort text, each field's meaning, codes and
  unit, each numeric field's training-split minimum and maximum (age within the cell's band), the
  cell and its training record count, and the guidance of the chosen level; at `summary`, also the
  quartiles of each numeric field over the same-label records of the other cells, that reference
  group's record count and each mapped concept's share of it in percent (the `profile` block).
  Every `SETTINGS` run sends them. A `bko-skeleton` request also sends each asked record's drawn
  concept names: ontology labels drawn at the fused BKB's aggregate shares, never a record's values.
  Of the BKO arms, only `bko-direct-perturbed` sends record values: the 19 unmapped fields of rows
  of open Diabetes 130 data, as the Diabetes 130 entry below says (ADR
  [`research/identifier-perturbation`](adr/research/identifier-perturbation.md)); no other BKO arm
  sends a record. The CLLM arms (`cllm`, `cllm-uncurated`; ADR
  [`research/cllm-arm-design`](adr/research/cllm-arm-design.md)) send, on each call, the system
  text, the cell, its training record count and each numeric field's training-split bounds, then up
  to `bko.evaluate.cllm.SHOW` training records whole, every frame column: the cell's own first, then
  its sex-and-label group's, then its label group's. They send no finding, profile or summary. Per
  frame:
  - UCI (every named run with a paid arm whose dataset is UCI, `bko.evaluate.named_runs.RUNS`,
    and this command):
    every bound and quartile is a real training value of a de-identified patient record, and a
    small split or reference group exposes them together as a record; a CLLM call sends the
    records themselves, under the archive's CC BY 4.0 license. Every paid arm and this command
    refuse data held under an agreement that records no permission to send it ([Data](#data), Agreements); check a
    DUA's terms before recording one ([`RESEARCH.md`](RESEARCH.md#6-datasets)).
  - Synthea (every named run with a paid arm whose dataset is Synthea): through the BKO arms, the
    study and cohort text, the codebook with each indicator's label, age's training-split bounds
    within the cell's band, the cell and its training record count, and the gap findings with their ontology chains;
    at `summary`, also the reference group's (the training records outside the cell that share its
    label) record count, each mapped concept's share of it and its age quartiles; through the CLLM
    arms, whole training records too. Its rows are generated by Synthea from a pinned seed, so no
    bound, quartile, share or record describes a person.
  - Diabetes 130 (every named run with a paid arm whose dataset is Diabetes 130, the perturbation
    pilots included): the same
    kinds of quantity as UCI's, over the frame's fields ([Data](#data), The Diabetes frame); every
    bound and quartile is a real training value of a de-identified encounter, and a CLLM call sends
    frame records whole, under the archive's CC BY 4.0 license. The archive's identifiers,
    `encounter_id` and `patient_nbr`, never enter the frame, so no call sends them; nor does any
    call send a column the frame leaves out (`weight`, `payer_code`, `medical_specialty`, the
    admission and discharge ids, the raw `diag_1` to `diag_3` codes). A `bko-direct-perturbed`
    rewrite call sends its system prompt (`bko.generate.rewrite.REWRITE_SYSTEM`), each of the 19
    unmapped fields' meaning with its codes or unit (`bko.evaluate.perturb.DIABETES_UNMAPPED`),
    each count field's training-split minimum and maximum, an accepted aggregate disclosure, the
    call's value counts per field (`bko.evaluate.quotas`), each a count over a value or range of the
    cell's label-group training rows (a range's ends are real training values, and each range holds
    at least 1% of the label group's rows, so a label group under 100 rows can expose a single
    row's value), and each row's values of those 19 fields with its `row` index; no mapped field,
    `age`, `sex` or label. Each row is one of the holdout's `bko-direct` rows, a mosaic of
    de-identified encounters' values.

  A run whose settings name a model in `bko.generate.weights.LOCAL_MODELS` sends each call to the
  llama.cpp server on `127.0.0.1` instead (Local models below), so nothing leaves the machine.
- **Local models**: an on-device model answers a run's calls through `llama-server` (Homebrew
  `llama.cpp`) on `http://127.0.0.1:8081` (ADR
  [`research/local-generator-backend`](adr/research/local-generator-backend.md)). Fetch and verify
  its pinned weights into `data/models/`, serve it with its output in the run's folder
  (`server.log`, [Runs](#runs)), probe it, and stop the server when the run ends. A `-thinking`
  model id, `gemma-4-26b-a4b-q8_0-thinking` or `nemotron-3-nano-30b-a3b-q8_0-thinking`, is its
  base model with thinking on, served from the base's weights file:

      uv run --no-sync python -m bko.generate.local fetch <model> [--data-dir <dir>]
      mkdir -p runs/<run-id> && test ! -e runs/<run-id>/server.log && eval "$(uv run --no-sync python -m bko.generate.local serve-command <model> [--data-dir <dir>])" > runs/<run-id>/server.log 2>&1 &
      uv run --no-sync python -m bko.generate.local probe <model>
      pkill -f "llama-server .*--alias <model>"

  `fetch` verifies a present file's size and sha256, or streams the file into `<file>.part`,
  logging its progress, and renames it once it matches the pin; it removes a download that does
  not match and exits 2 on a mismatch or a failed download; a `-thinking` id's `fetch` verifies or
  fetches its base's file. `serve-command` prints the server's argv, thinking off, or for a
  `-thinking` id the base's file under the variant's alias with
  `--jinja --reasoning on --reasoning-format deepseek`; give it the `--data-dir` `fetch` used.
  `probe` waits for the server's `/health` while it loads, sends one call for four records of
  `<55/female/1`, and logs each decoded record, the seconds it took and the reply's reasoning
  length as `reasoning_chars=`; it exits 2 if the server does not ready, serves another model
  (a base id's server for its `-thinking` id, or the reverse, included), fails the pre-flight
  below, or nothing decodes. Its own no-reasoning `WARNING` fires only when the pre-flight
  reasoned and the full call did not. Stop the server with `pkill` whether or not the probe
  passes. One model is served at a time; a local model bills nothing. A run's first call refuses a
  server serving another model. For a `-thinking` id it then sends one pre-flight call, the
  probe's at `max_tokens` 64, and refuses a server whose reply carries no `reasoning_content`,
  such as one started with `--reasoning-format none`: the run fails before its first call with
  `the local server for <model> does not reason: build=<build> path=<path> sha256=<sha256> preflight_reasoning_chars=0 stop_reason=<reason> content_chars=<n>`,
  each later call of the run fails with the same message without another pre-flight, and `probe`
  exits 2 with that `ERROR` line. A pre-flight that fails retryably, such as a status of 500 or
  above, leaves the next call to check again. A checked server logs the `local server` line (the
  model, `thinking=` from the model id's pin rather than read from the server, the served file,
  the server's build and the file's pinned sha256, and for a `-thinking` id
  `preflight_reasoning_chars=`). Each live call of a `-thinking` id logs one `reasoning` line from
  `bko.generate.reasoning`:
  `reasoning request_id=<id> stop_reason=<reason> reasoning_chars=<n> reasoning_tokens=<n> content_chars=<n> output_tokens=<n>`,
  its request id and stop reason with whitespace and unprintable characters written as `\u`
  escapes and cut at `bko.generate.reasoning.FIELD_CHARS`. Its reasoning tokens are the server's
  `/tokenize` re-tokenization of `reasoning_content`, not a count kept while generating; a
  `/tokenize` that fails or waits past `TOKENIZE_SECONDS` logs them as `unknown`, and the run's
  first such failure logs one `tokenize failed: <status or exception>` `WARNING`. The line is a
  `WARNING` ending ` reason=truncated_while_reasoning` when the reply stopped at `max_tokens` with
  reasoning and no content. A replay logs no `reasoning` line, as it logs no `local server` line.
  A local run keeps at most four calls in flight (`bko.generate.local.SLOTS`), logging
  `local in_flight=` when `--in-flight` asked for more.
- **Option order**: the definition or `direct` comes first and its options follow it, as in
  `per-subgroup --guidance summary --log ...`; an option before it is refused, exit 2.
- **Guidance levels**: `cell`, `coherent`, `findings`, `ontology`, `profile` and
  `summary` (the default), each adding one component to the one before it; the ADR defines each.
- **What it logs**: the call log at `--log` gets a `run` line naming the frame under `frame`, a
  `request` line before each call and a `reply` or `error` line after it, and an `end` line with
  the run's status, the replayed log's
  sha256 (or null), the resumed log's sha256 and answered calls under `resume` (or null; generate
  always writes null), `rehearsal` (false: generate never rehearses), `stop` (null: generate has no
  stop rule), `retryable` (whether the API caused a failure: a connection error, a timeout, a 408,
  409 or 429, or a status of 500 or above), `rate_limited` (whether any retry met a 429) and totals,
  `live_usd` among them, each call's whole request and response and its USD spend included. An
  interrupt or any other failure between a `request` line and its `reply` line writes the `error`
  line. A call cancelled by an operator's interrupt records `KeyboardInterrupt`, in `score` as
  here; a call that fails records its own error, and the calls beside it land rather than being
  cancelled. stderr gets the split line, each ontology fragment's size, the request's name, cells
  and
  rows, a `quota` line per asked cell, a `log` line (`log path=%s schema=%d model=%s effort=%s
  level=%s per_call=%d max_tokens=%d replay=%s resume=%s rehearsal=%s in_flight=%d`: the replayed
  log's sha256 or `none`, `resume=none`, `rehearsal=false` and `in_flight=1`), the
  plan (calls, level, and the most the output tokens can cost), a `send call=%d cell=%s asked=%d`
  line before each call (with ` shown=%d`, the records it shows, for a CLLM arm's call) and a
  `generate call=%d cell=%s level=%s asked=%d decoded=%d undecoded=%d stop_reason=%s
  input_tokens=%d output_tokens=%d usd=%.6f request_id=%s elapsed_s=%.3f in_flight=%d` line after
  it, ending with the call's seconds from send to landing and the calls in flight when it landed,
  itself included (or an ERROR `failed` line naming the call and cell), the run's totals
  and spend (`usd` and `live_usd`) with the replayed and resumed logs' sha256 or `none`,
  `rehearsal=false` and the log's path, and last the count of stdout's record lines (the header
  excluded) and the sha256 of stdout, its header included. Its WARNING lines: `served`, a call
  served by a model other than the settings name; `short`, a call that wrote placeholders, with
  their reason; `unreached`, a gap statement no label-compatible mapping row reaches, left out of
  the prompt; `noreference`, a cell with no reference record, so no profile or summary; and the UCI
  loader's line naming the records it dropped for a missing value. The SDK client retries a failed
  attempt up to `bko.generate.api.MAX_RETRIES` times, backing off between tries, and logs each retry
  at INFO between the call's `send` line and its result; each such record, on logger
  `anthropic._base_client`, ends ` call=%d` while a call is in flight. Each `reply` and `error`
  line records the call's `attempts`, and the `end` line counts the calls `retried`. A retried attempt can be billed,
  so logged spend is a lower bound when `retried` is above 0.
- **Exit codes**: 0 the candidates are written; 1 an unexpected error, with its traceback; 2 the
  command could not run, including a missing credential, a call that got no response, a
  completeness request no cell holds a gap for, `--request completeness` with another
  `<definition>`, and a `--log` path that exists (checked before any download or call); 3 the data
  or a release contradicts its pin or schema.
- **Reproducibility**: none for a `<definition>` run. Each call spends money and a rerun's candidates differ; the call log
  is the record, so the command never overwrites one.
- **The gate does not run it.** `make check` drives it through a fake `Send` over an archive and
  releases built in `tmp_path` (`tests/test_generate_cli.py`); no gate test reaches the API.
- **The release test**: `tests/test_generate_pinned_releases.py` builds every cell's guidance under
  each definition over the real mapping table, UCI snapshot and pinned releases in the data root,
  and renders it at every level. It never fetches and never calls the API; run it with
  `--releases` after changing guidance or the prompt, once the command or `make cells` and
  `python -m bko.ontology check` have fetched the data:

      uv run --no-sync pytest tests/test_generate_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

- **The direct sampler**: `direct` makes no API call and needs no credential. It fills the same
  request from the fused BKB's concept shares (ADR
  [`research/bkb-direct-generation`](adr/research/bkb-direct-generation.md)), seeded by the
  required `--seed`, a non-negative integer; it takes no `--guidance`, and `--log` names its draw
  log, which must not exist and is created readable by its owner only:

      mkdir -p runs/<run-id> && test ! -e runs/<run-id>/draws.jsonl && test ! -e runs/<run-id>/candidates.jsonl && test ! -e runs/<run-id>/candidates.log && uv run --no-sync python -m bko.generate direct [--request <e1|completeness>] --seed <n> --log runs/<run-id>/draws.jsonl [--data-dir <dir>] > runs/<run-id>/candidates.jsonl 2> runs/<run-id>/candidates.log </dev/null

  Its stdout is a candidates file whose header names generator `direct` and the seed, with a null
  definition and guidance, and whose call plan is the one a Messages run of the same request
  writes, so `bko.validate` and `bko.fusion` read it unchanged; every line holds a `candidate`.
  The draw log holds one `draw` line per record: `call`, `cell`, `position`, the `concepts` drawn
  present, their `closed` closure under `is_a`, and the `fallback` fields read from the label
  group's records. stderr gets the split line, each ontology fragment's size, the request and
  `quota` lines, a `direct` line per cell (rows, the cell's records, its label group's records), a
  `fallback` line per cell and field read from the label group, a `draws` line (seed, rows, the
  draw log's path), and the stdout `candidates` line. It exits 2, writing no line, when no training
  record matches a draw (a `cannot draw` message) or the draw log exists (checked before any
  download). The same seed and inputs write the same stdout and draw log, byte for byte. The gate
  drives it over an archive and releases built in `tmp_path` (`tests/test_generate_cli.py`).
- **The live test**: `tests/test_generate_live.py` makes paid calls and checks each reply's keys,
  types and codes: one call for 2 records over the UCI frame on each of two models, and 10 calls for
  `SETTINGS.per_call` records over the Diabetes frame, from invented guidance and no `data/` read.
  It runs only under `--live`. Run it deliberately, after a change to
  the schema or the call, into a `--basetemp` that does not exist yet, since pytest wipes it:

      test ! -e runs/<run-id>/pytest && mkdir -p runs/<run-id> && uv run --no-sync pytest tests/test_generate_live.py --live --basetemp=runs/<run-id>/pytest -q -p no:cacheprovider --no-cov -o log_cli=true --log-cli-level=INFO

## Gap closure

`bko.fusion` ([README](../src/bko/fusion/README.md)) joins the generated records that pass
validation to the training records of the cell each was asked for and lists the gaps again, the
completeness criterion's reading of "after fusion" (ADR
[`research/gap-closure-design`](adr/research/gap-closure-design.md) holds which records join).
Two commands, inside `.venv`; the first spends money and needs an Anthropic credential, the
second spends nothing:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/calls.jsonl && test ! -e runs/<run-id>/candidates.jsonl && test ! -e runs/<run-id>/candidates.log && uv run --no-sync python -m bko.generate per-subgroup --guidance summary --request completeness --log runs/<run-id>/calls.jsonl > runs/<run-id>/candidates.jsonl 2> runs/<run-id>/candidates.log
    mkdir -p runs/<run-id>-closure && test ! -e runs/<run-id>-closure/closure.csv && test ! -e runs/<run-id>-closure/closure.log && uv run --no-sync python -m bko.fusion runs/<run-id>/candidates.jsonl [--data-dir <dir>] > runs/<run-id>-closure/closure.csv 2> runs/<run-id>-closure/closure.log

The `test ! -e` guards matter: the shell empties `candidates.jsonl` before `bko.generate` can
refuse an existing log, and `candidates.jsonl` is the only copy of the candidates the second
command reads.

The direct sampler's `--request completeness` file ([Generation](#generation)) replaces the first
command for free; the second reads it unchanged, and its `header` line reads `definition=None
guidance=None`.

- **What it does**: it reads the candidates file, fetches UCI Heart Disease and the releases the
  field mappings name into the data root if absent, splits the records as [Data](#data) says,
  and lists the training split's gaps under every definition. It refuses, in this order: a file
  that is not a candidates file, through `bko.generate.candidates.read` (no header, or a malformed
  record line); another request; another UCI archive; another split; other releases; a plan
  whose per-cell sums differ from the completeness request, naming the cells that differ; a line
  of an unplanned call; a call whose lines name another cell than its plan; and a planned call
  that wrote fewer of its positions than it asked. It validates each requested cell's lines with `bko.validate`, an
  `undecoded` line counting as generated and rejected, joins the accepted records as the ADR
  says, and lists the gaps again over the training and joined records.
- **What it writes**: one CSV on stdout, the same bytes on every run over the same candidates
  file, archive and releases, and nothing unless the whole report is built. A row per
  (definition, cell, statement) pair listed before or after the join: `definition`, the
  statement columns of [Gap detection](#gap-detection), `age_band`, `sex`, `target`, `before` and
  `after`. A closed pair reads `True,False`, an open one `True,True` and a new one `False,True`.
  It prints no verdict; [EVIDENCE.md](EVIDENCE.md#c1-gap-closure-after-fusion) reads the counts
  against [RESEARCH.md](RESEARCH.md#4-success-criteria).
- **What it logs**: stderr gets the data root and library versions, the candidates file's path,
  record-line count (the header excluded) and sha256 (the `candidates` digest `bko.generate`
  logged, when the file is its stdout), a `header` line (`request=`, `definition=`, `guidance=`, `fingerprint=`), the split
  line, each ontology fragment's size, a `request` line per requested cell, a
  `validate` line naming each cell before `bko.validate`'s counts for it, a `join` line per
  requested cell (lines, passed, off-cell, past its rows, joined, and the concepts the join newly
  gives the cell), a `closure` line per definition (before, after, closed, open, new and the
  share), and last the row count and sha256 of stdout, written once the report is. Its WARNING
  lines: `off_cell`, accepted records dropped for naming another cell; `past_quota`, accepted
  on-cell records dropped past the cell's rows; and `unjoined`, a requested cell no record joined.
- **Exit codes**: 0 the report is written; 1 an unexpected error, with its traceback; 2 the
  command could not run, including a missing candidates file and a training split no cell holds
  a per-subgroup gap in, and a BKB the fragments cannot ground, fuse or query
  (`bko.bkb.errors.BKBError`); 3 the candidates file is malformed or does not answer the
  completeness request (a header for another run, an unplanned or misplaced call, or a short
  call), the data or a release contradicts its pin or schema, or a BKB document is malformed
  (`DocumentFormatError`). Each refusal is logged, with no traceback.
- **Headerless files**: ADR [`research/gap-closure-design`](adr/research/gap-closure-design.md)
  (Consequences) says what becomes of a candidates file with no header line.
- **Reproducibility**: the same bytes on every run over the same inputs. The Messages API takes
  no seed, so the candidates file is the paid run's record, not a rerun.
- **The gate does not run it.** `make check` drives it over an archive, releases and candidates
  built in `tmp_path`; no gate test reaches the API.
- **The release test**: `tests/test_fusion_pinned_releases.py` joins records that show every
  mapped concept to the real training split, the control that closes every per-subgroup gap. Run
  it with `--releases` once the data root holds the snapshot and releases:

      uv run --no-sync pytest tests/test_fusion_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

## Language-model classifier

`bko.slm` ([README](../src/bko/slm/README.md)) fine-tunes a LoRA adapter on the pinned OLMo 3 7B
base model as a classifier of one arm's training set, and scores each test record's P(yes) (ADR
[`research/slm-comparison-design`](adr/research/slm-comparison-design.md) holds the design, ADR
[`tooling/slm-extra`](adr/tooling/slm-extra.md) the stack and the base model). It makes no API
call, and the gate does not run it: the gate's tests use a tiny randomly initialised model.
Inside `.venv`:

    uv run --no-sync python -m bko.slm fetch
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/slm.log && caffeinate -i -s uv run --no-sync python -m bko.slm train <run> --folder runs/<folder-id> --rows runs/<folder-id>-rows/rows.jsonl --holdout <h> --arm <arm> 2> runs/<run-id>/slm.log </dev/null
    mkdir -p runs/<run-id>-score && test ! -e runs/<run-id>-score/slm-probabilities.csv && test ! -e runs/<run-id>-score/slm.log && caffeinate -i -s uv run --no-sync python -m bko.slm score <run> --folder runs/<folder-id> --rows runs/<folder-id>-rows/rows.jsonl --holdout <h> (--arm <arm> | --base) > runs/<run-id>-score/slm-probabilities.csv 2> runs/<run-id>-score/slm.log </dev/null

- **`fetch`** downloads each pinned file of `bko.slm.weights.OLMO_3_7B` at its revision into
  `data/models/olmo-3-1025-7b/<revision[:16]>/`, through a `.part` file renamed only once its
  size and sha256 match, and verifies a file already present.
- **`train`** reads the folder and its `rows` store, bound as `rows` binds them, and prepares the
  holdout as the run did. The training set is the prepared training split for `none`, the
  un-ablated split for `real`, and the split plus the arm's stored fill for any other arm; a run
  whose experiment decontaminates is refused, since the test split is scored whole. It verifies
  every weight file's sha256 before the load, loads the model in BF16 on MPS, and writes the
  adapter to `data/models/slm-adapters/<folder-id>/<h>-<arm>/`, refusing a path that exists:
  `adapter_config.json`, `adapter_model.safetensors` and `adapter_meta.json`, which names the
  run, folder, store sha256, holdout, arm and base revision. An adapter stays on the machine. Its log holds the `versions` line, a `step` line per
  optimizer step, a `train seconds=` line and the `adapter` line with the adapter's sha256.
- **`score`** writes `record_id,probability` for every test record of the holdout, from the
  adapter of `--arm` or the base model with `--base`, and logs each `utility` measure with the
  calibration measures, then `probabilities rows=<n> sha256=<sha>`. It refuses an adapter
  directory holding any other file, or whose meta differs from its own arguments.
- **Power**: a fine-tune runs for hours. Check that `pmset -g batt | head -1` names AC Power
  first; `caffeinate -i -s` keeps the machine awake on AC power.
- **Exit codes**: 0 done; 1 an unexpected error; 2 the command could not run (no MPS, an absent
  model directory, a failed download, an adapter path that exists or is absent, any other
  `OSError`); 3 a file or a record contradicts its pin or contract (a weight file's sha256, a
  stored fill's identity or its absence, an adapter directory's files or meta).

## Runs

A run of record keeps its artifacts in its own folder, `runs/<run-id>/` at the repository root,
gitignored like `data/` and present only on the machine that ran it. `<run-id>` is the run's id in
[EVIDENCE.md](EVIDENCE.md), such as `e1-3`; a run that checks or extends another takes that run's
id and a suffix, such as `e1-3-replay` or `c1-2-closure`. A fill made in the Streamlit app is not a
run of record; its folder is named `app-<UTC time>-<arm>`. `bko.datasets.run_folders.RUN_ID`
matches both forms. Each artifact takes a fixed role name, so
a reader finds it without knowing the command that wrote it. `bko.app.runs.ROLES` is the list's
home; this table mirrors it, and `tests/test_run_folders.py` holds the two equal:

| Role | What writes it |
|---|---|
| `calls.jsonl` | the call log of a run that calls the Messages API |
| `scores.csv` | `python -m bko.evaluate score`'s stdout |
| `scores.log` | its stderr |
| `server.log` | the local model server's output, written by the operator's serve command |
| `holdout-candidates.jsonl` | every candidate a run's BKO arms generated, per holdout |
| `rows.jsonl` | `python -m bko.evaluate rows`'s store of every fill, per holdout |
| `launch.log` | `python -m bko.evaluate launch`'s record of the power reading and command |
| `rehearsal` | `python -m bko.evaluate launch --rehearse`'s directory of the run's roles |
| `criteria.csv` | `python -m bko.evaluate compare`'s stdout |
| `criteria.log` | its stderr |
| `likelihood.csv` | `python -m bko.evaluate likelihood`'s stdout |
| `likelihood.log` | its stderr |
| `privacy.csv` | `python -m bko.evaluate privacy`'s stdout |
| `privacy.log` | its stderr |
| `truth.csv` | `python -m bko.evaluate truth`'s stdout |
| `truth.log` | its stderr |
| `representativeness.csv` | `python -m bko.evaluate representativeness`'s stdout |
| `representativeness.log` | its stderr |
| `syntheval.csv` | `python -m bko.evaluate syntheval`'s stdout |
| `syntheval.log` | its stderr |
| `trace.jsonl` | `python -m bko.evaluate trace`'s stdout |
| `trace.log` | its stderr |
| `attribution.csv` | `python -m bko.evaluate attribution`'s stdout |
| `attribution.log` | its stderr |
| `candidates.jsonl` | `python -m bko.generate`'s stdout |
| `candidates.log` | its stderr |
| `draws.jsonl` | `python -m bko.generate direct`'s draw log |
| `closure.csv` | `python -m bko.fusion`'s stdout |
| `closure.log` | its stderr |
| `gaps-uci.csv` | `python -m bko.gaps uci`'s stdout |
| `gaps-uci.log` | its stderr |
| `gaps-synthea.csv` | `python -m bko.gaps synthea`'s stdout |
| `gaps-synthea.log` | its stderr |
| `gaps-support2.csv` | `python -m bko.gaps support2`'s stdout |
| `gaps-support2.log` | its stderr |
| `gaps-acs.csv` | `python -m bko.gaps acs`'s stdout |
| `gaps-acs.log` | its stderr |
| `gaps-nhanes.csv` | `python -m bko.gaps nhanes`'s stdout |
| `gaps-nhanes.log` | its stderr |
| `gaps-nhamcs.csv` | `python -m bko.gaps nhamcs`'s stdout |
| `gaps-nhamcs.log` | its stderr |
| `gaps-physionet2012.csv` | `python -m bko.gaps physionet2012`'s stdout |
| `gaps-physionet2012.log` | its stderr |
| `first-flag.csv` | `python -m bko.gaps first-flag`'s stdout |
| `first-flag.log` | its stderr |
| `contradictions.csv` | `python -m bko.evaluate contradictions`'s stdout |
| `contradictions.log` | its stderr |
| `reproduction.csv` | `python -m bko.ontology reproduce`'s stdout |
| `reproduction.log` | its stderr |
| `hierarchy.csv` | `python -m bko.ontology elk`'s stdout |
| `hierarchy.log` | its stderr |
| `training.csv` | the augmented training set of an app fill |
| `pytest` | a `live` test's `--basetemp` directory |
| `slm-probabilities.csv` | `python -m bko.slm score`'s stdout |
| `slm.log` | `python -m bko.slm train` or `score`'s stderr |

The commands in [Evaluation](#evaluation), [Generation](#generation) and
[Gap closure](#gap-closure) write their folders. A free run of record writes its folder with one of:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-uci.csv && test ! -e runs/<run-id>/gaps-uci.log && uv run --no-sync python -m bko.gaps uci > runs/<run-id>/gaps-uci.csv 2> runs/<run-id>/gaps-uci.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-synthea.csv && test ! -e runs/<run-id>/gaps-synthea.log && uv run --no-sync python -m bko.gaps synthea > runs/<run-id>/gaps-synthea.csv 2> runs/<run-id>/gaps-synthea.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-support2.csv && test ! -e runs/<run-id>/gaps-support2.log && uv run --no-sync python -m bko.gaps support2 > runs/<run-id>/gaps-support2.csv 2> runs/<run-id>/gaps-support2.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-acs.csv && test ! -e runs/<run-id>/gaps-acs.log && uv run --no-sync python -m bko.gaps acs > runs/<run-id>/gaps-acs.csv 2> runs/<run-id>/gaps-acs.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-nhanes.csv && test ! -e runs/<run-id>/gaps-nhanes.log && uv run --no-sync python -m bko.gaps nhanes > runs/<run-id>/gaps-nhanes.csv 2> runs/<run-id>/gaps-nhanes.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-nhamcs.csv && test ! -e runs/<run-id>/gaps-nhamcs.log && uv run --no-sync python -m bko.gaps nhamcs > runs/<run-id>/gaps-nhamcs.csv 2> runs/<run-id>/gaps-nhamcs.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-physionet2012.csv && test ! -e runs/<run-id>/gaps-physionet2012.log && uv run --no-sync python -m bko.gaps physionet2012 > runs/<run-id>/gaps-physionet2012.csv 2> runs/<run-id>/gaps-physionet2012.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/first-flag.csv && test ! -e runs/<run-id>/first-flag.log && uv run --no-sync python -m bko.gaps first-flag > runs/<run-id>/first-flag.csv 2> runs/<run-id>/first-flag.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/contradictions.csv && test ! -e runs/<run-id>/contradictions.log && uv run --no-sync python -m bko.evaluate contradictions [--dataset NAME]... > runs/<run-id>/contradictions.csv 2> runs/<run-id>/contradictions.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/contradictions.csv && test ! -e runs/<run-id>/contradictions.log && uv run --no-sync python -m bko.evaluate contradictions <run> [--runs-dir DIR] $(printf -- '--rows %s ' runs/<store-id>-rows/rows.jsonl <more>...) > runs/<run-id>/contradictions.csv 2> runs/<run-id>/contradictions.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/reproduction.csv && test ! -e runs/<run-id>/reproduction.log && uv run --no-sync python -m bko.ontology reproduce > runs/<run-id>/reproduction.csv 2> runs/<run-id>/reproduction.log </dev/null

- **Runs whose rows name `scratchpad/`**: their artifacts are where the row says. The latest such
  run of each entry also has a folder whose files are hard links to those artifacts, with write
  permission removed: `e1-3`, `c1-2`, `c1-2-closure`, `g1-13`, `e2-5`, `r1-2`, `e10-37` and
  `p2-2`. A hard link is the same file under a second name, sharing its bytes and its mode, so
  the row's paths, sha256 and commands stay true; `c1-2`'s live check, a pytest directory, stays only where its row says. Never
  write into a linked folder or restore a linked file's write permission: a redirect into a
  writable link empties the register's artifact before the command runs.
- **Another machine**: a run shows there only after its folder is copied into that machine's `runs/`.

## App

`bko.app` ([README](../src/bko/app/README.md)) serves a Streamlit page over the stages, with
`make setup`'s `app` extra:

    make app

It serves `http://127.0.0.1:8501` until interrupted. The recipe's flags are the app's whole
configuration: the loopback bind, a fixed port (a second launch fails rather than moving), no
browser and no first-run prompt, no usage statistics, no file watcher, a toolbar with no deploy
button, and no rerun that aborts a running action. Launch it only through `make app`; a bare
`streamlit run` drops them.

- **What it does**: Analyze fetches UCI Heart Disease and the releases the field mappings name into
  the data root if absent, splits the records as [Data](#data) says, and lists the count-based
  cells and each definition's gaps as [Gap detection](#gap-detection) does. Fill and score fills
  E1's gap request on that split with the chosen arm, then fits every method of
  `bko.evaluate.experiment.METHOD_ARMS` beside it, CTGAN and TVAE included, and scores them all
  with `bko.evaluate.run.score_holdout`, with no replica fill, so the page shows no fidelity and no
  train-on-synthetic reading. It charts each arm's recall, overall and on the flagged subgroups,
  beside a table of its recall, precision and AUC-PR, with every measure behind an expander. It
  lists each definition's gaps before and after the added rows join. It refuses an experiment
  with a training fraction below 1 or a removal.
  Download offers the augmented training set, in the format `python -m bko.baselines` prints, and
  the split's scores, in the format `python -m bko.evaluate score` writes.
- **What it judges**: a demo fill judges nothing. It shows one split's measures; E1's verdict comes
  only from `python -m bko.evaluate compare` over E1's holdouts
  ([EVIDENCE.md](EVIDENCE.md#e1-uci-heart-disease-augmentation)), whose stored criteria table the
  Earlier runs view shows.
- **Earlier runs**: the page's second view lists the run folders under `runs/` ([Runs](#runs)) and
  shows each one's stored artifacts by role: a call log's `end` totals; E1's criteria table, the
  means over its holdouts and each BKO arm's validity counts, with E1's validity verdict on its
  candidate; candidates per call and cell; a closure table's pairs and counts; the UCI gap listing
  per detector and E2's gap readings per ablation and detector; R1's reproduction rows; a demo
  fill's scores (as means over its one holdout) and training set; a trace file's records. It
  recomputes nothing from
  `data/` and writes nothing; it derives only means, counts and E1's validity from the stored
  files, and leaves the `shared-split` fingerprint out of the means, since it is no quantity. It
  reads only the last 64 KiB of a call log, for its `end` line, and lists `.log` files,
  `first-flag.csv`, `contradictions.csv` and a `live` test's `pytest` directory without reading
  them. A folder whose name is not a run id, a symbolic link, and an entry of the wrong kind are
  not listed; an artifact that cannot be read shows its error in place of its section.
- **Provenance**: a run folder that holds `trace.jsonl` ([Evaluation](#evaluation), `trace`) gets a
  picker over its records by arm, holdout and position, each shown with the facts that determine it
  (ADR [`research/record-explainability`](adr/research/record-explainability.md)). The view reads
  only that file, again only when its modification time or size changes. A demo fill of a
  Messages arm, live or replayed, gets a picker over its added rows, each traced from the fill's own
  call log and candidates file with no API call and no read of the data root, again only when
  either file changes. Both render a trace only as
  frames, JSON and plain code blocks, and the download carries no source record id.
- **What spends money**: only a live fill by an arm of `bko.evaluate.experiment.BKO_ARMS`. It starts
  only while its confirmation box is ticked, the box clears after each fill, the client is built
  only then, and the fill stops once its call log's spend reaches `bko.app.context.SPEND_CEILING`.
  One action runs at a time in the process, and a fill sends one call at a time through the async
  client (`run=app`, in-flight 1, the experiment's idle limit). The credential is the SDK's own
  (`ANTHROPIC_API_KEY` or `ant auth login`); a missing one stops the fill at its first call, before
  any spend.
- **Replay**: a BKO fill can instead answer every call from the call log of an earlier fill by the
  same arm, picked from a list, with no API call. It refuses a log whose requests differ from the
  fill's or that holds calls the fill does not make (ADR
  [`research/call-log-replay`](adr/research/call-log-replay.md)), so a log replays only at the
  commit, data and releases it was recorded with.
- **What it writes**: besides the data root, one run folder per fill ([Runs](#runs)),
  `runs/app-<UTC time>-<arm>/`, holding `scores.csv` and `training.csv` and, for a BKO fill, live or
  replayed, `calls.jsonl` in the format [Generation](#generation) describes and
  `holdout-candidates.jsonl` in the format [Evaluation](#evaluation) describes, its header's run
  `app`. A stopped fill keeps what it wrote, and its message names the folder and the spend its call
  log recorded. A closed tab or Ctrl-C does not interrupt a running action; the process finishes it
  first.
- **The gate drives it** through `streamlit.testing.v1.AppTest` over `src/bko/app/streamlit_app.py`
  with fake openers, fast method arms and a `ScriptedSend` or a call log recorded from one in
  `tmp_path`; no gate test reaches the API.
- **The live test**: `tests/test_app_live.py` makes one paid BKO fill through the page against the
  data root (fetched if absent) and runs only under `--live`. Run it deliberately, as a G1 run:

      test ! -e runs/<run-id>/pytest && mkdir -p runs/<run-id> && uv run --no-sync pytest tests/test_app_live.py --live --basetemp=runs/<run-id>/pytest -q -p no:cacheprovider --no-cov -o log_cli=true --log-cli-level=INFO </dev/null

- **The release test**: `tests/test_app_pinned_releases.py` drives a free fill through the page
  over the releases already in `data/`, under `--releases`:

      uv run --no-sync pytest tests/test_app_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

- **After a Streamlit bump**: `uv run --no-sync python -m streamlit config show` lists every option
  the installed release accepts; each recipe flag must still be one of them.

## Video

`video/` ([README](../video/README.md)) holds the ManimGL scenes of the demo video, a silent
animation of how the system works, with a label and a one-line explainer per beat, that a presenter
talks over.

- **What it does**: `make video` renders each scene of `video.captions.ORDER`, joins them and
  writes `video/media/bko-demo.mp4`. It exits 1 when a scene fails and 2 when FFmpeg or the
  environment is missing, and it writes nothing under `runs/`.
- **Install**: `make video-env` builds `video/.venv` from `video/requirements.txt`, with hashes,
  on Python 3.14, an interpreter that environment pins for itself, apart from `.python-version`.
  ManimGL is not an extra (ADR
  [`tooling/video-scenes-outside-the-lock`](adr/tooling/video-scenes-outside-the-lock.md)). That
  environment is not the project's, so `uv.lock` and `.venv` are untouched.
- **Changing the requirements**: edit `video/requirements.in`, then from the repository root:

      uv pip compile video/requirements.in --universal --python-version 3.14 --generate-hashes -o video/requirements.txt

- **What it needs**: FFmpeg and ffprobe on PATH, the font Helvetica Neue and a GPU session. No
  LaTeX and no API call.
- **Where it lands**: `video/media/`, gitignored.
- **What the gate checks**: `make lint` and `make sizes` reach `video/`; `make typecheck` reports
  nothing in `video/`, and coverage and the docs tests do not reach it. The `tests/test_video_*.py`
  files hold the model, the captions' facts, the driver's exit codes and the scene order, read as
  source; the gate never renders.
- **Length**: `ffprobe -v error -show_entries format=duration -of csv=p=0 video/media/bko-demo.mp4`
  reports it.
- **Model against the data**: `tests/test_video_model_releases.py` compares the example model's
  cell counts, is_a edges, holders, share and gaps with UCI Heart Disease's default split and the
  pinned HPO release already in `data/`, under `--releases`:

      uv run --no-sync pytest tests/test_video_model_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

## Benchmark environments

A benchmark tool that needs a system library, another Python, or packages the gate has no other
use for runs in its own hashed environment under `bench/<tool>/`, outside the lock (ADR
[`tooling/benchmark-tools-outside-the-lock`](adr/tooling/benchmark-tools-outside-the-lock.md)).
`bench/<tool>/` holds `requirements.in`, the hashed `requirements.txt` compiled from it, one script
that imports no `bko`, the script's own test under `tests/` and a README. `bko` runs the script as a
subprocess and never imports the tool.

- **SynthEval** ([README](../bench/syntheval/README.md)): `make syntheval-env` builds
  `bench/syntheval/.venv` from `bench/syntheval/requirements.txt`, with hashes and wheels only, on
  Python 3.14; `uv.lock` and `.venv` are untouched. On macOS LightGBM, which SynthEval imports,
  needs Homebrew's OpenMP runtime, for this environment only:

      brew install libomp

  `make syntheval-test` runs `bench/syntheval/tests/` with the project's pytest; it calls
  `bench/syntheval/score.py` under the environment's interpreter on a tiny fixture fill and exits
  2 without the environment. `python -m bko.evaluate syntheval` reads with it
  ([Evaluation](#evaluation)).
- **Changing the requirements**: edit `bench/<tool>/requirements.in`, then from the repository root:

      uv pip compile bench/syntheval/requirements.in --universal --python-version 3.14 --generate-hashes -o bench/syntheval/requirements.txt

- **What the gate checks**: `make lint` and `make sizes` reach `bench/`; `make typecheck` and
  coverage do not. The gate never imports SynthEval: `tests/test_evaluate_syntheval.py` tests the
  export, the call and the parsing with a fake runner and parses the script with `ast`, its flags
  against `score_argv`'s, and a PR that changes the script runs `make syntheval-test`.

## Platforms

The gate is POSIX-only: macOS and Linux. The checkers shell out to `git`, and the Makefile recipes
assume a POSIX shell.

**Use an SSH remote.** The checkers run `git` with global config blanked, which also drops any
global credential helper. With an HTTPS remote, `make rebased` cannot authenticate its `ls-remote`
probe, so it passes only in its degraded form and never confirms the trunk is current.

## Make targets

| Target | What it runs | Notes |
|---|---|---|
| `setup` | `uv sync --locked` with the `dev`, `data`, `synth`, `eval`, `ontology`, `llm`, `app` and `slm` extras and the pinned interpreter | Refuses a stale lock |
| `sync` | `uv sync --locked --all-extras` | The whole lock, every extra |
| `lint` | `ruff check .` then `ruff format --check .` | `ruff format` reaches Python fenced in markdown |
| `format` | `ruff format .` | Rewrites files, documentation fences included; not part of the gate |
| `typecheck` | `ty check` | `ty.toml` sets the platform and paths; `pyrightconfig.json` is for editors |
| `test` | `pytest -n auto --dist worksteal --cov` | Sources, branch coverage and `fail_under` live in `pyproject.toml` |
| `sizes` | `python -m tools.check_file_size` | ADR [`tooling/file-size-cap`](adr/tooling/file-size-cap.md) |
| `cells` | `python -m bko.datasets cells` | Not part of the gate; fetches UCI Heart Disease on its first run ([Data](#data)) |
| `synthea` | `python -m bko.datasets synthea` | Not part of the gate; needs a JDK, and fetches the Synthea jar and generates on its first run ([Data](#data)) |
| `reproduce` | `python -m bko.ontology reproduce` | Not part of the gate; fetches MONDO and DO on its first run ([Fusion reproduction](#fusion-reproduction)) |
| `app` | `python -m streamlit run src/bko/app/streamlit_app.py` with the app's flags | Not part of the gate; serves until interrupted, and a live BKO fill spends money ([App](#app)) |
| `synced` | `python -m tools.check_lock_drift` | Installed packages against `uv.lock` |
| `rebased` | `python -m tools.check_rebased` | Probes `origin` with `git ls-remote`; ADR [`tooling/refuse-a-stale-trunk-ref`](adr/tooling/refuse-a-stale-trunk-ref.md) |
| `rebased-final` | The same checker again | Runs last, because the trunk can move while the tests run |
| `check` | `rebased synced lint sizes typecheck test rebased-final` | `.NOTPARALLEL`, so the order is a guarantee |
| `video-env` | `uv venv` then `uv pip sync --require-hashes --only-binary :all: video/requirements.txt` into `video/.venv` | Not part of the gate; installs outside the lock ([Video](#video)) |
| `video` | `python -m video.render` in `video/.venv` | Not part of the gate; needs `make video-env` and FFmpeg; no API call ([Video](#video)) |
| `syntheval-env` | `uv venv` then `uv pip sync --require-hashes --only-binary :all: bench/syntheval/requirements.txt` into `bench/syntheval/.venv` | Not part of the gate; installs outside the lock; on macOS needs `brew install libomp` ([Benchmark environments](#benchmark-environments)) |
| `syntheval-test` | `pytest bench/syntheval/tests` | Not part of the gate; needs `make syntheval-env`; runs the SynthEval script's real test ([Benchmark environments](#benchmark-environments)) |

Every step from `lint` down runs as `uv run --no-sync`, except `video-env` and `video`, which run
in `video/.venv`, and `syntheval-env`, which builds `bench/syntheval/.venv`: the gate measures the
environment `make setup` built and never re-locks or installs during a run.

The `check` order puts what invalidates other steps first: a rebase can move `uv.lock`, which
invalidates `synced`, and a drifted environment makes `lint` and `typecheck` report errors that
belong to the environment rather than the tree. `sizes` precedes `test` because it is the cheaper
of the two.

## No CI, no hooks

There is no CI workflow and no git hook. `make check` is the whole gate and a person runs it before
opening a pull request. Name in the PR whether the whole gate or a targeted subset stands behind
it.

A targeted run while working:

- `make rebased synced lint sizes`, always: each guards something a narrow test run cannot see.
- `make typecheck` for any `.py` change.
- The test modules for what changed, without coverage, for example
  `uv run --no-sync pytest tests/test_file_size.py -q -p no:cacheprovider --no-cov </dev/null`. A
  run whose every selected test skips exits 1, so a selection that reaches nothing cannot pass;
  `--allow-all-skipped` opts out where that is the intent.
- Every test module that names the changed file, found with `grep -rlF '<basename>' tests/`.

## Gate hazards

- **Stage before the gate.** `make sizes`, the conflict-marker scan and the root-file check read
  `git ls-files`, so an untracked file is under no gate.
- **Do not edit the tree while the gate runs.** Coverage resolves line numbers at report time, so an
  edit fakes a coverage miss below the edit point.
- **Do not run two covered gates at once.** They consume each other's `.coverage.*` fragments.
  Check `pgrep -f "pytest -n auto"` first and delete leftover `.coverage.*` files.
- **Paid and release tests are opt-in, each behind its own flag.** A test marked `live` calls a
  paid API and is skipped unless `--live` is passed; a test marked `releases` reads the pinned
  releases in `data/` and is skipped unless `--releases` is passed. No `-m` expression selects
  either, and one flag does not run the other's tests.
- **A session guard refuses every other Anthropic request.** `tests/conftest.py` fails any request
  that reaches the SDK's request sink outside a `live` test's body: at collection, in a fixture, or
  in an unmarked test (ADR
  [`tooling/paid-api-request-guard`](adr/tooling/paid-api-request-guard.md)). Without `--live` it
  also points `ANTHROPIC_BASE_URL` at the loopback discard port, so a subprocess cannot spend.
  The conftest imports anthropic, so every pytest run needs the `llm` extra. The app tests import
  `streamlit.testing`, so every pytest run also needs the `app` extra. A request from an AppTest
  script thread is refused too: the refusal kills the script thread, so the test fails on what
  AppTest's `run` raises, and the guard's message goes to the thread's exception hook.
- **A `live` run spends money.** `--live` lifts the guard for each `live` test's body and leaves
  subprocesses unguarded. Never set `ANTHROPIC_LOG=debug`: the SDK then logs prompts, training
  records included for a CLLM arm. Importing `bko.generate.api` holds the SDK's `anthropic` logger
  at INFO, so no entry point logs a request body. Setting `ANTHROPIC_LOG` leaves the bko INFO
  lines in place: the SDK's import installs a root handler, and every entry point's logging setup
  replaces it (`force=True`).
- **A test that changes the logging state fails.** A test that leaves the root logger's level or
  handlers, warning capture (`logging.captureWarnings`) or `warnings.showwarning`, or the level
  of any other logger changed errors at teardown with `logging state changed:` and each change,
  the short ones first and each logger as `name before -> after`. `tests/conftest.py` snapshots
  the state as each test's setup starts and compares it after its teardown, then restores it. A
  logger created during the test compares against `NOTSET`. A test that calls `configure`
  restores the state itself or stubs it.
- **No `bko.slm` test connects a socket.** `tests/conftest.py` patches `socket.socket.connect` to
  raise its `NETWORK_REFUSAL` for each test, fixtures included, of a module named `test_slm_*`,
  so a load that would reach the Hugging Face Hub fails the test.
- **One test reaches the network.** The `check_rebased` recipe case in `tests/test_git_runner.py`
  runs the real probe against `origin` and carries the `integration` marker;
  `-m "not integration"` deselects it.
- **Importing SDV writes outside the repository.** `sdv.single_table` creates SDV's user data
  directory (`platformdirs.user_data_dir("sdv", "sdv-dev")`) and copies its logger configuration
  there on import, so the baselines and evaluation tests touch it. No test reads it.
