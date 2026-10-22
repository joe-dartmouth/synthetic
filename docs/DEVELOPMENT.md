# Development: the environment, the gate and its hazards

How this repository is built, checked and run. [`AGENTS.md`](../AGENTS.md) keeps the working
summary; this file is the full record behind it. The repository's own checkers are described in
[`tools/README.md`](../tools/README.md).

## Environment

Requires `uv`. The interpreter version is the one in `.python-version`, and `uv` downloads it when
the machine lacks it. `make setup` installs the `dev`, `data`, `synth`, `eval`, `ontology`, `llm`
and `app` extras from the lock, and `source .venv/bin/activate` enters the environment.

- **Runtime dependencies are optional extras, declared and locked**: `data` (numpy, pandas, scipy),
  `synth` (sdv, imbalanced-learn, torch), `eval` (scikit-learn, sdmetrics, scipy), `ontology`
  (pronto), `llm` (anthropic) and `app` (streamlit) are installed by `make setup`. `synth` brings `torch` to every machine, and on Linux the lock adds the
  NVIDIA CUDA packages and `triton`. ADR [`tooling/uv-ruff-ty`](adr/tooling/uv-ruff-ty.md) states
  when an extra joins `make setup`. `make sync` installs every extra.
- **Try a new extra before it joins `make setup`, the lock-faithful way**: copy the `setup` recipe out of the `Makefile` and
  append `--extra <name>`. `uv sync` is exact, so a bare `uv sync --extra <name>` removes the
  others. Never `uv pip install -e ".[...]"`: it resolves fresh rather than from the lock, and
  `make synced` refuses the result.
- **Change a dependency with `uv lock`, then `make setup`.** Both install targets run
  `uv sync --locked`, which refuses a lock that disagrees with `pyproject.toml` instead of
  rewriting it, and every gate step runs `uv run --no-sync`, so nothing but an explicit `uv lock`
  writes `uv.lock`. `tests/test_repo_gate.py` compares the two offline as well.
- **Build backends are pinned** in `[tool.uv] build-constraint-dependencies`, because uv resolves
  them outside the lock: hatchling for `bko`.

## Data

The evaluation data is downloaded or generated, never committed. It lives under `data/` at the
repository root, which `.gitignore` ignores, and only
[`bko.datasets`](../src/bko/datasets/README.md) writes there. Each download (UCI Heart Disease, the
Synthea jar, and the MONDO, HPO and DO releases) is pinned by size and sha256 in
`bko.datasets.sources`. A download that differs from its pin is refused; the commands delete and
fetch again a snapshot on disk that fails its pin, and the UCI loader refuses one. A generated population is checked against its own `run.json`: the
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
  the release tests read `data/` only when `--releases` is passed ([Ontologies](#ontologies)). No
  test runs Java, no gate test reaches the Anthropic API, and the one test that reaches the network
  is under [Gate hazards](#gate-hazards).

## Baselines

`bko.baselines` ([README](../src/bko/baselines/README.md)) builds each comparison method's training
set from the training split `bko.datasets` computes (ADR
[`research/baseline-augmentation-design`](adr/research/baseline-augmentation-design.md)). One
command per method, inside `.venv`:

    uv run --no-sync python -m bko.baselines <none|class-weight> [--data-dir <dir>]
    uv run --no-sync python -m bko.baselines <smote|adasyn|ctgan|tvae> --rows <n> [--strata cell|label] [--seed <s>] [--data-dir <dir>]

- **What it does**: it fetches UCI Heart Disease if the data root lacks it, splits it with
  `SPLIT_SEED`, water-fills `--rows` over the strata (cells unless `--strata label`), runs the
  method and writes the training set to stdout as CSV: `record_id` (negative for an added row), the
  predictors, `target` and `weight`. `--seed` defaults to `BASELINE_SEED` in
  `bko.baselines.request`.
- **What it logs**: stderr gets the split line `cells` prints, one `quota` line per stratum, a line
  with the method (and, for a row-adding method, its seed and strata), the real, synthetic and total
  row counts and the output's sha256, the fit and fill lines of CTGAN and TVAE, and the method's
  parameters. An experiment copies them into its configuration in
  [EVIDENCE.md](EVIDENCE.md#how-to-read-an-entry); the commit's `uv.lock` gives the library
  versions.
- **Exit codes**: as in [Data](#data), with 2 also for a request a method cannot fill and 3 for a
  method whose output breaks the contract.
- **Reproducibility**: a seed gives the same bytes on one machine with the same lock; CTGAN and
  TVAE promise no more, so the logged sha256 is the record.

## Evaluation

`bko.evaluate` ([README](../src/bko/evaluate/README.md)) scores every arm of E1 on its repeated
holdouts and judges a candidate arm against E1's criteria (ADR
[`research/e1-evaluation-protocol`](adr/research/e1-evaluation-protocol.md)). `score` spends money:
its three BKO arms call the Anthropic Messages API, so it needs the `llm` extra and an Anthropic
credential, and every call goes to the call log at `--log`, which must not exist. `compare` spends
nothing. Inside `.venv`, writing to the run's folder under the gitignored `runs/` ([Runs](#runs)):

    mkdir -p runs/<run-id>
    test ! -e runs/<run-id>/calls.jsonl && test ! -e runs/<run-id>/scores.csv && test ! -e runs/<run-id>/scores.log && uv run --no-sync python -m bko.evaluate score --log runs/<run-id>/calls.jsonl [--data-dir <dir>] > runs/<run-id>/scores.csv 2> runs/<run-id>/scores.log
    mkdir -p runs/<run-id>-replay && test ! -e runs/<run-id>-replay/calls.jsonl && test ! -e runs/<run-id>-replay/scores.csv && test ! -e runs/<run-id>-replay/scores.log && uv run --no-sync python -m bko.evaluate score --replay runs/<run-id>/calls.jsonl --log runs/<run-id>-replay/calls.jsonl [--data-dir <dir>] > runs/<run-id>-replay/scores.csv 2> runs/<run-id>-replay/scores.log
    uv run --no-sync python -m bko.evaluate compare runs/<run-id>/scores.csv [--candidate <arm>]

The `test ! -e` guards matter: the shell empties `scores.csv` and `scores.log` before `score` can
refuse an existing log, and `scores.csv` is the only copy `compare` reads. A judged run keeps
`compare`'s stdout and stderr in its folder:

    test ! -e runs/<run-id>/criteria.csv && test ! -e runs/<run-id>/criteria.log && uv run --no-sync python -m bko.evaluate compare runs/<run-id>/scores.csv > runs/<run-id>/criteria.csv 2> runs/<run-id>/criteria.log

- **What `score` does**: it fetches UCI Heart Disease if the data root lacks it, then for each of
  `E1`'s holdout seeds splits the data with that seed and gives every arm the split's training rows
  and the request that fills each underrepresented cell up to the threshold. A classifier fit on the
  real rows, weighted as the arm says, and the added rows is scored on the test split. The replica
  arms also fill the training split's own cell counts; those rows alone train a classifier and are
  compared with the training split for fidelity. It writes one row per holdout, arm, source, measure
  and scope to stdout as CSV; a measure a scope cannot define is empty. Each BKO arm reads the
  split's training rows' gaps under its definition, asks the API for each requested cell's
  shortfall, in the gap request and the replica request alike, validates every reply with
  `bko.validate`, drops what `bko.baselines.records.check` would refuse, and keeps the rest. An
  attempt that cannot fill the request inside `E1`'s budget is refused and retried like a
  baseline's. Every arm fills before any is scored. A test record that some arm's rows copy in every
  column leaves the test split each arm's utility is scored on. With `--replay`, every BKO call is
  answered from that call log instead of the API. A call that differs from the log's, or a logged
  call left unplayed, stops the run.
- **What `compare` does**: it reads that CSV and writes one row per paired contrast as CSV: the
  criterion, the scope, the criterion's verdict (`met`, `not met` or `inconclusive`), the arm, the
  arm it is set against, the measure, and the contrast's mean, interval, the upper bound at the
  family-wise level that `not met` reads (`refuting_high`), and holdout counts. It refuses a scores
  file whose holdouts are not `E1`'s, and one where a series it judges misses a holdout or holds an
  empty value. `--candidate` judges another row-adding arm, a dry run whose verdict is not E1's. It
  also judges validity from the `candidates` and `replica-candidates` rows, the gap and replica
  fills' counts: the candidate's `passed` summed over `generated` summed over both fills, met at
  `E1.validity` or above. It logs each arm's mean recall and AUC-PR over the holdouts, and each
  replica arm's mean replica recall, AUC-PR and fidelity, with no verdict. A scores file is compared
  at the commit that scored it: a replica arm's validity needs its `replica-candidates` rows.
- **What it logs**: each stderr line starts with its time. `score` logs an `experiment` line with
  the budget, the validity threshold and the spend ceiling, and a `versions` line (the checkout's
  commit and whether its tree is dirty, `unknown` without git, then Python, the platform, the
  evaluation stack, the SDK and the ontology parser), then per holdout the split line `cells`
  prints, one `quota` line per stratum, one `arm` line per fill (a replica fill is named
  `<arm>-replica`) with the attempt, the seed, the added rows' sha256 and the method's parameters, a
  `retry` warning for each refused attempt before the last, and a `holdout` line with the progress
  and seconds, then the CSV's row count and sha256. It also logs per BKO attempt a `guided` line
  (outcome, seed, call range, records asked, candidates generated and passed, dropped for each
  reason, kept, the run's spend), a `spend` line before each BKO call with the run's spend so far
  and the ceiling, `bko.generate`'s `log` line and per-call lines and the call log's totals line, a
  `test` line per holdout with the test records any arm copied, and at the end of the run, a stopped
  run included, one `validity` line per BKO arm, its counts over both fills. A fill that stops the
  run logs a `refused` or `contract` error naming the holdout and arm. `compare` logs a `compare`
  line with the candidate, whether it is a dry run and the scores file's sha256, then one
  `criterion` line per criterion with its verdict, a `criterion=validity` line, a `validity` line
  per other BKO arm, a `candidates` line per BKO arm and fill (`fill=gap`, `fill=replica`) with its
  counts and rate, a `copies` line per arm, a `mean` line per arm, measure and scope, and a
  `replica` line per replica arm, measure and scope. An experiment copies them into its Runs row in
  [EVIDENCE.md](EVIDENCE.md#how-to-read-an-entry).
- **Where the configuration lives**: `bko.evaluate.experiment.E1`; no flag changes a value;
  `--log` and `--replay` choose where calls are recorded and answered from, never a value.
- **Exit codes**: as in [Data](#data), with 2 also for a request with nothing to fill, a fit that
  does not converge, a method that refuses a request on every attempt or scores that cannot be
  judged, and 3 for a method's output or a scores file that breaks its contract; 2 also for an
  existing call log, BKO arms with no `--log`, `--log` or `--replay` with no BKO arm to build, a
  call that gets no response, a failed gap query, a run whose spend reached `E1.spend_ceiling` and
  `candidates` or `replica-candidates` rows whose validity cannot be read; 3 also for a replay that
  does not match its log. `compare` writes its table before it reads validity, so it keeps the table
  when it exits 2 on validity.
- **Reproducibility**: a holdout seed gives the same split and requests, and the baselines keep
  the promise of [Baselines](#baselines). A BKO arm's rows do not repeat, since the API takes no
  seed; the paid run's call log is their record (ADR
  [`research/call-log-replay`](adr/research/call-log-replay.md)), and `--replay` rewrites the
  scores byte for byte from it.

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
`bko.datasets.sources` pins, and holds the table that maps their concepts to UCI fields and to the
Synthea condition the ablation removes (ADR
[`research/ontology-mapping-design`](adr/research/ontology-mapping-design.md)). One command, inside
`.venv`:

    uv run --no-sync python -m bko.ontology check [--data-dir <dir>]

- **What it does**: it fetches the two releases into the data root if absent, verifies each against
  its pin, parses them with pronto, and checks every concept row: the concept exists and is not
  obsolete, its label is the release's name, a Synthea row's SNOMED-CT code is a
  `skos:exactMatch` of its concept and of no other live concept, and an out-of-scope concept is
  not an `is_a` ancestor of a mapped one. stdout gets a CSV with the columns `table` (`field`,
  `code` or `out-of-scope`), `subject` (the dataset side, such as `cp in 1|2|3`), `concept`,
  `label` and `status` (`ok`, or how the release disagrees), one row per concept row, the same
  bytes on every run over the same releases.
- **What it logs**: stderr gets the parser's version, each release's snapshot identity (as in
  [Data](#data)), the `data-version` its header names, its term counts, and one warning per
  disagreeing row.
- **Exit codes**: 0 every row agrees; 1 an unexpected error, with its traceback; 2 the command
  could not run; 3 a release contradicts its pin or is not OBO the loader reads (stdout is
  empty); 4 a row disagrees with its release (the report is written first).
- **Re-pinning**: edit the release's `Source` (URL, size, sha256), run this command, review each
  row it reports, then run the release tests below.
- **The gate does not run it.** `make check` tests the loader on OBO text built in `tmp_path` and
  the table's invariants; a row naming a concept the release lacks passes `make check` and fails
  this command, so run it after editing the table or re-pinning a release.
- **The release tests**: `tests/test_ontology_pinned_releases.py` loads MONDO, HPO and DO from the
  data root and asserts that every row agrees and that the
  [fusion reproduction](#fusion-reproduction) differs nowhere. The gate skips them, and no `-m`
  expression runs them; `--releases` does:

      uv run --no-sync pytest tests/test_ontology_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

  They read and never fetch, so run `check` and `reproduce` first: `check` fetches MONDO and HPO,
  `reproduce` MONDO and DO. An absent release fails them. Run them after a re-pin, a change to
  `bko.ontology.obo`, or a change to pronto in `uv.lock`, from the checkout whose `data/` holds
  the releases.

## Fusion reproduction

`bko.ontology.santos` rebuilds the MONDO/DO fusion example of Santos et al. (section 7 of
[`papers/santos-2024.pdf`](../papers/santos-2024.pdf)) from the pinned MONDO and DO releases, with
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

## Gap detection

`bko.gaps` ([README](../src/bko/gaps/README.md)) lists a dataset's gaps four ways: as (cell,
statement) pairs under each definition of ADR `research/bko-formal-gap-definition`
(`per-subgroup`, `population-level` and `frequency-aware`, in that order), and as the age-sex
subgroup-label cells holding fewer records than `UNDERREPRESENTED_BELOW` (`count`). Two commands,
inside `.venv`:

    uv run --no-sync python -m bko.gaps uci [--data-dir <dir>]
    uv run --no-sync python -m bko.gaps synthea [--data-dir <dir>]

- **What `uci` does**: it fetches UCI Heart Disease and the releases the field mappings name into
  the data root if absent, splits the records as [Data](#data) says, and lists the training
  split's gaps. stdout gets a CSV whose column `detector` names a definition or `count`. A
  definition's row fills the statement's `source`, `antecedent`, `antecedent_label`, `consequent`
  and `consequent_label` and its cell's `age_band`, `sex` and `target`; a population-level gap
  holds in every cell, so it has a row per cell. A `count` row fills the cell and its `count`.
- **What `synthea` does**: it generates the population `bko.gaps.experiment.E2_RUN` names if absent
  ([Data](#data)), fetches the releases the code mappings name, and for each ablation in
  `E2_ABLATIONS` lists every definition's gaps and the count-based ones before and after the
  removal. stdout gets a CSV with the column `ablation`, then `uci`'s columns without `count`,
  then `before`, `after` and `truth`: whether the row is a gap before the removal, after it, and
  among the gaps the removed patients account for. Every cell uses the ablation's age bands
  (`AblationSpec.edges`). A definition's cells carry no label, so its rows leave `target` blank,
  and a (cell, statement) pair is listed when it is a gap before or after the removal or in the
  truth; a `count` row's cell is listed when it is a gap before or after the removal or its count
  fell. [EVIDENCE.md](EVIDENCE.md#e2-synthea-gap-detection-ablation) (E2) holds the reading.
- **What it logs**: stderr gets the data root and the pandas and scipy versions, each input's
  identity (the split or the population, the parser and each release), each ontology fragment's
  size, how many records show each mapped concept, each cell's record count, held-concept count
  and assumed concepts, each concept the frequency-aware definition flags with its counts and
  posterior mass, each fused BKB's size and name, for `uci` each definition's gap and cell counts
  and the count-based cells with the threshold, for `synthea` each ablation, each definition's gap
  counts before and after the removal and its listed rows, and each detector's recovery (truth,
  eligible, recovered, outside the truth, new outside the truth, and vanished), with a warning when
  per-subgroup or the count-based detector puts a gap new outside the truth or loses one, which
  removing whole patients cannot cause, and last the row count and sha256 of stdout.
- **Exit codes**: 0 the report is written; 1 an unexpected error, with its traceback; 2 the
  command could not run, including a mapped concept its release lacks; 3 the data or a release
  contradicts its pin or schema, including a release naming an `is_a` parent it lacks.
- **Reproducibility**: the same bytes on every run over the same archive, population and releases.
- **The gate does not run it.** `make check` runs it over an archive, a population and releases
  built in `tmp_path` (`tests/test_gaps_cli.py`).

## Generation

`bko.generate` ([README](../src/bko/generate/README.md)) asks the Anthropic Messages API for
candidate records that fill a gap request, E1's count-based one or the completeness request (ADR
[`research/record-generation-design`](adr/research/record-generation-design.md)). One command,
inside `.venv`, with `make setup`'s `llm` extra and an Anthropic credential the SDK can resolve:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/calls.jsonl && test ! -e runs/<run-id>/candidates.jsonl && test ! -e runs/<run-id>/candidates.log && uv run --no-sync python -m bko.generate <definition> [--guidance <level>] [--request <e1|completeness>] --log runs/<run-id>/calls.jsonl [--data-dir <dir>] > runs/<run-id>/candidates.jsonl 2> runs/<run-id>/candidates.log

- **What it does**: it fetches UCI Heart Disease and the releases the field mappings name into the
  data root (`--data-dir`, default `data/`) if absent, splits the records as [Data](#data) says,
  and reads the training split's gaps under `<definition>`: `per-subgroup`, `population-level` or
  `frequency-aware`. With `--request e1`, the default, it asks each cell holding fewer training
  records than `UNDERREPRESENTED_BELOW` for its shortfall; with `--request completeness` it asks
  each cell holding a per-subgroup gap for `UNDERREPRESENTED_BELOW` rows, and exits 2 for any
  `<definition>` but `per-subgroup` ([Gap closure](#gap-closure)). It asks in calls of at most
  `Settings.per_call` records and writes one JSON line per asked record to stdout: `call`, `cell`, `position`, and the decoded `candidate` or the `undecoded` reason.
  Each call's lines are written and flushed as it returns, so a failed or interrupted run's stdout
  holds every batch that returned before the failure. Candidates are unfiltered and unrepaired;
  `bko.validate` checks them.
- **What leaves the machine**: each field's meaning, codes and unit, the plausibility ranges, the
  cell and its training record count, and the guidance of the chosen level. Never a record.
- **Guidance levels**: `cell`, `coherent`, `findings`, `ontology` (the default), `profile` and
  `summary`, each adding one component to the one before it; the ADR defines each.
- **What it logs**: the call log at `--log` gets a `run` line, a `request` line before each call and
  a `reply` or `error` line after it, and an `end` line with the run's status, the replayed log's
  sha256 (or null) and totals, each call's whole request and response and its USD spend included. An
  interrupt or any other failure between a `request` line and its `reply` line writes the `error`
  line. stderr gets the split line, each ontology fragment's size, the request's name, cells and
  rows, a `quota` line per asked cell, a `log` line (path, schema, model, effort, level, records per
  call, token limit, and the replayed log's sha256 or `none`), the plan (calls, level, and the most
  the output tokens can cost), a `send` line before each call and a `generate` line after it (or an
  ERROR `failed` line naming the call and cell), the run's totals and spend with the replayed log's
  sha256 or `none` and the log's path, and last the line count and sha256 of stdout. Its WARNING
  lines: `served`, a call served by a model other than the settings name; `short`, a call that wrote
  placeholders, with their reason; `unreached`, a gap statement no label-compatible mapping row
  reaches, left out of the prompt; `noreference`, a cell with no reference record, so no profile or
  summary; and the UCI loader's line naming the records it dropped for a missing value. The SDK
  client retries a failed attempt up to `bko.generate.api.MAX_RETRIES` times, backing off between
  tries, and logs each retry at INFO between the call's `send` line and its result; each `reply` and
  `error` line records the call's `attempts`, and the `end` line counts the calls `retried`. A
  retried attempt can be billed, so logged spend is a lower bound when `retried` is above 0.
- **Exit codes**: 0 the candidates are written; 1 an unexpected error, with its traceback; 2 the
  command could not run, including a missing credential, a call that got no response, a
  completeness request no cell holds a gap for, `--request completeness` with another
  `<definition>`, and a `--log` path that exists (checked before any download or call); 3 the data
  or a release contradicts its pin or schema.
- **Reproducibility**: none. Each call spends money and a rerun's candidates differ; the call log
  is the record, so the command never overwrites one.
- **The gate does not run it.** `make check` drives it through a fake `Send` over an archive and
  releases built in `tmp_path` (`tests/test_generate_cli.py`); no gate test reaches the API.
- **The release test**: `tests/test_generate_pinned_releases.py` builds every cell's guidance under
  each definition over the real mapping table, UCI snapshot and pinned releases in the data root,
  and renders it at every level. It never fetches and never calls the API; run it with
  `--releases` after changing guidance or the prompt, once the command or `make cells` and
  `python -m bko.ontology check` have fetched the data:

      uv run --no-sync pytest tests/test_generate_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

- **The live test**: `tests/test_generate_live.py` makes one paid call for 2 records and checks the
  reply's keys, types and codes. It runs only under `--live`. Run it deliberately, after a change to
  the schema or the call, into a `--basetemp` that does not exist yet, since pytest wipes it:

      test ! -e runs/<run-id>/pytest && mkdir -p runs/<run-id> && uv run --no-sync pytest tests/test_generate_live.py --live --basetemp=runs/<run-id>/pytest -q -p no:cacheprovider --no-cov -o log_cli=true --log-cli-level=INFO

## Gap closure

`bko.fusion` ([README](../src/bko/fusion/README.md)) joins the generated records that pass
validation to the training records of the cell each was asked for and lists the gaps again, the
completeness criterion's reading of "after fusion" (ADR
[`research/gap-closure-design`](adr/research/gap-closure-design.md) holds which records join).
Two commands, inside `.venv`; the first spends money and needs an Anthropic credential, the
second spends nothing:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/calls.jsonl && test ! -e runs/<run-id>/candidates.jsonl && test ! -e runs/<run-id>/candidates.log && uv run --no-sync python -m bko.generate per-subgroup --guidance ontology --request completeness --log runs/<run-id>/calls.jsonl > runs/<run-id>/candidates.jsonl 2> runs/<run-id>/candidates.log
    mkdir -p runs/<run-id>-closure && test ! -e runs/<run-id>-closure/closure.csv && test ! -e runs/<run-id>-closure/closure.log && uv run --no-sync python -m bko.fusion runs/<run-id>/candidates.jsonl [--data-dir <dir>] > runs/<run-id>-closure/closure.csv 2> runs/<run-id>-closure/closure.log

The `test ! -e` guards matter: the shell empties `candidates.jsonl` before `bko.generate` can
refuse an existing log, and `candidates.jsonl` is the only copy of the candidates the second
command reads.

- **What it does**: it reads the candidates file, fetches UCI Heart Disease and the releases the
  field mappings name into the data root if absent, splits the records as [Data](#data) says,
  and lists the training split's gaps under every definition. It refuses a file that is not the
  completeness request's answer: a cell the request does not ask for, or fewer lines for a
  requested cell than its rows. It validates each requested cell's lines with `bko.validate`, an
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
  line count and sha256 (the `candidates` digest `bko.generate` logged, when the file is its
  stdout), the split line, each ontology fragment's size, a `request` line per requested cell, a
  `validate` line naming each cell before `bko.validate`'s counts for it, a `join` line per
  requested cell (lines, passed, off-cell, past its rows, joined, and the concepts the join newly
  gives the cell), a `closure` line per definition (before, after, closed, open, new and the
  share), and last the row count and sha256 of stdout, written once the report is. Its WARNING
  lines: `off_cell`, accepted records dropped for naming another cell; `past_quota`, accepted
  on-cell records dropped past the cell's rows; and `unjoined`, a requested cell no record joined.
- **Exit codes**: 0 the report is written; 1 an unexpected error, with its traceback; 2 the
  command could not run, including a missing candidates file and a training split no cell holds
  a per-subgroup gap in; 3 the candidates file is malformed or does not answer the completeness
  request, or the data or a release contradicts its pin or schema.
- **Reproducibility**: the same bytes on every run over the same inputs. The Messages API takes
  no seed, so the candidates file is the paid run's record, not a rerun.
- **The gate does not run it.** `make check` drives it over an archive, releases and candidates
  built in `tmp_path`; no gate test reaches the API.
- **The release test**: `tests/test_fusion_pinned_releases.py` joins records that show every
  mapped concept to the real training split, the control that closes every per-subgroup gap. Run
  it with `--releases` once the data root holds the snapshot and releases:

      uv run --no-sync pytest tests/test_fusion_pinned_releases.py --releases -q -p no:cacheprovider --no-cov </dev/null

## Runs

A run of record keeps its artifacts in its own folder, `runs/<run-id>/` at the repository root,
gitignored like `data/` and present only on the machine that ran it. `<run-id>` is the run's id in
[EVIDENCE.md](EVIDENCE.md), such as `e1-3`; a run that checks or extends another takes that run's
id and a suffix, such as `e1-3-replay` or `c1-2-closure`. A fill made in the Streamlit app is not a
run of record; its folder is named `app-<UTC time>-<arm>`. `bko.app.runs.RUN_ID` matches both
forms. Each artifact takes a fixed role name, so
a reader finds it without knowing the command that wrote it. `bko.app.runs.ROLES` is the list's
home; this table mirrors it, and `tests/test_run_folders.py` holds the two equal:

| Role | What writes it |
|---|---|
| `calls.jsonl` | the call log of a run that calls the Messages API |
| `scores.csv` | `python -m bko.evaluate score`'s stdout |
| `scores.log` | its stderr |
| `criteria.csv` | `python -m bko.evaluate compare`'s stdout |
| `criteria.log` | its stderr |
| `candidates.jsonl` | `python -m bko.generate`'s stdout |
| `candidates.log` | its stderr |
| `closure.csv` | `python -m bko.fusion`'s stdout |
| `closure.log` | its stderr |
| `gaps-uci.csv` | `python -m bko.gaps uci`'s stdout |
| `gaps-uci.log` | its stderr |
| `gaps-synthea.csv` | `python -m bko.gaps synthea`'s stdout |
| `gaps-synthea.log` | its stderr |
| `reproduction.csv` | `python -m bko.ontology reproduce`'s stdout |
| `reproduction.log` | its stderr |
| `training.csv` | the augmented training set of an app fill |
| `pytest` | a `live` test's `--basetemp` directory |

The commands in [Evaluation](#evaluation), [Generation](#generation) and
[Gap closure](#gap-closure) write their folders. A free run of record writes its folder with one of:

    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-uci.csv && test ! -e runs/<run-id>/gaps-uci.log && uv run --no-sync python -m bko.gaps uci > runs/<run-id>/gaps-uci.csv 2> runs/<run-id>/gaps-uci.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/gaps-synthea.csv && test ! -e runs/<run-id>/gaps-synthea.log && uv run --no-sync python -m bko.gaps synthea > runs/<run-id>/gaps-synthea.csv 2> runs/<run-id>/gaps-synthea.log </dev/null
    mkdir -p runs/<run-id> && test ! -e runs/<run-id>/reproduction.csv && test ! -e runs/<run-id>/reproduction.log && uv run --no-sync python -m bko.ontology reproduce > runs/<run-id>/reproduction.csv 2> runs/<run-id>/reproduction.log </dev/null

- **Runs whose rows name `scratchpad/`**: their artifacts are where the row says. The latest run of
  each such entry also has a folder whose files are hard links to those artifacts, with write
  permission removed: `e1-3`, `c1-2`, `c1-2-closure`, `g1-13`, `e2-5` and `r1-2`. A hard link is
  the same file under a second name, sharing its bytes and its mode, so the row's paths, sha256 and
  commands stay true; `c1-2`'s live check, a pytest directory, stays only where its row says. Never
  write into a linked folder or restore a linked file's write permission: a redirect into a
  writable link empties the register's artifact before the command runs.
- **Another machine**: a run shows there only after its folder is copied into that machine's `runs/`.

## App

`bko.app` ([README](../src/bko/app/README.md)) serves a Streamlit page over the stages, with
`make setup`'s `app` extra:

    make app

It serves `http://127.0.0.1:8501` until interrupted. The recipe's flags are the app's whole
configuration: the loopback bind, a fixed port (a second launch fails rather than moving), no
browser and no first-run prompt, no usage statistics, no file watcher, and no rerun that aborts a
running action. Launch it only through `make app`; a bare `streamlit run` drops them.

- **What it does**: Analyze fetches UCI Heart Disease and the releases the field mappings name into
  the data root if absent, splits the records as [Data](#data) says, and lists the count-based
  cells and each definition's gaps as [Gap detection](#gap-detection) does. Fill and score fills
  E1's gap request on that split with the chosen arm, then fits every method of
  `bko.evaluate.experiment.METHOD_ARMS` beside it, CTGAN and TVAE included, and scores them all
  with `bko.evaluate.run.score_holdout`, with no replica fill, so the page shows no fidelity and no
  train-on-synthetic reading. It lists each definition's gaps before and after the added rows join.
  Download offers the augmented training set, in the format `python -m bko.baselines` prints, and
  the split's scores, in the format `python -m bko.evaluate score` writes.
- **What it judges**: nothing. The page shows one split's measures; E1's verdict comes only from
  `python -m bko.evaluate compare` over E1's holdouts ([EVIDENCE.md](EVIDENCE.md#e1-uci-heart-disease-augmentation)).
- **What spends money**: only a live fill by an arm of `bko.evaluate.experiment.BKO_ARMS`. It
  starts only while its confirmation box is ticked, the box clears after each fill, the client is
  built only then, and the fill stops once its call log's spend reaches
  `bko.app.context.SPEND_CEILING`. One action runs at a time in the process. The credential is the
  SDK's own (`ANTHROPIC_API_KEY` or `ant auth login`); a missing one stops the fill at its first
  call, before any spend.
- **Replay**: a BKO fill can instead answer every call from the call log of an earlier fill by the
  same arm, picked from a list, with no API call. It refuses a log whose requests differ from the
  fill's or that holds calls the fill does not make (ADR
  [`research/call-log-replay`](adr/research/call-log-replay.md)), so a log replays only at the
  commit, data and releases it was recorded with.
- **What it writes**: besides the data root, one run folder per fill ([Runs](#runs)),
  `runs/app-<UTC time>-<arm>/`, holding `scores.csv` and `training.csv` and, for a BKO fill, live
  or replayed, `calls.jsonl` in the format [Generation](#generation) describes. A stopped fill
  keeps what it wrote. A closed tab or Ctrl-C does not interrupt a running action; the process
  finishes it first.
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

## Platforms

The gate is POSIX-only: macOS and Linux. The checkers shell out to `git`, and the Makefile recipes
assume a POSIX shell.

**Use an SSH remote.** The checkers run `git` with global config blanked, which also drops any
global credential helper. With an HTTPS remote, `make rebased` cannot authenticate its `ls-remote`
probe, so it passes only in its degraded form and never confirms the trunk is current.

## Make targets

| Target | What it runs | Notes |
|---|---|---|
| `setup` | `uv sync --locked` with the `dev`, `data`, `synth`, `eval`, `ontology`, `llm` and `app` extras and the pinned interpreter | Refuses a stale lock |
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

Every step from `lint` down runs as `uv run --no-sync`: the gate measures the environment
`make setup` built and never re-locks or installs during a run.

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
  script thread is refused too; the test then fails on AppTest's timeout rather than the guard's
  message.
- **A `live` run spends money.** `--live` lifts the guard for each `live` test's body and leaves
  subprocesses unguarded. Never set `ANTHROPIC_LOG=debug`: the SDK then logs prompts.
- **One test reaches the network.** The `check_rebased` recipe case in `tests/test_git_runner.py`
  runs the real probe against `origin` and carries the `integration` marker;
  `-m "not integration"` deselects it.
- **Importing SDV writes outside the repository.** `sdv.single_table` creates SDV's user data
  directory (`platformdirs.user_data_dir("sdv", "sdv-dev")`) and copies its logger configuration
  there on import, so the baselines and evaluation tests touch it. No test reads it.
