---
status: accepted
topic: tooling
created: 2026-11-24
updated: 2026-11-24
---
# A LoRA stack resolves beside the lock as an `slm` extra, and OLMo 3 7B fine-tunes on this machine

## Context

The direct-method study's LoRA comparison fine-tunes a small open language model as a classifier
of the records each arm produces; `bko.slm` holds the trainer. This record says whether the stack
resolves on Python 3.14 beside every locked package, and what one fine-tune costs here. Both
measurements below are planning measurements, made outside the project environment in a scratch
worktree before `bko.slm` existed.

**Scratch lock, 2026-11-22.** In a detached worktree of `origin/main` at `43523b7`, uv 0.13.8 locked
the project with one more extra, `slm = ["transformers", "peft", "tokenizers", "safetensors",
"accelerate"]`: `Resolved 131 packages`, exit 0. It added transformers 5.24.0, peft 0.22.0,
tokenizers 0.23.3, safetensors 0.8.0, accelerate 1.16.0 and huggingface-hub 2.4.0, with
annotated-doc 0.0.7, hf-xet 1.8.0, markdown-it-py 4.3.0, mdurl 0.1.2, psutil 7.2.2,
regex 2026.11.19, rich 15.0.0, shellingham 1.5.4 and typer 0.29.2. No package already in `uv.lock`
changed version; torch stayed 2.14.1 and numpy 2.5.4. `uv sync --locked --no-build` with every
extra exited 0, building only the project itself, and MPS was available.

**Fine-tune, 2026-11-22 to 2026-11-24.** On an Apple M5 Max (arm64, 128 GiB) on AC power, under
`caffeinate -i -s`, with no sleep in the power log between start and end:

- Training set: E6 holdout 0's `none` training split at f = 0.985, 54,177 records (3,749 positive,
  0.0692), each serialized as one line per feature in `frame.features` order, then the question
  "Question: was this patient readmitted within 30 days of discharge? Answer:" and the answer
  ` yes` or ` no`, one token each (10035 and 912). Loss on the answer token alone. Tokens per row:
  671 to 678, median 673.
- Base model: `allenai/Olmo-3-1025-7B`, revision `a81bae42db3975be1671e27b9c9a56da1a9f980f`, BF16 on
  MPS, no CPU fallback; 7,337,988,096 parameters, 39,976,960 trainable.
- LoRA r 16, alpha 32, dropout 0.05, every linear layer; AdamW 2e-4, 407 warmup steps then linear
  decay; micro-batch 4; one epoch, 13,545 steps; seed 0.
- Wall time: load 3.6 s, tokenize 39.0 s, training 100,762.1 s (about 28.0 h; 1.858 s per row),
  scoring 1,000 test rows 327.2 s before and 459.8 s after; 101,601 s in all. Peak MPS driver
  memory 78,264,778,752 bytes; peak memory footprint 80,354,529,264 bytes. Adapter
  159,974,280 bytes. Final loss 0.100320.
- Load: a 20-step probe ran at 1.153 s per row on a quieter machine. During the run a
  `llama-server` served a 26B model on the same GPU for several hours (seen by `ps` at 15:32 EST
  on 2026-11-23, gone by 19:56), and 1,208 of the 13,545 steps took over 15 s against a median of
  5.08 s. The training rate here therefore lies between 1.153 s per row unloaded (a 20-step
  extrapolation) and 1.858 s per row under that load, about 17 to 28 hours for 54,177 rows.
- Scores on the first 1,000 test rows by rank (104 positive), before and after: AUC-PR 0.1274 and
  0.1433, Brier 0.1300 and 0.0954, mean P(yes) 0.2951 and 0.0602. The fine-tune moved the
  model's probabilities to about the training rate (0.0692), which is most of the Brier change; a
  constant at the test rate scores Brier 0.0932, and the AUC-PR is 0.04 over the 0.104 rate on 104
  positives with no interval. So one epoch shows the recipe runs end to end, not that the
  classifier ranks records well. A planning note, not a result;
  [`docs/EVIDENCE.md`](../../EVIDENCE.md) records none.
- The run's logs are kept outside the repository; the serialized rows, the adapter, its
  checkpoints and the weights, which were fitted on or derived from patient records or are
  re-fetchable by their pinned hashes, were deleted after the run on 2026-11-24.

## Decision Drivers

* **Must not move a locked package**, since every recorded run replays against the lock.
* **Must build nothing from source**, so the environment installs the same everywhere.
* **Must use a base model whose weights, code and training data are all published**, so a reader can
  say what the classifier saw before the study's records.
* **Must load weights only as safetensors**, with no remote code.

## Decision

The lock holds an **`slm` extra of transformers, peft, tokenizers, safetensors and accelerate,
with torch**, which the `synth` extra already locks; `make setup` installs it because `bko.slm`
imports it. `uv.lock` holds the exact versions, and every package the other extras lock is at
the version it holds without the `slm` extra. We fine-tune
**`allenai/Olmo-3-1025-7B` at revision `a81bae42db3975be1671e27b9c9a56da1a9f980f`**: Apache-2.0
(its model card, read at that revision on 2026-11-24), trained on the Dolma 3 mixes its card
links, which the `dolma3_mix-6T-1025` dataset card licenses under ODC-By v1.0 (read the same day),
from the files `bko.slm.weights.OLMO_3_7B` pins by size and sha256 (read from the repository at
that revision on 2026-11-24), as safetensors alone.

## Rationale

The scratch lock shows the stack sits beside the locked torch and numpy without moving either, so
`bko.slm` imports from `src/` in the project environment rather than in a second one. The
transferable figure for the comparison's design (its holdouts, seeds and share mixes, which unit
`slm-lora-day-budget` fixes in ADR `research/slm-comparison-design`) is the rate per
training row, 1.153 to 1.858 s, times the rows each fine-tune trains on: an arm's set is the
training split plus its fill, larger than the 54,177 rows measured here, and a share mix holds the
total at the split's size. The design should budget at the loaded rate. OLMo 3 7B publishes its
weights and code under Apache-2.0 and its training data under ODC-By; its tokenizer gives ` yes` and
` no` one token each, so the classifier reads one position.

## Consequences

**Gained**: a resolved, measured stack and a pinned base model that `bko.slm` adopts.
**Accepted**: a fine-tune of a full split takes 17 to 28 hours here, so the trainer's design runs
few fine-tunes; every clone that runs `make setup` downloads the stack.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Adding the extra to the lock before `bko.slm` imported it | Weighed before the first import: ADR `tooling/uv-ruff-ty` adds an extra when its first import lands | 2026-11-22 |
| `uv pip install` into the project environment | Resolves past the lock, and `make synced` refuses the result | 2026-11-22 |
| A separate hashed environment, as `video/` has | The trainer imports from `src/`, which a separate environment would need installed a second way | 2026-11-22 |
| A smaller model, such as OLMo 2 1B or SmolLM2-360M | The team chose OLMo 3 7B as the one base model to measure; no smaller model was measured here, so unit `slm-lora-day-budget` may still pick one against this rate | 2026-11-22 |
| A model whose training data is not published | A reader could not say what the classifier saw before the study's records | 2026-11-22 |

## Enforcement

- Owner: `pyproject.toml`
- Guard: `bko.slm.weights` verifies every pinned file's size and sha256 before the model loads,
  and loads local files alone, as safetensors, with no remote code; `python -m bko.slm score`
  loads an adapter directory only when it holds the adapter's configuration, its safetensors
  weights and its meta alone, with `use_safetensors`
- Pinned by: `tests/test_repo_gate.py`, which asserts `make setup` installs the `slm` extra;
  `tests/test_slm_weights.py`, which refuses a file whose sha256 differs before the loader runs,
  a file outside the pin and a pickle format, and holds every URL to the pinned revision

## Reversal Conditions

`uv lock` with the extra stops exiting 0, or a re-lock moves a package other extras hold; or the
base model's weights stop shipping as safetensors at the pinned revision.

## Related

- [uv, ruff and ty](uv-ruff-ty.md)
- [Python 3.14, without synthcity](python-314-without-synthcity.md)
- [Video scenes outside the lock](video-scenes-outside-the-lock.md)
- [On-device generators](../research/local-generator-backend.md)
- [LoRA comparison design](../research/slm-comparison-design.md)
