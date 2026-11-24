---
status: proposed
topic: tooling
created: 2026-11-24
updated: 2026-11-24
---
# A LoRA stack resolves beside the lock as an `slm` extra, and OLMo 3 7B fine-tunes on this machine

## Context

The direct-method study's LoRA comparison fine-tunes a small open language model as a classifier
of the records each arm produces. The lock holds no transformers or peft. This record is the spike
that unit `slm-lora-trainer` builds on: whether the stack resolves on Python 3.14 beside every
locked package, and what one fine-tune costs here. Both measurements below are planning
measurements, made outside the project environment in a scratch worktree, with nothing committed to
`pyproject.toml`, `uv.lock` or `src/`.

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
- Load: a 20-step probe ran at 1.153 s per row on a quieter machine. During the run another
  process served a 26B model on the same GPU for several hours, and 1,208 of 13,551 logged steps
  took over 15 s against a usual 4.5 to 6 s, so the wall time above is an upper bound for this
  machine, not its unloaded speed.
- Check that it learned the question it will be asked, on the first 1,000 test rows by rank (104
  positive): AUC-PR 0.1274 and Brier 0.1300 before, 0.1433 and 0.0954 after. A planning note, not a
  result; [`docs/EVIDENCE.md`](../../EVIDENCE.md) records none.

## Decision Drivers

* **Must not move a locked package**, since every recorded run replays against the lock.
* **Must build nothing from source**, so the environment installs the same everywhere.
* **Must use a base model whose weights, code and training data are all published**, so a reader can
  say what the classifier saw before the study's records.
* **Must load weights only as safetensors**, with no remote code.

## Decision

We will add, when unit `slm-lora-trainer` first imports them, an **`slm` extra holding
transformers 5.24.0, peft 0.22.0, tokenizers 0.23.3, safetensors 0.8.0 and accelerate 1.16.0**
(with huggingface-hub 2.4.0 and the packages above that the lock pulls in), and fine-tune
**`allenai/Olmo-3-1025-7B` at revision `a81bae42db3975be1671e27b9c9a56da1a9f980f`**: Apache-2.0
(its model card, read at that revision), trained on the Dolma 3 mixes its card links, which
the `dolma3_mix-6T-1025` dataset card licenses under ODC-By v1.0, loaded from these files only:

| File | Bytes | sha256 |
|---|---|---|
| `config.json` | 1620 | `277d7ea5c6159d4c2d6986a8a23757d1ef06acaa7abc2c40e0513446f7db0f85` |
| `generation_config.json` | 69 | `52e8c2bef1d50a9aa99ce1da2a586ec40f524c1eded9c9ece289f3146b909ded` |
| `merges.txt` | 916646 | `b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502` |
| `model-00001-of-00003.safetensors` | 4969984976 | `0490d6668e613a29b23367e3a7aa9cc6aced3d162694445bb969ed7622b3c4e2` |
| `model-00002-of-00003.safetensors` | 4981161496 | `e127ea479fb6e208fe9d48d23b11212b5722f4873f6eef9c009b7a855866c641` |
| `model-00003-of-00003.safetensors` | 4644917240 | `f3ddff10052ffe5de5c6b4cad45c422c0d898acc6beb21b1b8531244adfb3c70` |
| `model.safetensors.index.json` | 29630 | `47a8266860f2a5f95c021fb4d7d3a469a95ee8b445742becbb6b19632cfc6e95` |
| `special_tokens_map.json` | 207 | `4cb2b52960bababa8ec27153831e405ef7811e3fad1226106e09c28e715cfb21` |
| `tokenizer_config.json` | 4308 | `ffa5b2cfee6e73d5dc0a7d3b7e02eb6ed60a491dd782f2bd8ad2b4659b2aafc6` |
| `tokenizer.json` | 7137177 | `73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca` |
| `vocab.json` | 1611056 | `9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195` |

## Rationale

The scratch lock shows the stack sits beside the locked torch and numpy without moving either, so
the trainer can import from `src/` in the project environment rather than in a second one. The
fine-tune shows one epoch over a full E6 training split fits on this machine in about a day under
load, which is what the trainer's design (its holdouts, seeds and share mixes) is chosen against.
OLMo 3 7B is the newest release whose weights, code and training data are all open, under
Apache-2.0 and ODC-By; its tokenizer gives ` yes` and ` no` one token each, so the classifier reads
one position. Staying `proposed` until the trainer lands keeps the lock unchanged by a decision no
code uses yet.

## Consequences

**Gained**: a resolved, measured stack and a pinned base model for the trainer to adopt.
**Accepted**: a fine-tune of a full split takes about a day here, so the trainer's design runs few
fine-tunes; the extra adds 15 packages to the lock when it lands.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Adding the extra to the lock now | No code imports it yet; ADR `tooling/uv-ruff-ty` adds an extra when its first import lands | 2026-11-22 |
| `uv pip install` into the project environment | Resolves past the lock, and `make synced` refuses the result | 2026-11-22 |
| A separate hashed environment, as `video/` has | The trainer imports from `src/`, which a separate environment would need installed a second way | 2026-11-22 |
| OLMo 2 1B or SmolLM2-360M | Smaller, but an older family or one whose training mixture is only partly published | 2026-11-22 |
| A model whose training data is not published | A reader could not say what the classifier saw before the study's records | 2026-11-22 |

## Enforcement

- Owner: `pyproject.toml`
- Pinned by: nothing -- convention only

## Reversal Conditions

`uv lock` with the extra stops exiting 0 under the locked torch; or the base model's weights stop
shipping as safetensors at the pinned revision.

## Related

- [uv, ruff and ty](uv-ruff-ty.md)
- [Python 3.14, without synthcity](python-314-without-synthcity.md)
- [Video scenes outside the lock](video-scenes-outside-the-lock.md)
- [On-device generators](../research/local-generator-backend.md)
