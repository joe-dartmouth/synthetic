---
status: accepted
topic: research
created: 2026-11-22
updated: 2026-11-22
---
# TabDDPM is an in-repo baseline on the locked torch, at the authors' base configuration with its steps fixed by validation loss

## Context

Reviewers expect a diffusion synthesizer among the comparators. synthcity, which ships one, does not
resolve on Python 3.14 (ADR
[`tooling/python-314-without-synthcity`](../tooling/python-314-without-synthcity.md)); torch is
already in the lock for CTGAN and TVAE. TabDDPM models a row's measured fields with Gaussian
diffusion and each coded field with multinomial diffusion, one MLP denoising both
([`kotelnikov-2023`](../../literature.md#tabular-generators)).

Every measurement below is a planning measurement: measured 2026-11-22 at commit `ad69381` on an
Apple M5 Max (arm64), torch 2.14.1 and scikit-learn 1.9.1, on one thread under
`VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1`, on E6 holdout 0's `none` training split at
f = 0.985 (54,177 rows), split 90/10 stratified by label with seed 0 (5,418 validation rows, none
dropped for a code the fit split lacks). Validation loss is `validation_loss` with 8 draws of every
row's step and noise from `derived_seed(0, "tabddpm-validation")`; each grid point is a separate
fit, since the learning rate anneals to its step count.

| Steps | Validation loss | Fit seconds |
|---|---|---|
| 1000 | 0.682270 | 23.3 |
| 5000 | 0.639078 | 113.7 |
| 10000 | 0.626537 | 227.5 |
| 20000 | 0.617005 | 454.7 |
| 30000 | 0.612465 | 682.3 |

At 20000 steps a fit on the whole split took 435.9 seconds and one draw of 10,000 rows
(`DRAW_BATCH`) took 23.1 seconds. An E6 fill draws to `MAX_DRAWN_ROWS`, since the removed cell is
19 of the split's 54,177 rows at f = 0.985 and rows are kept only by their own values, so one fill
is 100 draws: a projection, not a measurement, of about 2,742 seconds per holdout and fraction.

## Decision Drivers

* **Must add no dependency**, and run on the locked torch.
* **Must honour the baselines' contract**: rows kept by their own values through `fill()`, one
  seed giving equal rows on one machine at one thread, a shortfall handed back as a partial fill.
* **Must read the published method**, at its authors' configuration rather than a tuning of ours.
* **Must fix the one quantity the authors tune per dataset, the steps, once and before any run**.

## Decision

We will implement **TabDDPM in `bko.baselines.ddpm` (the diffusion mathematics) and
`bko.baselines.diffusion` (the runner) behind `Method.TABDDPM`**, at these values, which
`TABDDPM_PARAMETERS` holds:

- 20000 steps, chosen as the smallest grid point whose validation loss is within 1% of the grid's
  lowest (Context);
- 1000 diffusion timesteps on the cosine schedule, betas clipped at 0.999;
- an MLP denoiser of two 256-unit layers, a 128-wide sinusoidal time embedding, a label embedding
  through SiLU, dropout 0;
- AdamW at learning rate 0.001 and weight decay 1e-05, the rate set to `lr * (1 - step / steps)`
  after each update; batches of 4096 rows, each epoch a fresh permutation; no EMA;
- measured fields through a quantile transform to a normal, `n // 30` quantiles held between 10
  and 1000 and never above `n`;
- coded fields with the states the training split shows, in codebook order; the label conditions
  the denoiser and is drawn at sampling time from the training label counts.

The method comes from the paper (arXiv:2209.15421). The base configuration, AdamW, the linear
anneal, the label embedding through SiLU, sampling from the plain model rather than an EMA, the
quantile count and the loader come from the authors' repository `yandex-research/tab-ddpm` at
commit `b476257dd460b778ba09eb97f7a51d6490fa17f8`: `exp/adult/config.toml` reads
`d_layers = [256, 256]`, `dropout = 0.0`, `num_timesteps = 1000`, `gaussian_loss_type = "mse"`,
`scheduler = "cosine"`, `steps = 1000`, `lr = 0.001`, `weight_decay = 1e-05`, `batch_size = 4096`
and `normalization = "quantile"`, `tab_ddpm/modules.py` sets `dim_t = 128`, and `LICENSE.md` opens
"MIT License". The multinomial part is written from the paper's equations and Hoogeboom et al.'s
multinomial diffusion, which it cites. Its training term is the authors' objective: each step's KL,
or the decoder term at the first step, times the number of timesteps, plus the prior KL, summed
over coded fields and divided by their count. Sampling uses each step's beta as its variance, as the
authors' code does.

## Rationale

Writing it in-repo on the locked torch is the only way to have a diffusion baseline without a
second environment, since synthcity does not resolve. Taking
the authors' base configuration keeps the baseline theirs rather than ours; the one value they
tune per dataset, the steps, is fixed by the same rule E7 used for the perceptron's epochs, once,
on holdout 0's split, so no later run tunes it against its own results. The objective's weighting
matches the authors' code, verified at the pinned commit (`kl / pt + kl_prior` under uniform steps),
because reading the paper's equation as one step's term leaves the coded fields' gradient about a
thousand times weaker than the measured fields'. Labels drawn from the training counts, with rows
kept by their own values, are how the authors sample and how CTGAN and TVAE fill here.

## Consequences

**Gained**: a diffusion comparator for every frame, seeded and filled as the other generators are,
with no new dependency.
**Accepted**: an E6 fill is projected at about 46 minutes per holdout and fraction, mostly drawing,
because a thin cell is reached by rejection; the steps were fixed on one split of one frame and
are used on every frame.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| synthcity's TabDDPM | synthcity does not resolve on Python 3.14 (ADR `tooling/python-314-without-synthcity`) | 2026-11-22 |
| Copying the authors' multinomial diffusion file | It derives from a repository with no licence; the part is written from the equations | 2026-11-22 |
| A tuned configuration per dataset | The authors tune per dataset with Optuna; a search here would be a tuning of ours, run against our own results | 2026-11-22 |
| Drawing labels at the quota's shares | Not how the authors sample; rows are kept by their own values, as every generator here keeps them | 2026-11-22 |
| Early stopping on validation loss | A fit's length would move with its seed and data; the steps are fixed once instead | 2026-11-22 |
| Training on MPS | The baselines run on the CPU at one thread so a seed gives equal rows | 2026-11-22 |
| TabSyn | A latent-diffusion successor with its own VAE; the IDEAS entry keeps it | 2026-11-22 |

## Enforcement

- Owner: `src/bko/baselines/diffusion.py`, `src/bko/baselines/ddpm.py`
- Pinned by: `tests/test_baselines_diffusion.py`, `tests/test_baselines_ddpm.py`,
  `tests/test_baselines_methods.py`

## Reversal Conditions

The lowest-validation-loss rule picks another grid point when rerun under the locked torch on the
same split; or synthcity resolves on the project's Python, so the published implementation can
replace this one.

## Related

- [Baseline augmentation design](baseline-augmentation-design.md)
- [Classifier panel](classifier-panel.md)
- [Python 3.14, without synthcity](../tooling/python-314-without-synthcity.md)
