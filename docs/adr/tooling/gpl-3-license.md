---
status: accepted
topic: tooling
created: 2026-10-15
updated: 2026-10-29
---
# GPL-3.0-only for the repository's code, copyright Dartmouth College

## Context

The repository carried no license, so nothing in it could be reused. pybkb, whose core `bko.bkb`
ports, is licensed GPL-3.0 (`docs/ARCHITECTURE.md`, pybkb). Copying GPL-3.0 code into a repository
under a more permissive license is not allowed, so the repository's license decides whether the port
is possible at all.

## Decision Drivers

* **Must let the code be reused** under a standard license.
* **Must keep porting pybkb possible**, since `bko.bkb` adapts its code (ADR `research/bkb-layer-source`).
* **Must name the copyright holder** the team works for.

## Decision

- **The code is licensed GPL-3.0-only.** `LICENSE` is the verbatim GNU General Public License,
  version 3, and `pyproject.toml` declares `license = "GPL-3.0-only"` with `LICENSE` as its license
  file.
- **The copyright holder is Dartmouth College**, stated in `README.md`, for the repository's own
  contributions; the license text itself forbids edits, so it carries no copyright line. Code adapted
  from pybkb keeps its author's attribution in each adapted file (ADR `research/bkb-layer-source`).
- **Dependencies keep their own licenses**; the license covers this repository's code.
- **Five locked dependencies are BUSL-1.1**, each with `License-Expression: BUSL-1.1` in its
  installed metadata: `sdv`, `copulas`, `ctgan`, `rdt` and `deepecho`. The `synth` extra brings
  them through SDV, and the `eval` extra brings `copulas` through `sdmetrics`, which is MIT. Each
  grants use of the work and its derivatives except for a commercial service that lets third
  parties use its features: a synthetic data service (`sdv`), a synthetic data creation service
  (`copulas`, `ctgan`, `deepecho`) or a data transform service (`rdt`). Each changes to the MIT
  License four years after its release. This repository uses them for non-commercial research,
  distributes no work that combines them, and serves its app on `127.0.0.1` only.

## Rationale

GPL-3.0 is pybkb's license, so porting its code is licensing compatible (ADR `research/bkb-layer-source`). The team asked for GPL version 3, read as
version 3 only (SPDX `GPL-3.0-only`) rather than version 3 or later.

## Consequences

**Gained**: a reusable, standard license; the pybkb port is licensing compatible.
**Accepted**: anyone distributing a work that includes this code must license it under GPL-3.0
too; changing the license later needs the copyright holder's agreement; the BUSL-1.1 dependencies
rule out offering the baselines or the app to third parties as a commercial service before their
change dates.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| MIT | Copying GPL-3.0 pybkb code into an MIT repository is not allowed, which would force the BKB layer to be built clean-room | 2026-10-15 |
| GPL-3.0-or-later | The team asked for GPL 3.0; a later version's terms are not yet known | 2026-10-15 |

## Enforcement

- Owner: `LICENSE`
- Pinned by: `tests/test_repo_gate.py`
- `test_repo_gate.py` asserts that `LICENSE` is tracked, that it is the GPL version 3 text, and that
  `pyproject.toml` declares `GPL-3.0-only` with `LICENSE` as its license file.

## Reversal Conditions

**The license changes only if Dartmouth College agrees and no code adapted from a GPL-licensed work, pybkb's included, is in the tree**;
this record is then edited.

**The BUSL-1.1 reading is revisited if the app or the baselines are hosted for third parties, a
work combining this code with a listed package is distributed (a release archive or container
image), or a lock bump changes any listed package's `License-Expression`.**

## Related

- [src layout](src-layout.md)
- [uv, ruff and ty](uv-ruff-ty.md)
- [The BKB layer is pybkb's core](../research/bkb-layer-source.md)
