---
status: accepted
topic: tooling
created: 2026-10-15
updated: 2026-10-15
---
# GPL-3.0-only for the repository's code, copyright Dartmouth College

## Context

The repository carried no license, so nothing in it could be reused, and `proposal-bkb-port` left
open whether to port pybkb, which is licensed GPL-3.0 (`docs/ARCHITECTURE.md`, pybkb), or to build
the BKB layer clean-room. Copying GPL-3.0 code into a repository under a more permissive license is
not allowed, so the repository's license decides whether the port is possible at all.

## Decision Drivers

* **Must let the code be reused** under a standard license.
* **Must keep porting pybkb possible**, so `proposal-bkb-port` can still choose it.
* **Must name the copyright holder** the team works for.

## Decision

- **The code is licensed GPL-3.0-only.** `LICENSE` is the verbatim GNU General Public License,
  version 3, and `pyproject.toml` declares `license = "GPL-3.0-only"` with `LICENSE` as its license
  file.
- **The copyright holder is Dartmouth College**, stated in `README.md`; the license text itself
  forbids edits, so it carries no copyright line.
- **Dependencies keep their own licenses**; the license covers this repository's code.

## Rationale

GPL-3.0 is pybkb's license, so either path of `proposal-bkb-port` stays open: porting its code is
licensing compatible, and a clean-room build is unaffected. "GPL 3.0" was read as version 3 only
(SPDX `GPL-3.0-only`) rather than version 3 or later.

## Consequences

**Gained**: a reusable, standard license; the pybkb port stays possible.
**Accepted**: anyone distributing a work that includes this code must license it under GPL-3.0
too; changing the license later needs the copyright holder's agreement.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| MIT | Copying GPL-3.0 pybkb code into an MIT repository is not allowed, which would force `proposal-bkb-port` to build clean-room | 2026-10-15 |
| GPL-3.0-or-later | The team asked for GPL 3.0; a later version's terms are not yet known | 2026-10-15 |

## Enforcement

- Owner: `LICENSE`
- Pinned by: `tests/test_repo_gate.py`
- `test_repo_gate.py` asserts that `LICENSE` is tracked, that it is the GPL version 3 text, and that
  `pyproject.toml` declares `GPL-3.0-only` with `LICENSE` as its license file.

## Reversal Conditions

**The license changes only if Dartmouth College agrees and no GPL-licensed code is in the tree**;
this record is then edited.

## Related

- [src layout](src-layout.md)
- [uv, ruff and ty](uv-ruff-ty.md)
