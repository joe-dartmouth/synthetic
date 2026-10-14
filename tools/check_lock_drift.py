"""Refuse an environment whose installed packages disagree with `uv.lock`.

There is no CI: `make check`, run by hand, is the whole gate. So the gate's own environment is
load-bearing, and it can drift from the lock after it is created. The failure it produces reads
as a repository defect rather than an environment one: `make lint` and `make typecheck` both report
errors that belong to newer tools disagreeing with a tree that is fine. `make setup` installs from
the lock (`uv sync --frozen`), but a lock-faithful setup says nothing about what the environment
looks like a month later.

**No drift figure is quoted.** The quantity is unbounded and machine-local. Run this script for the
current number.

**Why this is not `uv sync --check`.** That command answers "would `uv sync` change this
environment", and `uv sync` builds an environment this project does not use: bare `uv sync --check`
reports the `dev` extra as *removable*, because extras are not default groups, and
`uv sync --all-extras --check` reports packages missing, because it wants every extra, which
`make setup` deliberately does not install. Both forms fail on every correctly configured machine,
so neither can gate. The check has to be the narrower one: of the packages that *are* installed,
does any disagree with the lock.

**Three exit codes**, on the same contract `tools/check_file_size.py` shares: `0` clean, `1`
drifted, `2` structural. Make reads only zero against non-zero; the `1`/`2` split is for the
operator and for `tests/test_repo_checkers.py`. Failure text goes to **stderr** and only the success
line to stdout, so a failure stays attributable when the streams are split.

Structural covers an unreadable or unparseable `uv.lock` and, the case that matters, **fewer
comparisons than `COMPARISON_FLOOR`**. A lock this module fails to parse into names would compare
nothing and exit clean, which reads exactly like an environment in agreement, so a scan that finds
too little is a failure rather than a pass.

**A locked package that is not installed is reported and does not fail.** `make setup` installs the
`dev` extra rather than the whole lock, so absence is the ordinary state and treating it as drift
would make this refuse the environment `make setup` produces.

**So this module cannot see a missing extra, and the step after it can.** A tree that has not
re-run `make setup` since an extra joined it passes here, because absence is not drift, and is
refused by `make typecheck` one step later: `ty` reports `unresolved-import` per site, with the
file, the line and the interpreter it searched, and `ty.toml` sets `error-on-warning = true`.
"""

from __future__ import annotations

import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final


LOCK_PATH: Final = Path(__file__).resolve().parent.parent / "uv.lock"

REMEDIATION: Final = "make setup"
"""The command that reconciles this environment to the lock.

**Not `make sync`.** `make sync` is `uv sync --frozen --all-extras`, so on a correctly configured
tree it installs every extra on top of what the gate needs. The prescribed fix for a small drift
would then change the shape of the environment the gate should have, and invalidate this module's
own success line about why the absent packages are absent. The remediation must not widen the
environment past what the gate declares.

`make setup` is the right command because it installs *from the lock* (`uv sync --frozen` over
the extras the gate needs), which lands on exactly the package set this module compares. **No
package count is written here**: this module prints the current one on every successful run.
"""

COMPARISON_FLOOR: Final = 6
"""Fewest installed-and-locked packages a healthy run may compare.

The floor sits well below the count a `make setup` environment compares, because that count moves
whenever an extra is added or the lock is relocked. It is chosen against the failure it exists to
catch: a **lock-parsing** regression. If :func:`locked_versions` stops producing names this module
can look up, the comparison count goes to nearly zero rather than to a dozen, so any floor in this
band separates the two cases. Against a bare interpreter with nothing installed, 0 are compared and
the exit is 2.

It is deliberately *not* justified by a name-normalization bug: `importlib.metadata.version`
normalizes the name it is handed, so a `pyyaml` against `PyYAML` mismatch cannot arise through this
API. Re-derive the current count from this script's own success line rather than from a figure
written here.
"""


def normalize(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name.

    This is `packaging.utils.canonicalize_name` verbatim, reimplemented rather than imported
    because `packaging` is only present transitively and `pyproject.toml`'s convention is that a
    directly-imported module must be declared. One line beats a dependency declaration for a
    `tools/` script.

    **It is for the lock side, not the lookup side.** `importlib.metadata.version` already
    normalizes what it is handed, so `version("PyYAML")` and `version("pyyaml")` both resolve; the
    work here is making the lock's keys unique and comparable so two spellings of one package
    cannot become two entries.

    Args:
        name: A distribution name in any spelling.

    Returns:
        The lowercased name with runs of `-`, `_` and `.` collapsed to a single `-`.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(lock_path: Path = LOCK_PATH) -> dict[str, str]:
    """Read every package the lock pins, keyed by normalized name.

    Args:
        lock_path: The `uv.lock` to read.

    Returns:
        Normalized name to locked version. Empty when the file cannot be read or parsed, which the
        caller reports as structural rather than clean.
    """
    # **`UnicodeDecodeError` is named explicitly, and it has to be.** A lock with invalid UTF-8
    # raises it, and it is a `ValueError`, neither `OSError` nor `TOMLDecodeError`, so without the
    # explicit name it would escape this handler, exit 1 with a traceback, and the Makefile would
    # read that as "drifted". `tomllib.load` calls `b.decode()` itself and lets the error
    # propagate, so handing it bytes does not convert the error. Reading bytes is kept because it
    # is the documented entry point.
    try:
        with lock_path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError, tomllib.TOMLDecodeError, UnicodeDecodeError:
        return {}
    packages = document.get("package")
    if not isinstance(packages, list):
        return {}
    versions: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            continue
        name, locked = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(locked, str):
            continue
        key = normalize(name)
        # **Last-wins would be a silent wrong answer**, so a repeated name refuses the whole lock
        # instead. `uv.lock` can hold two `[[package]]` blocks for one name when a resolution
        # forks on a platform or Python marker, and comparing the installed version against
        # whichever block came last reports drift that is not drift. **No entry count is written
        # here**: `grep -c '^name = ' uv.lock` is the derivation.
        if key in versions and versions[key] != locked:
            return {}
        versions[key] = locked
    return versions


def drift(locked: dict[str, str]) -> tuple[list[tuple[str, str, str]], int, int]:
    """Compare each locked package against what is installed.

    Args:
        locked: Normalized name to locked version, from :func:`locked_versions`.

    Returns:
        The disagreements as `(name, locked, installed)`, how many packages were compared, and how
        many were locked but absent.
    """
    drifted: list[tuple[str, str, str]] = []
    compared = absent = 0
    for name, want in sorted(locked.items()):
        try:
            have = version(name)
        except PackageNotFoundError:
            absent += 1
            continue
        compared += 1
        if have != want:
            drifted.append((name, want, have))
    return drifted, compared, absent


def main() -> int:
    """Report drift between the installed environment and `uv.lock`.

    Returns:
        0 when every compared package agrees, 1 when any disagrees, 2 when the lock could not be
        read or too few packages were compared to mean anything.
    """
    # `LOCK_PATH` is passed explicitly rather than left to the parameter default, which binds at
    # definition time: a caller that repoints the module attribute, which is how the tests drive
    # the unreadable-lock arm, would otherwise still read the original path, and the arm would
    # pass while testing nothing.
    locked = locked_versions(LOCK_PATH)
    if not locked:
        print(
            f"{LOCK_PATH} could not be read as a uv lock file holding one version per package, so "
            "nothing was compared. A gate that compares nothing exits clean and reads exactly "
            "like an environment in agreement, so this is a failure rather than a pass.",
            file=sys.stderr,
        )
        return 2

    drifted, compared, absent = drift(locked)

    if compared < COMPARISON_FLOOR:
        print(
            f"only {compared} of {len(locked)} locked packages were found installed, below the "
            f"floor of {COMPARISON_FLOOR}. Either this environment was never set up -- run "
            "`make setup` -- or this module stopped parsing the lock into names it can look up, "
            "which would pass every version through unchecked.",
            file=sys.stderr,
        )
        return 2

    if drifted:
        for name, want, have in drifted:
            print(f"{name}: lock pins {want}, environment has {have}", file=sys.stderr)
        print(
            f"{len(drifted)} of {compared} compared packages have drifted from uv.lock. `make "
            "check` measures the tree with these versions, so a failure it reports may belong to "
            f"the environment rather than to the repository. Run `{REMEDIATION}`.",
            file=sys.stderr,
        )
        return 1

    print(
        f"{compared} of {len(locked)} locked packages are installed and agree with uv.lock "
        f"({absent} locked but not installed: extras this environment did not install, plus "
        "platform-gated wheels. Absence is counted, not checked)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
