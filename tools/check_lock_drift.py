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

Structural covers every way `uv.lock` fails to read as one version per package, each with its own
message (:class:`LockReadError`), and, the case that matters, **fewer comparisons than
`COMPARISON_FLOOR`**. A lock this module fails to parse into names would compare nothing and exit
clean, which reads exactly like an environment in agreement, so a scan that finds too little is a
failure rather than a pass. For the same reason a `[[package]]` entry it cannot read refuses the
lock rather than being skipped: a skipped entry is a package whose drift goes unchecked.

**A locked package that is not installed is reported and does not fail.** `make setup` installs the
`dev` extra rather than the whole lock, so absence is the ordinary state and treating it as drift
would make this refuse the environment `make setup` produces.

**So this module cannot see a missing extra, and a later step of the gate can.** A tree that has
not re-run `make setup` since an extra joined it passes here, because absence is not drift, and is
refused by `make typecheck`, which `make check` runs after this step: `ty` reports
`unresolved-import` per site, with the file, the line and the interpreter it searched, and
`ty.toml` sets `error-on-warning = true`.
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


class LockReadError(Exception):
    """`uv.lock` could not be read as one version per package; the message names how.

    One type for every cause, because `main` refuses them all the same way, and a message per cause,
    because the operator's next step differs: an unreadable file, a file that is not UTF-8 TOML, a
    lock with no `[[package]]` array, an entry that is not a table or lacks a string `name` or
    `version`, and a package pinned at two versions.
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
        Normalized name to locked version, never empty.

    Raises:
        LockReadError: If the lock cannot be read as one version per package, naming the cause.
    """
    # **`UnicodeDecodeError` is named explicitly, and it has to be.** A lock with invalid UTF-8
    # raises it, and it is a `ValueError`, neither `OSError` nor `TOMLDecodeError`, so without the
    # explicit name it would escape, exit 1 with a traceback, and the Makefile would read that as
    # "drifted". `tomllib.load` calls `b.decode()` itself and lets the error propagate, so handing
    # it bytes does not convert the error. Reading bytes is kept because it is the documented entry
    # point.
    try:
        with lock_path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise LockReadError(f"could not be read ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise LockReadError(f"is not valid UTF-8 ({exc.reason} at byte {exc.start})") from exc
    except tomllib.TOMLDecodeError as exc:
        raise LockReadError(f"is not valid TOML ({exc})") from exc
    packages = document.get("package")
    if not isinstance(packages, list):
        found = "no `package` key" if packages is None else f"a `package` {type(packages).__name__}"
        raise LockReadError(f"holds {found} rather than a `[[package]]` array of tables")
    if not packages:
        raise LockReadError("holds an empty `[[package]]` array, so it pins nothing")
    versions: dict[str, str] = {}
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            raise LockReadError(f"has a `[[package]]` entry at index {index} that is not a table")
        name, locked = package.get("name"), package.get("version")
        if not isinstance(name, str):
            raise LockReadError(f"has a `[[package]]` entry at index {index} with no string `name`")
        if not isinstance(locked, str):
            raise LockReadError(
                f"has a `[[package]]` entry at index {index} (`{name}`) with no string `version`"
            )
        key = normalize(name)
        # **Last-wins would be a silent wrong answer**, so a repeated name refuses the whole lock
        # instead. `uv.lock` can hold two `[[package]]` blocks for one name when a resolution
        # forks on a platform or Python marker, and comparing the installed version against
        # whichever block came last reports drift that is not drift. **No entry count is written
        # here**: `grep -c '^name = ' uv.lock` is the derivation.
        if key in versions and versions[key] != locked:
            raise LockReadError(
                f"pins `{key}` at two versions, {versions[key]} and {locked} (index {index}), and "
                "one installed version cannot be compared against both"
            )
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
        read as one version per package or too few packages were compared to mean anything.
    """
    # `LOCK_PATH` is passed explicitly rather than left to the parameter default, which binds at
    # definition time: a caller that repoints the module attribute, which is how the tests drive
    # the unreadable-lock arms, would otherwise still read the original path, and the arm would
    # pass while testing nothing.
    try:
        locked = locked_versions(LOCK_PATH)
    except LockReadError as exc:
        print(
            f"{LOCK_PATH} {exc}, so it was not compared against this environment. Comparing what "
            "could be read would pass the rest unchecked, and a gate that compares nothing reads "
            "exactly like an environment in agreement, so this is a failure rather than a pass.",
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
