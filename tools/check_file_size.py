"""Refuse a tracked Python file that is too long to hold in one reading.

`ruff` cannot express this: it lints per line and per construct, never per file. So the rule is
custom code, and it runs from `make check` through the `sizes` target (ADR `tooling/file-size-cap`).

**Caps are per tree, first match wins**, and `CAPS` is the one home for the numbers. Tests get a
looser cap because a test file's length tracks the number of cases it holds rather than the number
of ideas, and splitting a suite past the point where its cases are grouped by subject buys nothing a
reader can use.

**`tools/` is under its own cap.** A size checker that exempts itself would do so invisibly.

**Discovery is `git ls-files`, not `rglob`.** `scratchpad/` is gitignored and holds Python;
matching it would need a hand-maintained exclude duplicating both `.gitignore` and `ruff.toml`'s
`exclude`, which is a second copy of a rule. Tracking is also the right semantics on its own: an
untracked stray belongs under no gate until someone commits it.

Exit codes are the contract, because the Makefile is what reads them and nothing else does:

* ``0`` -- every file is within its cap. The success line is the only output, on stdout.
* ``1`` -- at least one file is over. Offenders go to stderr worst-first.
* ``2`` -- the check could not be trusted: `git ls-files` failed, a tracked `.py` matched no rule,
  was a symlink or could not be read, or fewer files were discovered than `DISCOVERY_FLOOR`. The
  reason goes to stderr.

Failure text goes to stderr for the reason `tools/check_lock_drift.py` gives: a failure stays
attributable when the streams are split.

**The floor is not optional.** A pathspec that matches nothing exits 0 and reads exactly like a
clean tree. A floor cannot be a frozen set, because the population grows on nearly every branch;
it is set far enough below the real count to never need editing and far enough above zero to catch
a typo that empties the scan.
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path
from typing import Final

from tools._git import require_toplevel, run


REPO_ROOT: Final = Path(__file__).resolve().parent.parent

CAPS: Final[tuple[tuple[str, int], ...]] = (
    ("tests/", 1000),
    ("src/", 500),
    ("tools/", 500),
)
"""Each tracked directory prefix and the cap that applies inside it, first match wins.

Written down **once**, here. The docs point at this module rather than restating the numbers: a cap
quoted in three files is a cap that will read as three different numbers within a month.

**Nothing mechanically enforces that.** The single home is a convention; keep it by not restating
the numbers.
"""

DISCOVERY_FLOOR: Final = 15
r"""Fewer tracked Python files than this and the scan is presumed broken rather than the tree small.

The gap below the real count is deliberate slack: this exists to catch a scan that matches nothing
or almost nothing, not to track the population. `git ls-files | grep -c "\.py$"` is the way to
read the real figure, because a literal here would be one more number to maintain.

**It does not catch a *partial* shrink**, which is the shape an environment hijack produces, so it
is not the guard for that. `tools._git.require_toplevel` and the stripped `GIT_*` environment are.
"""


class UnmeasuredPathError(Exception):
    """A tracked Python file this check refuses to measure.

    Three causes: no rule in `CAPS` covers it, it is a symlink, or it could not be read. Its own
    type so `main` catches exactly these refusals: a broad `except ValueError` or `except OSError`
    would also catch an unrelated failure and report it under one of these messages, and two
    failures under one message is how a diagnostic stops being one.

    **A path that is not valid UTF-8 is not one of the causes.** `tools._git.run` decodes with
    `surrogateescape`, so such a path arrives as a string that `root / name` turns back into its
    original bytes, and it is measured like any other. Where the filesystem cannot hold the name
    (APFS refuses invalid UTF-8), the file is absent from the working tree and is refused as
    unreadable. Either way the outcome is a measurement or an exit 2, never a traceback.
    """


def _shown(name: str) -> str:
    r"""Render a path for a message, with any byte that is not UTF-8 as a `\x` escape.

    A lone surrogate from :func:`tools._git.run`'s decoding cannot be written to a strict stream,
    so it is escaped here rather than left to whichever error handler the stream has.

    Args:
        name: A repo-relative path as `tracked_python_files` returns it.

    Returns:
        The path, printable on any UTF-8 stream.
    """
    return name.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")


def tracked_python_files(root: Path) -> list[str]:
    """Every tracked `.py` path, repo-relative, in `git ls-files` order.

    Args:
        root: The repository root to list from.

    Returns:
        Repo-relative POSIX paths.

    Raises:
        subprocess.CalledProcessError: If `git ls-files` fails.
        RuntimeError: If the git toplevel is not `root`, which would make every returned path
            relative to a directory this function is not measuring.
        subprocess.TimeoutExpired: If git hangs, from either call.
        OSError: If git cannot be executed at all.
    """
    require_toplevel(root)
    # `-z` separates with NUL, so a path containing a newline cannot forge an entry. It is the only
    # flag here, and that is a consequence of the guard above rather than an oversight:
    # `--full-name` and the root pathspec `-- :/` anchor the names and the search at the toplevel,
    # which answers "cwd is not the toplevel", and `require_toplevel` has already refused that
    # case, so neither could change what this returns.
    listed = run(["ls-files", "-z"], root=root, check=True).stdout
    return [name for name in listed.split("\0") if name.endswith(".py")]


def cap_for(path: str) -> int | None:
    """The cap that applies to one repo-relative path.

    Args:
        path: A repo-relative POSIX path.

    Returns:
        The cap, or ``None`` when no rule covers the path -- which is a structural failure rather
        than a pass, because it means a tracked directory grew without anyone deciding its cap.
    """
    return next((cap for prefix, cap in CAPS if path.startswith(prefix)), None)


def line_count(path: Path) -> int:
    """How many lines a file holds, by `wc -l` semantics.

    Args:
        path: The file to measure.

    Returns:
        The number of newline bytes. **Not** ``len(splitlines())``: the two disagree on a file with
        no trailing newline, where `wc -l` reports one fewer. `ruff format` guarantees the trailing
        newline across this tree so they agree here, which is a fact about the tree rather than
        about the counting.
    """
    return path.read_bytes().count(b"\n")


def offenders(root: Path) -> tuple[list[tuple[str, int, int]], int]:
    """Every tracked Python file over its cap, worst first.

    Args:
        root: The repository root.

    Returns:
        A ``(offenders, scanned)`` pair, where each offender is ``(path, lines, cap)`` sorted by
        how far over it is. `scanned` is returned rather than logged so the caller can apply the
        floor, so the check and its anti-vacuity guard read the same number.

    Raises:
        UnmeasuredPathError: If a tracked `.py` file matches no rule in `CAPS`, is a symlink, or
            cannot be read.
    """
    found: list[tuple[str, int, int]] = []
    paths = tracked_python_files(root)
    for name in paths:
        cap = cap_for(name)
        if cap is None:
            raise UnmeasuredPathError(
                f"{_shown(name)} is a tracked Python file under no rule in CAPS. Add its directory "
                "to CAPS with a cap, rather than letting it grow unmeasured."
            )
        path = root / name
        # A tracked symlink is refused rather than followed. `read_bytes` on one measures whatever
        # it points at: a file outside the tree, or a fifo that never returns.
        if path.is_symlink():
            raise UnmeasuredPathError(
                f"{_shown(name)} is a tracked symlink. Its length is a property of whatever it "
                "points at, which may be outside this repository entirely, so it is refused rather "
                "than measured."
            )
        try:
            lines = line_count(path)
        except OSError as exc:
            raise UnmeasuredPathError(
                f"{_shown(name)} is tracked but could not be read ({exc.strerror or exc}), so its "
                "length is unknown. A file deleted from the working tree without `git rm` is the "
                "usual cause: stage the deletion or restore the file."
            ) from exc
        if lines > cap:
            found.append((name, lines, cap))
    found.sort(key=lambda row: (row[2] - row[1], row[0]))
    return found, len(paths)


def main() -> int:
    """Run the check over this repository and report.

    Returns:
        The process exit code: 0 clean, 1 over cap, 2 structural.
    """
    try:
        found, scanned = offenders(REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        # git's own diagnosis is on stderr and `capture_output` swallows it, so the bare
        # `CalledProcessError` string says only "returned non-zero exit status N".
        print(f"could not list tracked files: {exc}\n{(exc.stderr or '').strip()}", file=sys.stderr)
        return 2
    except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
        # `offenders` turns a failed read into `UnmeasuredPathError`, so an `OSError` here is git
        # failing to run, and the message can say the listing failed.
        print(f"could not list tracked files: {exc}", file=sys.stderr)
        return 2
    except UnmeasuredPathError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if scanned < DISCOVERY_FLOOR:
        print(
            f"only {scanned} tracked Python files were found, below the floor of "
            f"{DISCOVERY_FLOOR}. A scan that matches nothing exits clean and reads exactly like a "
            "tree with no violations, so this is a failure rather than a pass.",
            file=sys.stderr,
        )
        return 2

    if found:
        for name, lines, cap in found:
            print(
                f"{_shown(name)}: {lines} lines (cap {cap}, over by {lines - cap})", file=sys.stderr
            )
        print(f"{len(found)} of {scanned} tracked Python files are over cap.", file=sys.stderr)
        return 1

    print(f"{scanned} tracked Python files checked against {len(CAPS)} rules; none over cap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
