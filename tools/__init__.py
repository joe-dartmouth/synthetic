"""Repository tooling that is not part of the `bko` package.

A top-level `tools` rather than a module under `src/bko/`, because nothing here is imported by the
library and a size checker that shipped inside the thing it measures would be measuring itself for
the wrong reason. It is a package so `tests/test_file_size.py` can import the predicate and drive
it directly rather than only through a subprocess, and so the checkers can share one `git` runner
in :mod:`tools._git`.

**Every checker here must be run as `python -m tools.<name>`, not as a path.** Running the file
directly puts `tools/` itself on `sys.path[0]` rather than the repository root, so
`from tools._git import ...` raises `ModuleNotFoundError` and the gate dies at that step. Only two
of the three import the runner -- `check_lock_drift.py` does not -- but all three recipes carry the
module form, because a checker that works as a path only until it needs the runner is a trap rather
than an exception. `tests/test_git_runner.py` runs each recipe as the Makefile spells it.
"""
