"""Guards on `tools/_git.py`, the one `git` runner the repository's own checkers share.

Each property here is a defence in the runner that the checkers' own tests would not notice
losing, because those tests exercise the checkers rather than the runner. The controls are named
per test. The process-group cases guard the session and the group kill that stop a timed-out git
from orphaning its `ssh` helper.

**The `-m` invocation is pinned here too**, at the bottom. `tools/` is an importable package, so
`python tools/check_file_size.py` puts `tools/` itself on `sys.path[0]` and
`from tools._git import ...` then fails. The Makefile recipes carry the `python -m` form, and the
last cases here run each one as the Makefile spells it.
"""

from __future__ import annotations

import ast
import contextlib
import itertools
import operator
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests._corpus import REPO_ROOT
from tools._git import _end_group, child_environment, require_toplevel, run, toplevel_of


EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
"""`git hash-object` over zero bytes, which is what a closed stdin makes it read."""

_NO_SUCH_REF = ("rev-parse", "--verify", "refs/heads/no-such-branch-exists-here")
"""A call git answers with a non-zero status.

Deliberately not a malformed *option*: `git rev-parse --not-an-option` exits 0 and echoes the word
back, so it discriminates neither half of `check`. A tuple rather than a list, which also exercises
the read-only `Sequence` the runner accepts.
"""


def _init(path: Path) -> None:
    """Make `path` a git checkout, with no commits and no dependence on the operator's config."""
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["git", "init", "-q", "-b", "main", str(path)],
        check=True,
        capture_output=True,
        text=True,
        env=child_environment(),
        stdin=subprocess.DEVNULL,
        timeout=30,
    )


def test_the_child_environment_strips_every_git_variable(monkeypatch):
    """`GIT_DIR` and its siblings outrank `cwd`, so a set one makes git answer about another tree.

    Asserted over the prefix rather than over the three names the runner's docstring lists:
    `GIT_ALTERNATE_OBJECT_DIRECTORIES` and `GIT_OBJECT_DIRECTORY` redirect a read just as well, and
    a guard naming three would pass while a fourth got through.
    """
    monkeypatch.setenv("GIT_DIR", "/nowhere/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/nowhere")
    monkeypatch.setenv("GIT_INDEX_FILE", "/nowhere/index")
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", "/nowhere/objects")

    survivors = [key for key in child_environment() if key.startswith("GIT_")]

    # The scrub removes every `GIT_*` the *operator* set; the runner then adds back the ones it
    # chooses, which is why this compares against a declared set rather than an empty one: two
    # blank the foreign config files, two turn credential prompting off, and one stops a partial
    # clone fetching objects. Asserted as a set so a new addition has to be declared here, and none
    # of the variables planted above may appear.
    assert set(survivors) == {
        "GIT_ASKPASS",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_NO_LAZY_FETCH",
        "GIT_TERMINAL_PROMPT",
    }, f"an undeclared `GIT_*` variable reached the child: {survivors}"
    assert not [key for key in survivors if "/nowhere" in child_environment()[key]], (
        "and no planted value survived under any name"
    )


def test_no_askpass_program_runs_under_the_child_environment(tmp_path, monkeypatch):
    """A credential request must fail rather than run an askpass program.

    `GIT_TERMINAL_PROMPT=0` stops git prompting on the terminal and does not stop it running an
    askpass program, which on a desktop is a dialog popping up during `make check`. `git credential
    fill` asks exactly as a remote needing a password would, so it drives the same path offline.
    The control drops the runner's `GIT_ASKPASS` and must see the program run, or this test could
    pass on a git that never asks.
    """
    marker = tmp_path / "asked"
    askpass = tmp_path / "askpass.sh"
    askpass.write_text(f"#!/bin/sh\necho asked >> {marker}\necho x\n", encoding="utf-8")
    askpass.chmod(0o700)
    monkeypatch.setenv("SSH_ASKPASS", str(askpass))
    monkeypatch.setenv("GIT_ASKPASS", str(askpass))
    request = "protocol=https\nhost=example.invalid\n\n"

    def fill(env):
        return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            ["git", "-c", f"core.askPass={askpass}", "credential", "fill"],
            cwd=tmp_path,
            input=request,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=env,
        )

    scrubbed = child_environment()
    assert scrubbed["SSH_ASKPASS_REQUIRE"] == "never"
    fill(scrubbed)
    assert not marker.exists(), "an askpass program ran under the runner's environment"

    control = {key: value for key, value in scrubbed.items() if key != "GIT_ASKPASS"}
    fill(control)
    assert marker.exists(), "the control never asked, so the assertion above proves nothing"


def test_the_child_environment_passes_the_rest_of_the_environment_through(monkeypatch):
    """The scrub is a filter, not a fresh environment.

    Without this, blanking the whole mapping passes the test above. git needs `PATH` to be found at
    all and `HOME` for its own bookkeeping, so an over-broad scrub breaks every call rather than
    hardening it, but it breaks it loudly, which is why this is the cheaper of the two guards and
    not the more important one.
    """
    monkeypatch.setenv("BKO_NOT_A_GIT_VARIABLE", "kept")

    passed = child_environment()

    assert passed["BKO_NOT_A_GIT_VARIABLE"] == "kept"
    assert passed["PATH"] == os.environ["PATH"]


def test_a_global_config_cannot_make_a_read_execute_a_program(tmp_path):
    """The code-execution defence, with the positive control that makes it load-bearing.

    `core.fsmonitor` names a program git runs before it answers, and stripping `GIT_*` does not
    stop git reading `~/.gitconfig`. The second half of this test is the control: with
    `GIT_CONFIG_GLOBAL` removed from the same environment, the same call **does** run the script,
    so the assertion above it is measuring the blanking rather than a git that would not have run
    anything anyway.
    """
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    _init(repo)
    sentinel = tmp_path / "ran"
    hook = tmp_path / "fsmonitor.sh"
    hook.write_text(f'#!/bin/sh\ntouch "{sentinel}"\nprintf "/"\n')
    hook.chmod(0o755)
    (home / ".gitconfig").write_text(f"[core]\n\tfsmonitor = {hook}\n")

    scrubbed = {**child_environment(), "HOME": str(home)}
    subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=scrubbed,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert not sentinel.exists(), (
        "a global config named a program and git ran it inside a checker's own call"
    )

    exposed = {key: value for key, value in scrubbed.items() if key != "GIT_CONFIG_GLOBAL"}
    subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=exposed,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert sentinel.exists(), (
        "the control did not fire, so this git ignores `core.fsmonitor` and the assertion above "
        "proves nothing about the blanking"
    )


def test_the_runner_closes_stdin():
    """A git that reads stdin gets EOF rather than whatever the gate was launched with.

    `hash-object --stdin` is the cheapest observable: with `stdin` closed it hashes zero bytes and
    prints the empty blob. Inheriting the parent's descriptor makes the answer depend on how the
    suite was invoked, and a git that decides to *prompt*, for a credential or a pager, blocks until
    the timeout instead.
    """
    finished = run(["hash-object", "--stdin"], root=REPO_ROOT, check=True)

    assert finished.stdout.strip() == EMPTY_BLOB


def test_the_runner_decodes_output_that_is_not_utf8_rather_than_raising(tmp_path):
    """A byte that is not UTF-8 comes back as a lone surrogate, and encodes back to itself.

    A strict decode raises `UnicodeDecodeError` out of :func:`tools._git.run`, a traceback that
    exits 1, which `check_file_size` would report as a file over its cap and `check_rebased` as a
    branch that is behind. The blob holds bytes that open no valid UTF-8 sequence, so `cat-file`
    prints them verbatim; the round trip is what lets a decoded path still name its file.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init(repo)
    raw = b"\xff\xfe not utf-8\n"
    (repo / "blob.bin").write_bytes(raw)
    blob = run(["hash-object", "-w", "--", "blob.bin"], root=repo, check=True).stdout.strip()

    shown = run(["cat-file", "blob", blob], root=repo, check=True).stdout

    assert shown.startswith("\udcff\udcfe"), f"the bytes did not decode to surrogates: {shown!r}"
    assert shown.encode("utf-8", "surrogateescape") == raw


def test_the_runner_raises_on_a_failing_command_when_the_caller_asks():
    """`check=True` is forwarded, which `check_file_size.py` reads to notice a failed listing."""
    with pytest.raises(subprocess.CalledProcessError):
        run(_NO_SUCH_REF, root=REPO_ROOT, check=True)


def test_the_runner_returns_the_status_when_the_caller_does_not_ask():
    """`check=False` is the other half, and `check_rebased.py` reads the status itself.

    Without this the parameter could be ignored and hard-wired to `True`; the test above alone does
    not discriminate.
    """
    finished = run(_NO_SUCH_REF, root=REPO_ROOT, check=False)

    assert finished.returncode != 0


def test_the_runner_answers_about_the_directory_it_was_given(tmp_path):
    """`cwd` is the root, so two calls in one process can address two repositories."""
    other = tmp_path / "other"
    other.mkdir()
    _init(other)

    assert toplevel_of(other) == other.resolve()
    assert toplevel_of(REPO_ROOT) == REPO_ROOT.resolve()


def test_the_toplevel_question_has_no_answer_outside_a_checkout(tmp_path):
    """`None` rather than a raise: `check_rebased.py` needs this case apart from a failure."""
    assert toplevel_of(tmp_path) is None


def test_the_toplevel_guard_refuses_a_subdirectory_of_a_checkout():
    """The reproduced false pass: git walks upward, so a nested directory answers about its parent.

    `git ls-files` then reports relative to the toplevel and every returned path resolves against
    the wrong directory, so a checker that skipped this guard would print a verdict about the
    enclosing repository.
    """
    with pytest.raises(RuntimeError, match="not this git repository's toplevel"):
        require_toplevel(REPO_ROOT / "tools")


def test_the_toplevel_guard_refuses_a_directory_that_is_not_a_checkout(tmp_path):
    """The other refusal, same type and a different message."""
    with pytest.raises(RuntimeError, match="not inside a git checkout"):
        require_toplevel(tmp_path)


HELPER_PID = "helper.pid"
"""Where the hanging alias records the pid of the process git's shell started."""

HELPER_REAPED_WITHIN_S = 10.0
"""How long a killed helper may take to disappear: it is reparented and reaped, not instant."""

HELPER_STARTS_WITHIN_S = 30.0
"""How long the alias's shell may take to write the helper's pid, under a loaded machine."""

TERM_SEEN = "term.seen"
"""What the terminating alias writes when its shell receives `SIGTERM`."""


def _repo_whose_git_starts_a_helper(tmp_path: Path) -> Path:
    """A checkout with an alias that forks a long-lived grandchild, the shape of an `ssh` helper.

    git runs a `!` alias through a shell, the shell backgrounds a `sleep` and waits on it, and the
    `sleep` records its pid -- so git's child's child outlives git unless something kills the whole
    group. The alias is repository-local, which :func:`tools._git.child_environment` leaves in force
    while it blanks the global and system config.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init(repo)
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["git", "config", "alias.hang", f"!sleep 60 & echo $! > {HELPER_PID}; wait"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=child_environment(),
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return repo


def _helper_pid(repo: Path) -> int:
    """The grandchild's pid, once the shell has written it."""
    path = repo / HELPER_PID
    deadline = time.monotonic() + HELPER_STARTS_WITHIN_S
    while not path.is_file() or not path.read_text().strip():
        assert time.monotonic() < deadline, "the alias never started its helper"
        time.sleep(0.02)
    return int(path.read_text())


def _is_gone(pid: int) -> bool:
    """Whether `pid` disappears within :data:`HELPER_REAPED_WITHIN_S`."""
    deadline = time.monotonic() + HELPER_REAPED_WITHIN_S
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _hold_the_clock_for_the_helper(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Hold `communicate` until the helper exists, so the runner's timeout measures a started one.

    Without it a loaded machine can reach the timeout before the alias's shell has written its pid,
    and the case fails blaming the runner for contention. `communicate`'s timeout is relative to its
    own call, so the real wait still runs its full allowance after this returns.
    """
    real = subprocess.Popen.communicate

    def started_first(
        self: subprocess.Popen[str], sent: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]:
        _helper_pid(repo)
        return real(self, sent, timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", started_first)


def _reaped_or_killed(pid: int) -> bool:
    """Whether `pid` disappears on its own; killed here only when it did not.

    The kill is conditional on purpose: a pid already confirmed gone may have been reused, and on a
    machine running a regeneration a blind `SIGKILL` could land on one of its workers.
    """
    if _is_gone(pid):
        return True
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    return False


def test_a_timeout_kills_the_helper_git_started_as_well_as_git(tmp_path, monkeypatch):
    """A timeout bounds the terminal as well as the gate.

    `subprocess.run` kills git on a timeout and not what git forked, so an `ssh` transport helper
    could survive the freshness probe's timeout holding `/dev/tty` and keep prompting after the
    gate had degraded and moved on. :func:`tools._git.run` starts git in a session of its own and
    kills the whole process group, so the helper goes with it; otherwise the `sleep` outlives the
    call.
    """
    repo = _repo_whose_git_starts_a_helper(tmp_path)
    _hold_the_clock_for_the_helper(monkeypatch, repo)

    with pytest.raises(subprocess.TimeoutExpired):
        run(["hang"], root=repo, check=False, timeout=2)

    pid = _helper_pid(repo)
    assert _reaped_or_killed(pid), f"git's helper {pid} outlived the timeout that killed git"


def test_an_interrupt_kills_the_helper_too(tmp_path, monkeypatch):
    """The other exit: a Ctrl-C no longer reaches git, so the runner has to pass it on.

    A session of its own means the terminal's `SIGINT` goes to this process and not to git, which
    keeps a prompt from appearing, and would leave git and its helper running if the runner only
    handled the timeout. So any exception out of the wait kills the group. The
    interrupt is raised from inside the wait once the helper exists, which is where a real one
    lands.
    """
    repo = _repo_whose_git_starts_a_helper(tmp_path)

    def interrupted(self: subprocess.Popen[str], *args: object, **kwargs: object) -> object:
        _helper_pid(repo)
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)

    with pytest.raises(KeyboardInterrupt):
        run(["hang"], root=repo, check=False, timeout=30)

    pid = _helper_pid(repo)
    assert _reaped_or_killed(pid), f"git's helper {pid} outlived the interrupt"


def test_a_group_whose_only_member_is_a_zombie_is_not_an_error():
    """The group kill must not replace the exception it runs inside.

    macOS answers `EPERM`, not `ESRCH`, for a group whose only member is an unreaped zombie, while
    Linux signals such a group without error. That is the state git is in when a Ctrl-C or a
    deadline lands just as it exits and before it is reaped, and a kill that suppressed
    `ProcessLookupError` alone would let a `PermissionError` replace the operator's
    `KeyboardInterrupt`. `waitid` with `WNOWAIT` waits for the exit without reaping, which is how
    the zombie is held here.
    """
    process = subprocess.Popen(
        ["true"], start_new_session=True, stdin=subprocess.DEVNULL, text=True
    )
    try:
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        _end_group(process)
    finally:
        process.wait()


def test_the_group_is_asked_to_terminate_before_it_is_killed(tmp_path, monkeypatch):
    """`SIGTERM` first, so git can remove a lock file it holds; `SIGKILL` only after a grace.

    A `SIGKILL` alone skips git's own cleanup, so an interrupted command that writes the index
    would leave `.git/index.lock` behind. The alias's shell traps `TERM` and records it, which it
    cannot do for `KILL`.
    """
    repo = _repo_whose_git_starts_a_helper(tmp_path)
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            "git",
            "config",
            "alias.trapterm",
            (
                f"!trap 'echo term > {TERM_SEEN}; exit 0' TERM; "
                f"echo $$ > {HELPER_PID}; sleep 60 & wait"
            ),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=child_environment(),
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    _hold_the_clock_for_the_helper(monkeypatch, repo)

    with pytest.raises(subprocess.TimeoutExpired):
        run(["trapterm"], root=repo, check=False, timeout=1)

    assert (repo / TERM_SEEN).is_file(), "the group was killed without being asked to terminate"


def _respelled_checkout(tmp_path: Path) -> Path:
    """A checkout addressed through a case-changed spelling, or a skip where that names nothing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init(repo)
    respelled = Path(str(repo).swapcase())
    if not respelled.exists():
        pytest.skip("this filesystem is case-sensitive, so the respelled path names nothing")
    return respelled


def test_the_toplevel_guard_admits_a_case_changed_spelling_of_the_checkout(tmp_path):
    """On a case-insensitive filesystem a case-changed path names the same checkout.

    git reports its toplevel in the on-disk case and `Path.resolve()` keeps whatever case it was
    handed, so comparing the two as strings would refuse the checkout it was pointed at. The guard
    compares them as files.
    """
    require_toplevel(_respelled_checkout(tmp_path))


def test_the_rebase_check_admits_a_case_changed_spelling_of_the_checkout(tmp_path):
    """`check_rebased.py` asks the same question its own way, so it needs the same comparison.

    A string comparison would make its pre-flight exit 2, "git answers about ... rather than ...",
    naming one directory twice in two spellings.
    """
    import tools.check_rebased as subject

    assert subject._preflight(_respelled_checkout(tmp_path)) is None


_NETWORK_RECIPES = frozenset({"tools.check_rebased"})
"""The checkers whose real run reaches the network: `check_rebased` asks `origin` with `ls-remote`.

Its recipe case is marked `integration`; the other two read only the checkout and the environment.
"""


def _checker_recipes() -> list[list[str]]:
    """Every Makefile recipe line that runs something under `tools/`, as argv.

    Read out of the Makefile rather than spelled here: the two subprocess tests in
    `tests/test_file_size.py` write their own argv and so cannot see a recipe that has drifted away
    from the form they use.
    """
    lines = (REPO_ROOT / "Makefile").read_text().splitlines()
    found = [
        line.strip().split()
        for line in lines
        if line.startswith("\t") and "tools" in line and "pytest" not in line
    ]
    # Distinct recipes only: `rebased` and `rebased-final` run the same command, and running it
    # twice here would pay the real checker's network probe twice for no extra coverage.
    distinct = list(dict.fromkeys(map(tuple, found)))
    modules = sorted(recipe[-1] for recipe in distinct)
    assert modules == ["tools.check_file_size", "tools.check_lock_drift", "tools.check_rebased"], (
        f"the Makefile's checker recipes are not the three checkers, each run one way: {found}"
    )
    return [list(recipe) for recipe in distinct]


@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param(
            recipe,
            id=recipe[-1],
            marks=[pytest.mark.integration] if recipe[-1] in _NETWORK_RECIPES else [],
        )
        for recipe in _checker_recipes()
    ],
)
def test_each_checker_recipe_still_runs(recipe):
    """Run what the Makefile runs, with `uv run` replaced by this interpreter.

    This is the only thing that reads those recipes. Reverting one to the script form
    `python tools/check_file_size.py` would leave the rest of the suite green: `tools/` becomes
    `sys.path[0]` and `from tools._git import ...` raises `ModuleNotFoundError`. That is exactly
    the edit a reader who does not know `tools/` is a package would make.

    The `uv run` prefix is dropped so the case does not resolve an environment mid-suite; what
    carries the defect is the argv after `python`, which is taken verbatim.
    """
    after_python = list(itertools.dropwhile(lambda word: word != "python", recipe))[1:]
    assert after_python, f"no `python` in the recipe: {recipe}"

    finished = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, *after_python],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )

    assert "Traceback" not in finished.stderr, (
        f"the recipe {after_python} did not run: {finished.stderr}"
    )
    assert finished.returncode in {0, 1, 2}, (
        f"the recipe {after_python} exited {finished.returncode}, which is outside the three-code "
        f"contract: {finished.stderr}"
    )


def _tools_modules() -> list[Path]:
    """Every *checker* under `tools/`, which is every module defining a top-level `main`.

    Derived rather than excluded by name, so a new non-checker helper does not land inside the
    population: `tools/_remote.py` runs git but prints nothing and returns no verdict, so it has no
    verdict to state about the wrong repository. Its caller holds the toplevel guard. What
    separates the two kinds is exactly a `main` that returns an exit code, so that is what selects.
    """
    found = sorted(
        path
        for path in (REPO_ROOT / "tools").glob("*.py")
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
        )
    )
    assert len(found) >= 3, f"the checker scan found {len(found)} modules, so it is not scanning"
    return found


@pytest.mark.parametrize("module", _tools_modules(), ids=operator.attrgetter("name"))
def test_a_checker_that_runs_git_also_establishes_which_repository_answered(module):
    """The pairing `tools/_git.py` exists to keep, asserted rather than left to its docstring.

    A checker that took the environment scrubbing and not the toplevel guard would print a verdict
    about an enclosing repository as a verdict about this one. One module makes the pair
    *available* together; nothing makes it compulsory, so the half-copy is one import line away.

    `--show-toplevel` is accepted as the other way to satisfy this: `check_rebased.py` needs "git
    absent", "not a checkout" and "wrong toplevel" apart, so it asks the question through
    `toplevel_of` and prints its own exit codes rather than raising.
    """
    tree = ast.parse(module.read_text(), filename=str(module))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "tools._git"
        for alias in node.names
    }
    if "run" not in imported:
        pytest.skip(f"{module.name} does not run git through the shared runner")

    assert imported & {"require_toplevel", "toplevel_of"}, (
        f"{module.name} imports the shared git runner without either way of establishing which "
        "repository answered; git walks upward, so its verdict may be about an enclosing tree"
    )
