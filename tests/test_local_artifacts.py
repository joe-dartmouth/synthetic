"""The team's source PDFs and rendered slide exports stay local: gitignored and never tracked.

`papers/` holds the proposal, the research brief and the papers the docs cite, which the docs point
at through `docs/RESEARCH.md` or a DOI instead; a deck under `presentations/` is its Marp source,
which `marp` renders to PDF or PPTX on demand.
"""

from __future__ import annotations

import subprocess

from tests._corpus import REPO_ROOT
from tools._git import child_environment


LOCAL_PATHS = ("papers/santos-2024.pdf", "presentations/deck.pdf", "presentations/deck.pptx")
"""A path of each kind that must stay local."""


def test_each_local_artifact_path_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "--verbose", "--non-matching"],
        cwd=REPO_ROOT,
        env=child_environment(),
        input="\n".join(LOCAL_PATHS) + "\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    unmatched = [line for line in result.stdout.splitlines() if line.startswith("::")]
    assert (result.returncode, unmatched) == (0, [])


def test_no_local_artifact_is_tracked():
    listed = subprocess.run(
        ["git", "ls-files", "papers", "presentations/*.pdf", "presentations/*.pptx"],
        cwd=REPO_ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert (listed.returncode, listed.stdout) == (0, "")
