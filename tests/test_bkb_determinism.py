"""Fused BKBs and their queries come out the same under every string-hash seed."""

from __future__ import annotations

import os
import subprocess
import sys

from tests._corpus import REPO_ROOT


DRIVER = """
import hashlib
import sys

from bko.bkb.fusion import Normalization, fuse
from bko.bkb.reason import log_joint_complete
from bko.bkb.serialize import render
from tests._bkb_fixtures import PIRATE, RELIABILITIES, build, exported, exported_queries

fragments = [exported(name) for name in PIRATE]
query = next(q for q in exported_queries() if q.bkb == "pirate_fused_r1")
for normalization in Normalization:
    fused = fuse(fragments, list(RELIABILITIES), normalization=normalization)
    sys.stdout.write(hashlib.sha256(render(fused).encode()).hexdigest() + "\\n")
    result = log_joint_complete(fused, query.target, query.evidence)
    pairs = [(s, None if v is None else v.hex()) for s, v in sorted(result.items())]
    sys.stdout.write(repr(pairs) + "\\n")
    sys.stdout.write(repr(fused.non_mutex_pair()) + "\\n")
sources = [(f"__Source__[{name}]", "s") for name in "ABCDEFGH"]
priors = [(c, s, 0.1 + index / 10, []) for index, (c, s) in enumerate(sources)]
rows = [("A", "1", 0.5, sources), *priors]
result = log_joint_complete(build("many sources", rows), "A", {})
sys.stdout.write(result["1"].hex() + "\\n")
"""


def run_driver(seed: str) -> str:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    env["PYTHONHASHSEED"] = seed
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", DRIVER],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_fusion_and_queries_do_not_depend_on_the_hash_seed():
    outputs = {run_driver(seed) for seed in ("0", "1", "2", "4294967295")}
    assert len(outputs) == 1
    lines = outputs.pop().splitlines()
    assert len(lines) == 10
    assert "0x" in lines[4]
    assert lines[2] == lines[5] == lines[8] == "None"
