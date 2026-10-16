"""The pybkb port's provenance: license notices, and fixtures that match their manifest."""

from __future__ import annotations

import ast
import hashlib
import json
import re

import pytest

from tests._bkb_fixtures import PYBKB_DIR, manifest
from tests._corpus import REPO_ROOT


COMMIT = "1e1b62a02faa0abc73db4597076c2d38c1313df6"

PORTED = (
    "tests/pybkb/export.py",
    "tests/_bkb_fixtures.py",
    "src/bko/bkb/structure.py",
    "tests/test_bkb_structure.py",
)
"""Every module adapted from pybkb; later phases add theirs."""

NOTICE = re.compile(
    r"Adapted from pybkb \(https://github\.com/di2ag/pybkb\), by Chase Yakaboski, which is"
    r" licensed under the GNU General Public License, version 3\. Modified for bko on"
    r" \d{4}-\d{2}-\d{2}; ADR `research/bkb-layer-source` names the pybkb commit and lists the"
    r" changes\. This file is released under the GNU General Public License, version 3 only"
)


@pytest.mark.parametrize("path", PORTED)
def test_every_ported_module_carries_the_gpl_notice(path):
    docstring = ast.get_docstring(ast.parse((REPO_ROOT / path).read_text(encoding="utf-8")))
    assert docstring is not None
    assert NOTICE.search(" ".join(docstring.split())), path


def test_the_manifest_names_pybkbs_commit_and_carries_the_notice():
    assert manifest()["commit"] == COMMIT
    assert "Chase Yakaboski" in manifest()["notice"]


def test_every_committed_output_matches_its_manifest_sha256():
    on_disk = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PYBKB_DIR.glob("*.json")
        if path.name not in {"manifest.json", "run.json"}
    }
    assert on_disk == manifest()["outputs"]
    assert {"queries.json", "pirate_fused_r1.json", "aquatic_eco.json"} <= set(on_disk)


def test_the_manifest_names_the_committed_exporter():
    exporter = (PYBKB_DIR / "export.py").read_bytes()
    assert manifest()["exporter"] == hashlib.sha256(exporter).hexdigest()


def test_the_exporter_parses_with_the_python_311_grammar():
    ast.parse((PYBKB_DIR / "export.py").read_text(encoding="utf-8"), feature_version=(3, 11))


def test_the_export_ran_with_hash_seed_zero():
    assert json.loads((PYBKB_DIR / "run.json").read_text(encoding="utf-8")) == {"hash_seed": "0"}


def test_pybkbs_own_checks_held_when_the_fixtures_were_exported():
    assert manifest()["checks"] == dict.fromkeys(
        [
            "find_snodes_counts",
            "goldfish_collapsed_is_mutex",
            "goldfish_is_mutex",
            "json_matches_to_dict",
            "non_mutex_cases_are_refused",
            "pirate_fresh_equals_stored",
            "pirate_is_mutex",
            "pirate_stored_equals_fresh",
            "saved_equals_reload",
            "saved_is_mutex",
            "simple_build_equals_saved",
        ],
        True,
    )
