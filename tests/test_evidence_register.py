"""Each results-register entry has a row in RESEARCH section 11's register table."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ENTRY = re.compile(r"^## ([A-Z]\d+)\. ", re.MULTILINE)
"""An entry's heading in `docs/EVIDENCE.md`: its id, then a full stop."""

ROW = re.compile(r"^\| \[([A-Z]\d+)\]\(EVIDENCE\.md#[^)]+\) \| [^|]+ \|$", re.MULTILINE)
"""A row of the register table: the entry's id linked to its heading, then how it is read."""


def _register_table() -> str:
    """The table under RESEARCH section 11's `What each register entry is to the study`."""
    research = (ROOT / "docs" / "RESEARCH.md").read_text(encoding="utf-8")
    _, _, rest = research.partition("### What each register entry is to the study\n")
    section, _, _ = rest.partition("\n### ")
    return section


def test_every_evidence_entry_has_a_register_row():
    evidence = (ROOT / "docs" / "EVIDENCE.md").read_text(encoding="utf-8")
    entries = ENTRY.findall(evidence)
    rows = ROW.findall(_register_table())
    assert entries
    assert [entry for entry in entries if entry not in rows] == []
