"""Offline inputs for the `bko.ontology` tests: OBO release text and a URL-routing opener.

The MONDO:0024644 stanza of `MONDO_TEXT`, with the header lines that declare its prefixes, is from
MONDO release 2026-10-06 (CC BY 4.0); every other `MONDO_TEXT` stanza is invented.
`SANTOS_MONDO_TEXT` abridges MONDO release 2026-10-06 (CC BY 4.0) to the id, name and is_a lines
and the skos:exactMatch lines to DO of the six MONDO classes of Santos et al.'s example, with the
id and name of their parents, qualifiers dropped; `SANTOS_DO_TEXT` abridges Human Disease Ontology
release 2026-09-30 (CC0 1.0) the same way. `HPO_TEXT` is invented: it reuses HPO ids and labels
only, and no HPO file content.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path


_REDUNDANT_IS_A = (
    'is_a: MONDO:0024643 {source="MONDO:Redundant", '
    'source="https://orcid.org/0000-0002-6601-2165"} ! myocardial disorder'
)

MONDO_TEXT = f"""\
format-version: 1.2
data-version: releases/2026-10-06
idspace: SCTID http://identifiers.org/snomedct/
idspace: skos http://www.w3.org/2004/02/skos/core#
ontology: mondo

[Term]
id: MONDO:0024644
name: myocardial ischemia
xref: SCTID:414545008 {{source="MONDO:equivalentTo"}}
is_a: MONDO:0005053 ! ischemic disease
{_REDUNDANT_IS_A}
property_value: skos:exactMatch SCTID:414545008

[Term]
id: MONDO:0005053
name: ischemic disease

[Term]
id: MONDO:0024643
name: myocardial disorder

[Term]
id: MONDO:0006715
name: coronary stenosis

[Term]
id: MONDO:0005015
name: diabetes mellitus

[Term]
id: MONDO:0021661
name: coronary atherosclerosis

[Term]
id: MONDO:9999998
name: obsolete test term
is_obsolete: true
replaced_by: MONDO:0024644

[Term]
id: MONDO:9999999
is_obsolete: true
replaced_by: ENVO:03000144

[Term]
id: MONDO:9999997
name: Sjögren-like test term
"""
"""A MONDO-shaped release: one real stanza, its stub parents, a stanza per other MONDO id the
mapping table names, and invented edge cases."""

HPO_LABELS: Mapping[str, str] = {
    "HP:0000822": "Hypertension",
    "HP:0001658": "Myocardial infarction",
    "HP:0001681": "Angina pectoris",
    "HP:0001712": "Left ventricular hypertrophy",
    "HP:0001717": "Coronary artery calcification",
    "HP:0003074": "Hyperglycemia",
    "HP:0003115": "Abnormal EKG",
    "HP:0003124": "Hypercholesterolemia",
    "HP:0004421": "Elevated systolic blood pressure",
    "HP:0005145": "Coronary artery stenosis",
    "HP:0012250": "ST segment depression",
    "HP:0033992": "Chronotropic incompetence",
    "HP:0100749": "Chest pain",
    "HP:0500018": "Abnormal cardiac exercise stress test",
}
"""Every HP id the mapping table names, with its label in HPO release 2026-09-01."""

HPO_TEXT = (
    "format-version: 1.2\ndata-version: hp/releases/2026-09-01\nontology: hp.obo\n"
    + "".join(f"\n[Term]\nid: {curie}\nname: {name}\n" for curie, name in HPO_LABELS.items())
)
"""An HPO-shaped release: one bare stanza per id in `HPO_LABELS`."""

SANTOS_MONDO_TEXT = """\
format-version: 1.2
data-version: releases/2026-10-06
idspace: skos http://www.w3.org/2004/02/skos/core#
ontology: mondo

[Term]
id: MONDO:0006960
name: sciatic neuropathy
is_a: MONDO:0001397 ! mononeuropathy
property_value: skos:exactMatch DOID:11446

[Term]
id: MONDO:0001543
name: lesion of sciatic nerve
is_a: MONDO:0001829 ! lumbosacral plexus lesion
is_a: MONDO:0006960 ! sciatic neuropathy
is_a: MONDO:0024334 ! peripheral nerve lesion
property_value: skos:exactMatch DOID:12528

[Term]
id: MONDO:0001397
name: mononeuropathy
is_a: MONDO:0005244 ! peripheral neuropathy
property_value: skos:exactMatch DOID:1188

[Term]
id: MONDO:0002121
name: mononeuritis simplex
is_a: MONDO:0001397 ! mononeuropathy
is_a: MONDO:0002122 ! neuritis
property_value: skos:exactMatch DOID:1802

[Term]
id: MONDO:0002122
name: neuritis
is_a: MONDO:0005244 ! peripheral neuropathy
is_a: MONDO:0021166 ! inflammatory disease
property_value: skos:exactMatch DOID:1803

[Term]
id: MONDO:0021166
name: inflammatory disease
is_a: MONDO:7770007 ! disease by developmental or physiological process

[Term]
id: MONDO:0001829
name: lumbosacral plexus lesion

[Term]
id: MONDO:0024334
name: peripheral nerve lesion

[Term]
id: MONDO:0005244
name: peripheral neuropathy

[Term]
id: MONDO:7770007
name: disease by developmental or physiological process
"""
"""The MONDO side of Santos et al.'s example, abridged from release 2026-10-06."""

DOID_9473_STANZA = (
    "[Term]\nid: DOID:9473\nname: mononeuritis of lower limb\nis_a: DOID:1802 ! mononeuritis\n\n"
)
"""One DO stanza of the example, which a test removes."""

SANTOS_DO_TEXT = f"""\
format-version: 1.2
data-version: releases/2026-09-30/doid.obo
ontology: doid

[Term]
id: DOID:11446
name: sciatic neuropathy
is_a: DOID:12528 ! lesion of sciatic nerve

[Term]
id: DOID:12528
name: lesion of sciatic nerve
is_a: DOID:9473 ! mononeuritis of lower limb

{DOID_9473_STANZA}[Term]
id: DOID:1802
name: mononeuritis
is_a: DOID:1188 ! mononeuropathy

[Term]
id: DOID:1188
name: mononeuropathy
is_a: DOID:870 ! neuropathy

[Term]
id: DOID:1803
name: neuritis
is_a: DOID:870 ! neuropathy

[Term]
id: DOID:870
name: neuropathy
"""
"""The DO side of Santos et al.'s example, abridged from release 2026-09-30."""


def write_obo(directory: Path, text: str, name: str = "release.obo") -> Path:
    """Write `text` as UTF-8 to `directory / name` and return the path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class RoutingOpener:
    """Serve a body per URL, or raise; record every call.

    `error` is raised on open, for any URL. An unknown URL fails the test.
    """

    def __init__(self, bodies: Mapping[str, bytes], error: BaseException | None = None) -> None:
        self.bodies = bodies
        self.error = error
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.calls.append((url, timeout))
        if self.error is not None:
            raise self.error
        if url not in self.bodies:
            msg = f"unexpected URL {url}"
            raise AssertionError(msg)
        return io.BytesIO(self.bodies[url])
