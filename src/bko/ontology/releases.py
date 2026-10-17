"""Fetch the pinned releases a command reads into the data root, then load them.

The loader in `bko.ontology.obo` never fetches; the commands of `bko.ontology` and `bko.gaps`
fetch through `load_releases`, which logs each release's identity.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Mapping
from pathlib import Path

from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.sources import Source
from bko.ontology.obo import load_ontology
from bko.ontology.terms import Ontology


logger = logging.getLogger(__name__)


def load_releases(
    sources: Mapping[str, Source], data_dir: Path, *, open_url: UrlOpener
) -> dict[str, Ontology]:
    """Fetch each release into `data_dir` if absent, then verify and parse it, logging its identity.

    Args:
        sources: The pinned release for each concept prefix.
        data_dir: The data root.
        open_url: The HTTPS opener `ensure` fetches with.

    Returns:
        Each prefix's release, in prefix order.

    Raises:
        DatasetError: As `bko.datasets.fetch.ensure` and `load_ontology`.
    """
    logger.info("parser pronto %s", importlib.metadata.version("pronto"))
    ontologies: dict[str, Ontology] = {}
    for prefix, source in sorted(sources.items()):
        ensure(source, data_dir, open_url=open_url)
        ontology = load_ontology(source, data_dir)
        logger.info(
            "release %s data_version=%s snapshot=%s terms=%d obsolete=%d",
            source.name,
            ontology.version,
            source.sha256[:16],
            len(ontology.terms),
            sum(term.obsolete for term in ontology.terms.values()),
        )
        ontologies[prefix] = ontology
    return ontologies
