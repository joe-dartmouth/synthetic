"""Load the pinned releases a set of concepts names, and the ontology fragments they scope.

This is the release loader the command lines share. It lives apart from `bko.gaps.detect` so that
a module reading fragments does not import pronto or the network fetch with them.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from pathlib import Path

from bko.bkb.structure import BKB
from bko.datasets.fetch import UrlOpener
from bko.datasets.sources import Source
from bko.gaps.detect import ontology_fragments
from bko.gaps.errors import GapError
from bko.ontology.releases import load_releases
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.gaps")


def load_scope(
    concepts: Collection[str],
    data_dir: Path,
    *,
    releases: Mapping[str, Source],
    open_url: UrlOpener,
) -> tuple[dict[str, Ontology], tuple[BKB, ...]]:
    """Load the release for each concept's prefix and the fragment each scopes to `concepts`.

    Releases absent from `data_dir` are fetched through `open_url`. Each fragment's class and
    statement counts go to the `bko.gaps` logger at INFO.

    Args:
        concepts: The CURIEs the mappings name.
        data_dir: The data root the releases are read from and fetched into.
        releases: The pinned release for each concept prefix.
        open_url: The HTTPS opener.

    Returns:
        The loaded ontologies by prefix, and their fragments over `concepts`.

    Raises:
        GapError: A concept's prefix has no pinned release.
    """
    prefixes = sorted({concept.partition(":")[0] for concept in concepts})
    unpinned = [prefix for prefix in prefixes if prefix not in releases]
    if unpinned:
        msg = f"no pinned release for the prefix {', '.join(unpinned)}"
        raise GapError(msg)
    sources = {prefix: releases[prefix] for prefix in prefixes}
    ontologies = load_releases(sources, data_dir, open_url=open_url)
    fragments = ontology_fragments(ontologies, concepts)
    for fragment in fragments:
        logger.info(
            "scope release=%s classes=%d statements=%d",
            fragment.name,
            len(fragment.inodes),
            len(fragment.snodes),
        )
    return ontologies, fragments
