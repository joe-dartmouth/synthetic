"""Load the UCI split, its cells and the pinned releases with the ontology fragments they scope.

`load_scope` is the release loader the command lines and the app share, and `load_uci_scope` the
UCI load sequence of the commands that split UCI once (`bko.evaluate` splits per holdout, and the
app splits the records of its `bko.evaluate.data.Data`).
It lives apart from `bko.gaps.detect` so that a module reading fragments does not import pronto or
the network fetch with them.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bko.bkb.structure import BKB
from bko.datasets.fetch import UrlOpener
from bko.datasets.sources import Source
from bko.datasets.split import SPLIT_SEED, Split, uci_split
from bko.gaps.detect import ontology_fragments
from bko.gaps.errors import GapError
from bko.gaps.observe import Cells, record_cells
from bko.ontology.releases import load_releases
from bko.ontology.schema import FieldMapping
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


@dataclass(frozen=True, slots=True, eq=False)
class UciScope:
    """UCI's split, the releases its field mappings name, and its training cells.

    Attributes:
        parts: The split.
        ontologies: Each loaded release, by concept prefix.
        fragments: Each release's fragment over the mappings' concepts.
        cells: The training records observed whole and per cell.
    """

    parts: Split
    ontologies: dict[str, Ontology]
    fragments: tuple[BKB, ...]
    cells: Cells


def load_uci_scope(
    data_dir: Path,
    *,
    uci: Source,
    releases: Mapping[str, Source],
    field_rows: Sequence[FieldMapping],
    open_url: UrlOpener,
    seed: int = SPLIT_SEED,
) -> UciScope:
    """Split UCI, load the releases `field_rows` name, and observe the training cells.

    Raises:
        DatasetError: As `uci_split` and `record_cells`.
        GapError: As `load_scope`.
    """
    parts = uci_split(data_dir, source=uci, open_url=open_url, seed=seed)
    ontologies, fragments = load_scope(
        {row.concept for row in field_rows}, data_dir, releases=releases, open_url=open_url
    )
    cells = record_cells(parts.train, field_rows, name=uci.name)
    return UciScope(parts, ontologies, fragments, cells)
