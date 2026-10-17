"""The failure `bko.gaps` adds to those `bko.datasets.errors` defines; the exit codes follow it."""

from __future__ import annotations

from bko.datasets.errors import DatasetError


class GapError(DatasetError):
    """Gap detection cannot run on its input.

    A dataset holds no record, a row reads a field the records lack, or a concept's prefix has no
    pinned release. Exit 2: the request cannot be met.
    """
