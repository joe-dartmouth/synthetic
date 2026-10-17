"""What `bko.bkb` raises when a BKB cannot be built, fused, queried or read.

The exit codes follow `bko.datasets.errors`: 2 the operation could not run, 3 a document
contradicts its format.
"""

from __future__ import annotations

from typing import ClassVar


class BKBError(Exception):
    """A BKB could not be built, fused or queried."""

    exit_code: ClassVar[int] = 2


class StructureError(BKBError):
    """An I-node or S-node would break the BKB: an unknown I-node, a bad name or probability."""


class FusionError(BKBError):
    """`bko.bkb.fusion.fuse` was given fragments, reliabilities or sources it cannot fuse."""


class QueryError(BKBError):
    """A query cannot be answered as asked.

    It names an unknown or source target, evidence or anchor, its evidence is incomplete, or an
    inference cannot be followed.
    """


class DocumentFormatError(BKBError):
    """A BKB document breaks the format `bko.bkb.serialize` documents."""

    exit_code: ClassVar[int] = 3
