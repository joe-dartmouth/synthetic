"""The failures `bko.ontology` adds to those `bko.datasets.errors` defines."""

from __future__ import annotations

from bko.datasets.errors import ContractError, DatasetError


class ReleaseFormatError(ContractError):
    """A pinned release is not OBO the loader reads.

    pronto refuses it or reports a construct it skipped, its header lacks `data-version` or names
    an import, or it holds no term. Exit 3, as for any contract failure.
    """


class ScopeError(DatasetError):
    """A fragment's scope names a class its release cannot supply.

    The class lacks the release's prefix, is absent from it or obsolete in it, or the scope is
    empty. Exit 2: the request cannot be met.
    """
