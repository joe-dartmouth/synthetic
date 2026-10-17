"""The failure `bko.ontology` adds to those `bko.datasets.errors` defines."""

from __future__ import annotations

from bko.datasets.errors import ContractError


class ReleaseFormatError(ContractError):
    """A pinned release is not OBO the loader reads.

    pronto refuses it or reports a construct it skipped, its header lacks `data-version` or names
    an import, or it holds no term. Exit 3, as for any contract failure.
    """
