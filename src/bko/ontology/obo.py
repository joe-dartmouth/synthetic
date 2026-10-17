"""Parse a pinned OBO release into `bko.ontology.terms` values: the only module importing pronto.

pronto is given a verified path, never a handle: it honours `encoding` only for a path and
otherwise guesses from the first buffer. `import_depth=0` keeps it off the network. Warnings it
raises while parsing are recorded: one saying a construct was skipped refuses the release; any
other concerns a value no term field keeps, such as a date, and is logged.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from types import MappingProxyType

import pronto

from bko.datasets.errors import UnavailableError
from bko.datasets.fetch import verified_path
from bko.datasets.sources import Source
from bko.ontology.errors import ReleaseFormatError
from bko.ontology.terms import MATCH_PREDICATES, Ontology, Term


logger = logging.getLogger(__name__)


def load_ontology(source: Source, data_dir: Path) -> Ontology:
    """Verify `source`'s snapshot under `data_dir` and parse it; never fetches.

    Raises:
        UnavailableError: If the snapshot is absent.
        ContractError: If it contradicts its pin.
        ReleaseFormatError: As `parse_obo`.
    """
    return parse_obo(verified_path(source, data_dir), name=source.name)


def parse_obo(path: Path, *, name: str) -> Ontology:
    """Parse the OBO file at `path` into an `Ontology` called `name`.

    Raises:
        UnavailableError: If there is no file at `path`.
        ReleaseFormatError: If pronto refuses it or skips a construct, the header lacks
            `data-version` or names an import, or it holds no term.
    """
    if not path.is_file():
        msg = f"no file at {path}"
        raise UnavailableError(msg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            parsed = pronto.Ontology(path, import_depth=0, encoding="utf-8")
        except (SyntaxError, ValueError, KeyError) as exc:
            msg = f"{path}: pronto cannot read it: {type(exc).__name__}: {exc}"
            raise ReleaseFormatError(msg) from exc
    _refuse_skipped(caught, path)
    metadata = parsed.metadata
    if metadata.imports:
        msg = f"{path} imports {sorted(metadata.imports)}; a pinned release must be self-contained"
        raise ReleaseFormatError(msg)
    if not metadata.data_version:
        msg = f"{path} has no data-version header"
        raise ReleaseFormatError(msg)
    terms = {term.id: _term(term) for term in parsed.terms()}
    if not terms:
        msg = f"{path} holds no term"
        raise ReleaseFormatError(msg)
    return Ontology(
        name=name,
        version=metadata.data_version,
        terms=MappingProxyType(dict(sorted(terms.items()))),
    )


def _refuse_skipped(caught: list[warnings.WarningMessage], path: Path) -> None:
    # pronto's NotImplementedWarning subclasses NotImplementedError (pronto/utils/warnings.py).
    skipped = [str(w.message) for w in caught if issubclass(w.category, NotImplementedError)]
    if skipped:
        msg = f"{path}: pronto skipped constructs: {'; '.join(skipped)}"
        raise ReleaseFormatError(msg)
    if caught:
        logger.warning(
            "%s: pronto reported %d unread values; no term field keeps them", path, len(caught)
        )


def _term(term: pronto.Term) -> Term:
    return Term(
        curie=term.id,
        name=term.name,
        parents=tuple(sorted(p.id for p in term.superclasses(distance=1, with_self=False))),
        xrefs=tuple(sorted(x.id for x in term.xrefs)),
        matches=_matches(term),
        obsolete=term.obsolete,
        replaced_by=tuple(sorted(term.replaced_by.ids)),
    )


def _matches(term: pronto.Term) -> tuple[tuple[str, str], ...]:
    # A resource annotation is told by its `resource` value: pronto's class decorators hide
    # `ResourcePropertyValue` from the type checker, so `isinstance` against it cannot be checked.
    found: list[tuple[str, str]] = []
    for pv in term.annotations:
        resource = getattr(pv, "resource", None)
        if isinstance(resource, str) and pv.property in MATCH_PREDICATES:
            found.append((pv.property, resource))
    return tuple(sorted(found))
