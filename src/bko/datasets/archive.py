"""Read one member of a verified zip archive as text, capped in size.

`bko.datasets.diabetes` reads Diabetes 130's two members through it.
"""

from __future__ import annotations

import io
import zipfile
import zlib
from typing import Final

from bko.datasets.errors import ContractError
from bko.datasets.sources import Source


MEMBER_LIMIT: Final = 64 * 2**20
"""The most bytes a member may hold once decompressed; the archive's largest holds under 20 MiB."""


def member_text(data: bytes, member: str, source: Source) -> str:
    """Return `member` of the zip `data` as UTF-8 text, refusing one over `MEMBER_LIMIT` bytes.

    Raises:
        ContractError: If the archive has no such member, the member holds more than
            `MEMBER_LIMIT` bytes, or it cannot be read as UTF-8 text.
    """
    where = f"{source.name}: cannot read {member}"
    try:
        return _capped(data, member, where).decode("utf-8")
    except KeyError as exc:
        msg = f"{where}: the archive has no such member"
        raise ContractError(msg) from exc
    except (
        zipfile.BadZipFile,
        zlib.error,
        UnicodeDecodeError,
        NotImplementedError,
        RuntimeError,
        EOFError,
    ) as exc:
        msg = f"{where}: {type(exc).__name__}: {exc}"
        raise ContractError(msg) from exc


def _capped(data: bytes, member: str, where: str) -> bytes:
    """Return `member`'s bytes, refusing one whose stated or read size passes `MEMBER_LIMIT`."""
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        size = bundle.getinfo(member).file_size
        if size > MEMBER_LIMIT:
            msg = f"{where}: {size} bytes, over the {MEMBER_LIMIT}-byte cap"
            raise ContractError(msg)
        with bundle.open(member) as handle:
            body = handle.read(MEMBER_LIMIT + 1)
    if len(body) > MEMBER_LIMIT:
        msg = f"{where}: over the {MEMBER_LIMIT}-byte cap"
        raise ContractError(msg)
    return body
