"""Offline NHAMCS inputs: `ed2022` text and its archive, built in memory.

The real data here is `REAL_LINE` and `REAL_ADMITTED`, the file's first and twelfth records byte
for byte, kept so the parser is proven against the real layout. Every other record is built from
`REAL_LINE` with invented items.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.nhamcs import FLAGS, MEMBER, POSITIONS
from tests._datasets_fixtures import archive


REAL_LINE: Final = (
    "0920604001002280232-07011022011102-7000000001-8098407201610406607203-80215451-0009-0009-"
    "0009-0009020103-9-901-9  -9  -9  T833-9  -9  -9  -9  00-7-7-7-70000000000000000000000100"
    "10010001100000000000000000010100-9000000-91010060000000000000000016116028495932209214132"
    "9050447393352-9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   "
    "-9   -9   -9   -9   -9   -9   -9   -9   02010101010102-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7"
    "-7-7-7-7-7-705020701098908111814308301000000000001000000000000-7-7-7-7  -7  -7  -7  -7  "
    "-7-7-00720000000340010101020102010201000000010102-7-7-7010202010202010101010201010102010"
    "5010101010211d00141561094         087         094                     d00438561117129   "
    "   115122      117129                  d00273161061         057         058         061 "
    "        d00867161195         057         065         195         d00049561059         05"
    "7         058         059         d01461161368         331         112         368      "
    "   d00141561094         087         094                                                 "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                   3202220122201100001 3665.56954  8.364"
    "13-007"
)
"""The real file's first record, 2,382 characters without its CRLF: a visit of a 23-year-old
woman by hospital 34, diagnosis T833, not admitted, no chronic condition."""


REAL_ADMITTED: Final = (
    "0951014001402040655-070110220111010100010000003098810201812808409102030215451-0009-0009-"
    "0009-0009010004-9-9-9-9  -9  -9  K859R109-9  -9  -9  0000-7-7-71000000001100000101000005"
    "10010001000000000000000000110101-9100000-90000051000000000010000110447315005-9   -9   -9"
    "   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   -9   "
    "-9   -9   -9   -9   -9   -9   -9   -9   0102-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7"
    "-7-7-7-7-7-701010201097011601512308401001000000000000000001000060117R99--9  -9  -9  -9  "
    "02-7-00720000000340120101020102010201000000010102-7-7-7010202010202010101010201010102010"
    "5010101010211d01461161368         331         112         368         d00255121060      "
    "   057         058         060                                                          "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                                                        "
    "                                                   3202220122201100001 3665.56954   .   "
    "  0000"
)
"""The real file's twelfth record, 2,382 characters without its CRLF: a visit of a 65-year-old
woman by hospital 34, diagnoses K859 then R109, admitted, with five chronic items ticked."""


def line(visit: int, /, **items: str) -> str:
    """`REAL_LINE` with `PATCODE` set to `visit` and the named items replaced.

    Each value must be its item's width; a diagnosis is written with its padding, `-9  `, and
    `HOSPCODE` stays 034 unless given.
    """
    text = list(REAL_LINE)
    fields = {"PATCODE": f"{visit:03d}", **items}
    for name, value in fields.items():
        first, last = POSITIONS[name]
        assert len(value) == last - first + 1, (name, value)
        text[first - 1 : last] = value
    return "".join(text)


def chronic(*names: str) -> dict[str, str]:
    """The chronic items with `names` set to 1, every other 0, and `NOCHRON` to match."""
    return {**{name: str(int(name in names)) for name in FLAGS}, "NOCHRON": "0" if names else "1"}


BLANK_CHRONIC: Final = {**dict.fromkeys(FLAGS, "0"), "NOCHRON": "2"}
"""The chronic item left entirely blank."""


def text(*lines: str) -> str:
    """The member's text: each line ended by CRLF."""
    return "".join(f"{record}\r\n" for record in lines)


LATE_TOKENS: Final = {"DIAG3": "E119", "DIAG4": "N179", "DIAG5": "R079"}
"""A token `mixed()` writes in one later diagnosis column only, and in no other column."""


def mixed() -> list[str]:
    """Twelve visits across age bands, sexes, labels, diagnoses and chronic items.

    `DIAG1` and `DIAG2` cycle through the same tokens; each of `LATE_TOKENS` sits in its column
    of one visit, so only a codebook read from every diagnosis column holds it.
    """
    diagnoses = ("I509", "A00-", "A000", "J45-", "ZZZ0", "-9  ")
    flags = ((), ("CHF",), ("ASTHMA", "HTN"), ("DIABTYP2",))
    late = {1: "DIAG3", 4: "DIAG4", 9: "DIAG5"}
    return [
        line(
            n + 1,
            AGE=f"{(5, 30, 50, 70, 94, 17)[n % 6]:03d}",
            SEX="1" if n % 2 else "2",
            ADMITHOS=str(int(n % 3 == 0)),
            DIAG1=diagnoses[n % len(diagnoses)],
            DIAG2=diagnoses[(n + 2) % len(diagnoses)],
            **({late[n]: LATE_TOKENS[late[n]]} if n in late else {}),
            **(BLANK_CHRONIC if n == 7 else chronic(*flags[n % len(flags)])),
        )
        for n in range(12)
    ]


MIXED_DIGEST: Final = "db981858585024a87b39fa22f73a886eb100598b12cacb886f1e469dca80bf83"
"""The sha256 of `mixed()`'s records as `bko.datasets.frames.digest` renders them."""


def nhamcs_archive(*lines: str) -> bytes:
    """A zip holding `text(*lines)`, or of `mixed()` when none is given, as `MEMBER`."""
    return archive(text(*(lines or mixed())), member=MEMBER)
