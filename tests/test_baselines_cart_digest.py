"""`cart`'s fills keep their bytes, so a refactor of `bko.baselines.classical` moves none."""

from __future__ import annotations

import hashlib

import pytest

from bko.baselines.classical import cart
from bko.baselines.request import Request, Strata, Stratum
from tests._frame_fixtures import SHORT, TOY, toy_records


CART_DIGESTS = {
    0: "229ca8355334471438110acc37c2e7c3a5522f166d96ab398e725e2948968770",
    1: "c001afb4b5470e313ac83e686300b16145368e1971b0fb5ce7a6f75009218287",
    2: "9780f510b4c113f7ba4e4c6e1fe01fc2b9b391052e5e582b494ee3793831ece7",
}
"""The sha256 of `cart`'s rows on `toy_records` at each seed, as CSV, recorded before any edit."""


@pytest.mark.parametrize("seed", sorted(CART_DIGESTS))
def test_cart_fills_keep_their_bytes(seed):
    quota: dict[Stratum, int] = dict.fromkeys(SHORT, 5)
    request = Request(quota=quota, strata=Strata.CELL, seed=seed, frame=TOY)
    added = cart(toy_records(), request).added
    found = hashlib.sha256(added.to_csv(lineterminator="\n").encode()).hexdigest()
    assert found == CART_DIGESTS[seed]
