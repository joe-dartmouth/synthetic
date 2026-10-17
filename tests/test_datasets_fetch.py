"""Pinned downloads: fetched once, verified on use, refused on any mismatch, never partial."""

from __future__ import annotations

import email.message
import http.client
import io
import json
import logging
import re
import urllib.error
import urllib.request
from http.client import HTTPMessage

import pytest

from bko.datasets import fetch
from bko.datasets.errors import ContractError, UnavailableError
from bko.datasets.fetch import (
    RECORD_NAME,
    TIMEOUT_SECONDS,
    ensure,
    open_https,
    read_verified,
    snapshot_dir,
    verified_path,
    verify,
)
from bko.datasets.sources import UCI_HEART_DISEASE
from tests._datasets_fixtures import FIXED_NOW, FakeOpener, pin, plant


BODY = b"pinned bytes\n" * 100


def leftovers(tmp_path, source):
    """Staging directories left beside the snapshot."""
    parent = snapshot_dir(source, tmp_path).parent
    return (
        sorted(p.name for p in parent.iterdir() if p.name.startswith("."))
        if parent.exists()
        else []
    )


def test_a_matching_download_becomes_a_snapshot(tmp_path):
    source = pin(BODY)
    path = ensure(source, tmp_path, open_url=FakeOpener(BODY), now=lambda: FIXED_NOW)
    assert path.read_bytes() == BODY
    record = json.loads((path.parent / RECORD_NAME).read_text(encoding="utf-8"))
    assert record == {
        "retrieved_at": FIXED_NOW.isoformat(),
        "sha256": source.sha256,
        "size": source.size,
        "url": source.url,
    }
    assert leftovers(tmp_path, source) == []


def test_a_present_snapshot_is_not_fetched_again(tmp_path):
    source = pin(BODY)
    opener = FakeOpener(BODY)
    first = ensure(source, tmp_path, open_url=opener)
    second = ensure(source, tmp_path, open_url=opener)
    assert first == second
    assert len(opener.calls) == 1


def test_a_corrupt_snapshot_is_fetched_again(tmp_path, caplog):
    source = pin(BODY)
    plant(tmp_path, source, b"x" * len(BODY))
    opener = FakeOpener(BODY)
    with caplog.at_level(logging.WARNING):
        path = ensure(source, tmp_path, open_url=opener)
    assert path.read_bytes() == BODY
    assert len(opener.calls) == 1
    assert "fails its pin or record" in caplog.text


def test_a_snapshot_without_its_record_is_fetched_again(tmp_path):
    source = pin(BODY)
    opener = FakeOpener(BODY)
    path = ensure(source, tmp_path, open_url=opener)
    (path.parent / RECORD_NAME).unlink()
    ensure(source, tmp_path, open_url=opener)
    assert len(opener.calls) == 2
    assert (path.parent / RECORD_NAME).is_file()


def test_a_checksum_mismatch_leaves_no_snapshot(tmp_path):
    source = pin(BODY)
    with pytest.raises(ContractError, match=source.sha256):
        ensure(source, tmp_path, open_url=FakeOpener(b"y" * len(BODY)))
    assert not snapshot_dir(source, tmp_path).exists()
    assert leftovers(tmp_path, source) == []


def test_a_short_download_is_refused(tmp_path):
    source = pin(BODY)
    with pytest.raises(ContractError, match=f"{len(BODY) - 1} bytes"):
        ensure(source, tmp_path, open_url=FakeOpener(BODY[:-1]))
    assert not snapshot_dir(source, tmp_path).exists()
    assert leftovers(tmp_path, source) == []


def test_a_download_in_many_chunks_is_hashed_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CHUNK_BYTES", 7)
    source = pin(BODY)
    assert ensure(source, tmp_path, open_url=FakeOpener(BODY)).read_bytes() == BODY


def test_an_oversized_stream_stops_at_the_chunk_that_crosses_the_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CHUNK_BYTES", 7)
    source = pin(BODY)
    body = CountingBody(BODY * 3)
    with pytest.raises(ContractError, match=f"{source.url}: more than the pinned"):
        ensure(source, tmp_path, open_url=lambda url, timeout: body)
    assert body.read_bytes < len(BODY) + 7


class CountingBody(io.BytesIO):
    """A response body that counts the bytes read from it, readable after it is closed."""

    read_bytes = 0

    def read(self, size: int | None = -1, /) -> bytes:
        chunk = super().read(size)
        self.read_bytes += len(chunk)
        return chunk


@pytest.mark.parametrize(
    "error",
    [
        http.client.IncompleteRead(b"partial"),
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
    ],
    ids=["incomplete-read", "timeout", "connection-reset"],
)
def test_a_failure_mid_stream_leaves_no_partial_file(tmp_path, monkeypatch, error):
    monkeypatch.setattr(fetch, "CHUNK_BYTES", 7)
    source = pin(BODY)
    with pytest.raises(UnavailableError, match=source.url):
        ensure(source, tmp_path, open_url=FakeOpener(BODY, mid_stream=error))
    assert not snapshot_dir(source, tmp_path).exists()
    assert leftovers(tmp_path, source) == []


def test_a_download_past_its_deadline_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "DEADLINE_SECONDS", -1.0)
    source = pin(BODY)
    with pytest.raises(UnavailableError, match="took longer than"):
        ensure(source, tmp_path, open_url=FakeOpener(BODY))


def test_a_snapshot_a_concurrent_run_published_first_is_verified_and_used(tmp_path):
    source = pin(BODY)
    opener = FakeOpener(BODY, on_open=lambda: plant(tmp_path, source, BODY))
    assert ensure(source, tmp_path, open_url=opener).read_bytes() == BODY
    assert leftovers(tmp_path, source) == []


def test_a_snapshot_gets_the_mode_of_a_plain_directory(tmp_path):
    source = pin(BODY)
    final = ensure(source, tmp_path, open_url=FakeOpener(BODY)).parent
    plain = tmp_path / "plain"
    plain.mkdir()
    assert final.stat().st_mode & 0o777 == plain.stat().st_mode & 0o777


@pytest.mark.parametrize(
    "record",
    ["{", "{}", "[]", '{"retrieved_at": 5}'],
    ids=["not-json", "no-retrieved-at", "not-an-object", "not-a-string"],
)
def test_a_snapshot_with_a_bad_record_is_fetched_again(tmp_path, caplog, record):
    source = pin(BODY)
    opener = FakeOpener(BODY)
    path = ensure(source, tmp_path, open_url=opener)
    (path.parent / RECORD_NAME).write_text(record, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        ensure(source, tmp_path, open_url=opener)
    assert len(opener.calls) == 2
    assert f"replacing snapshot {path.parent} of {source.name}" in caplog.text


def test_an_oversized_download_is_refused(tmp_path):
    source = pin(BODY)
    with pytest.raises(ContractError, match="more than the pinned"):
        ensure(source, tmp_path, open_url=FakeOpener(BODY + b"!"))
    assert not snapshot_dir(source, tmp_path).exists()


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection refused"),
        urllib.error.HTTPError(
            "https://example.invalid", 404, "Not Found", email.message.Message(), None
        ),
        TimeoutError("timed out"),
        http.client.IncompleteRead(b""),
        ConnectionResetError("reset"),
    ],
    ids=["url-error", "http-404", "timeout", "incomplete-read", "connection-reset"],
)
def test_network_failures_are_unavailable(tmp_path, error):
    source = pin(BODY)
    with pytest.raises(UnavailableError, match=source.url):
        ensure(source, tmp_path, open_url=FakeOpener(error=error))
    assert not snapshot_dir(source, tmp_path).exists()
    assert leftovers(tmp_path, source) == []


def test_the_opener_gets_the_pinned_url_and_a_finite_timeout(tmp_path):
    source = pin(BODY)
    opener = FakeOpener(BODY)
    ensure(source, tmp_path, open_url=opener)
    assert opener.calls == [(source.url, TIMEOUT_SECONDS)]
    assert 0 < TIMEOUT_SECONDS < float("inf")


@pytest.mark.parametrize(
    "url", ["http://example.invalid/x", "file:///etc/hosts", "ftp://example.invalid/x"]
)
def test_open_https_refuses_every_other_scheme(url):
    with pytest.raises(UnavailableError, match="only https://"):
        open_https(url, 1.0)


def test_read_verified_returns_a_good_snapshot(tmp_path):
    source = pin(BODY)
    plant(tmp_path, source, BODY)
    assert read_verified(source, tmp_path) == BODY


def test_read_verified_refuses_a_bad_snapshot(tmp_path):
    source = pin(BODY)
    plant(tmp_path, source, b"z" * len(BODY))
    with pytest.raises(ContractError, match="pinned"):
        read_verified(source, tmp_path)


def test_read_verified_never_fetches_an_absent_snapshot(tmp_path):
    with pytest.raises(UnavailableError, match="no snapshot"):
        read_verified(pin(BODY), tmp_path)


def test_verified_path_returns_the_snapshot_file(tmp_path):
    source = pin(BODY)
    plant(tmp_path, source, BODY)
    assert verified_path(source, tmp_path) == snapshot_dir(source, tmp_path) / source.filename


def test_verified_path_refuses_an_absent_snapshot(tmp_path):
    with pytest.raises(UnavailableError, match="no snapshot"):
        verified_path(pin(BODY), tmp_path)


def test_verified_path_refuses_changed_bytes(tmp_path):
    source = pin(BODY)
    changed = bytearray(BODY)
    changed[0] ^= 1
    plant(tmp_path, source, bytes(changed))
    with pytest.raises(ContractError, match="pinned"):
        verified_path(source, tmp_path)


def test_verify_refuses_a_missing_file(tmp_path):
    with pytest.raises(UnavailableError, match="no snapshot"):
        verify(tmp_path / "absent", pin(BODY))


def test_the_uci_pin_is_https_with_a_full_sha256():
    assert UCI_HEART_DISEASE.url.startswith("https://")
    assert len(UCI_HEART_DISEASE.sha256) == 64
    assert set(UCI_HEART_DISEASE.sha256) <= set("0123456789abcdef")


def redirect(newurl):
    request = urllib.request.Request("https://example.invalid/x")
    handler = fetch.HttpsOnlyRedirects()
    return handler.redirect_request(request, io.BytesIO(), 302, "Found", HTTPMessage(), newurl)


@pytest.mark.parametrize("newurl", ["http://example.invalid/y", "ftp://example.invalid/y"])
def test_a_redirect_off_https_is_refused(newurl):
    expected = re.escape(f"redirect from https://example.invalid/x to {newurl}")
    with pytest.raises(UnavailableError, match=expected):
        redirect(newurl)


def test_a_redirect_to_https_is_followed():
    followed = redirect("https://mirror.example.invalid/y")
    assert followed is not None
    assert followed.full_url == "https://mirror.example.invalid/y"
