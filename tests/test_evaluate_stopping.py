"""`bko.evaluate.stopping`: an operator's signal or `STOP` file stops a supervised run."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from bko.evaluate import stopping
from bko.evaluate.stopping import STOP_FILE, OperatorStop


def test_a_backoff_waits_its_seconds_and_ends_early_on_the_stop_file(tmp_path):
    operator = OperatorStop(tmp_path / STOP_FILE)
    began = time.monotonic()
    operator.wait(0.3)
    assert time.monotonic() - began >= 0.3
    assert not operator.stopping
    (tmp_path / STOP_FILE).touch()
    began = time.monotonic()
    operator.wait(60.0)
    assert time.monotonic() - began < 1.0
    assert operator.by == "STOP"


def test_a_signal_before_the_child_starts_is_forwarded_as_sigint_and_the_handlers_restored(
    tmp_path,
):
    before = signal.getsignal(signal.SIGTERM)
    child = "import signal; print(signal.sigwait({signal.SIGINT}) == signal.SIGINT)"
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-c", child], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE
        )
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    operator = OperatorStop(tmp_path / STOP_FILE)
    with operator.installed():
        assert signal.getsignal(signal.SIGTERM) == operator.handle
        os.kill(os.getpid(), signal.SIGTERM)
        operator.started(process.pid)
        operator.handle(signal.SIGHUP, None)
        stdout, _ = process.communicate(timeout=30)
        operator.ended()
    assert signal.getsignal(signal.SIGTERM) is before
    assert stdout == b"True\n"
    assert operator.by == "SIGTERM"
    assert operator.stopping


def test_only_the_first_signal_is_forwarded_and_never_to_an_ended_child(tmp_path, monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(stopping.os, "kill", lambda pid, number: sent.append((pid, number)))
    operator = OperatorStop(tmp_path / STOP_FILE)
    operator.started(4242)
    operator.handle(signal.SIGINT, None)
    operator.handle(signal.SIGTERM, None)
    operator.ended()
    operator.started(4343)
    assert sent == [(4242, signal.SIGINT)]
    assert operator.by == "SIGINT"
    unstarted = OperatorStop(tmp_path / STOP_FILE)
    unstarted.handle(signal.SIGHUP, None)
    unstarted.ended()
    assert sent == [(4242, signal.SIGINT)]


def test_a_child_gone_before_the_forward_is_no_error(tmp_path, monkeypatch):
    def gone(pid: int, number: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(stopping.os, "kill", gone)
    operator = OperatorStop(tmp_path / STOP_FILE)
    operator.started(4242)
    operator.handle(signal.SIGTERM, None)
    assert operator.stopping
