---
status: accepted
topic: tooling
created: 2026-10-18
updated: 2026-10-18
---
# A session-wide guard on the Anthropic SDK's request sink

## Context

`bko.generate` calls the Anthropic Messages API, and every call costs money. The `live` marker and
the `--live` flag already skip the tests a human marked as paid, but a marker only labels what
someone remembered to mark. A test, fixture or module import that reaches the API without the
marker would spend money on every gate run, and so would a CLI test whose subprocess builds a real
client.

The SDK sends every request through one method per base class. Measured 2026-10-18 on the locked
anthropic 1.13.0:

```
grep -rln "def request(" .venv/lib/python3.14/site-packages/anthropic/
.venv/lib/python3.14/site-packages/anthropic/_base_client.py
```

`_base_client.py` defines `request` on `SyncAPIClient` and on `AsyncAPIClient`, and every resource
method (`messages.create`, `messages.stream`, `messages.count_tokens`, `models.retrieve` and the
rest) reaches it. The SDK reads `ANTHROPIC_BASE_URL` from the environment when a client is built
without a `base_url`.

## Decision Drivers

* **Must refuse every request outside a `live` test's body**, including one made at collection or
  in a module- or session-scoped fixture.
* **Must keep a subprocess from spending** in a run without `--live`.
* **Must survive production code that catches broad `Exception`** around a call.
* **Should not depend on a list of methods**, which a later caller walks past.

## Decision

We will patch **`SyncAPIClient.request` and `AsyncAPIClient.request` once per session**, in
`pytest_configure` through a `pytest.MonkeyPatch` of the guard's own, and undo the patch in
`pytest_unconfigure`. The patched sink is open only while `pytest_runtest_call` runs a test marked
`live`, and a `live` test runs only under `--live`. Anywhere else it refuses with `pytest.fail`,
whose message starts "a request outside a `live` test's body reached the Anthropic API". Without
`--live` the same patcher sets `ANTHROPIC_BASE_URL` to `http://127.0.0.1:9`, the loopback discard port.

## Rationale

The sink is the one place every request passes, so a method the SDK adds later is refused without
the guard learning its name. Configure time precedes collection and every fixture, so nothing in
the session reaches the network first. The guard's own patcher is outside any test's
`monkeypatch`, so a test calling `monkeypatch.undo()` cannot restore the real sink. `pytest.fail`
raises a `BaseException`, which an `except Exception` around a call does not swallow. A patch does
not cross a process boundary, so the discard base URL covers children: they inherit it and the SDK
reads it before any profile.

## Consequences

**Gained**: no test, fixture, import or subprocess of a run without `--live` can spend money, and a
test that tries fails with a message naming the fix.
**Accepted**: the patch is in-process; the environment variable covers subprocesses only without
`--live`, so under `--live` a child is unguarded. The guard depends on the private
`anthropic._base_client` at the locked version. A credential provider that opens its own
`httpx2.Client` runs inside a guarded request today (`lib/credentials/_providers.py:435` on
1.13.0), so the guard does not see its traffic separately. `tests/conftest.py` imports anthropic,
so every pytest run needs the `llm` extra, which `make setup` installs.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A session-scoped autouse fixture | Collection-time code runs before any fixture, so a request at import escapes it | 2026-10-18 |
| A per-test monkeypatch | Wider-scoped fixtures run first, and the test's `monkeypatch.undo()` disarms it | 2026-10-18 |
| Raising in the client constructor | Building a client is free; only a request costs, and tests build clients to check configuration | 2026-10-18 |
| Patching a list of resource methods | A denylist: a method added later or not listed walks past it | 2026-10-18 |
| The `live` marker alone | It labels only the tests a human remembered to mark | 2026-10-18 |

## Enforcement

- Owner: `tests/conftest.py`
- Pinned by: `tests/test_paid_guard.py`, `tests/test_paid_guard_sessions.py`

## Reversal Conditions

The SDK stops routing every request through one `request` per base class, or a credential provider
starts requesting outside it. The check is
`tests/test_paid_guard.py::test_no_client_class_overrides_the_patched_sink` going red after a lock
bump.

## Related

- [uv, ruff and ty, wrapped by a Makefile](uv-ruff-ty.md)
