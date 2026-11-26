# `bko.core`: helpers no stage owns

Hold the code that two or more stages import and that belongs to neither. Today that is the
stamped logging setup: the line format (`FORMAT`), its local-time-with-offset stamp (`Stamp`), the
handler that writes it (`stamped_handler`), and the one `configure` every entry point calls; and
the `versions` line (`bko.core.versions.log_versions`: the commit, whether its tree is dirty, and
the libraries' versions), which `bko.gaps` and `bko.evaluate` both log and
`bko.evaluate.provenance` re-exports.

A helper joins only when two or more stages import it and it belongs to neither; a stage's own
public API stays in that stage (ADR
[`tooling/src-layout`](../../../docs/adr/tooling/src-layout.md)). `bko.core` imports no other
`bko` subpackage.

- **Dependencies:** No optional extra.
- **Status:** built.
