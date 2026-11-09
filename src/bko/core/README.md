# `bko.core`: helpers no stage owns

Hold the code that two or more stages import and that belongs to neither. Today that is the stamped logging setup: the line format, its local-time-with-offset stamp, and the one `configure` every entry point calls.

A helper joins only when two or more stages import it and it belongs to neither; a stage's own public API stays in that stage (ADR [`tooling/src-layout`](../../../docs/adr/tooling/src-layout.md)). `bko.core` imports no other `bko` subpackage.

- **Dependencies:** No optional extra.
- **Status:** built.
