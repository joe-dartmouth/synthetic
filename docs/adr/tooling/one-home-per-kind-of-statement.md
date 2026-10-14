---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-14
---
# One home per kind of statement

## Context

A fact written in two documents drifts: one copy is edited and the other is not, and a reader
cannot tell which is current. Agent guidance makes this worse, because the same instructions are
wanted by Claude Code, Codex and Gemini CLI, each of which reads a different file.

## Decision Drivers

* **Must give every kind of statement exactly one document that owns it.**
* **Should let every other document point at the owner rather than copy it.**

## Decision

Each kind of statement has one home:

| Kind of statement | Home |
|---|---|
| Project summary for a human arriving at the repository | `README.md` |
| Working guidance for agents and contributors: commands, layout, gotchas, conventions, document index | `AGENTS.md` |
| Environment, platforms, every Make target, gate hazards | `docs/DEVELOPMENT.md` |
| A design decision, its alternatives and its reversal conditions | `docs/adr/<topic>/<slug>.md` |
| What each repository checker does and its exit-code contract | `tools/README.md` |
| A pipeline stage's role, extra, owner and status | `src/bko/<stage>/README.md` |
| A constant's value | The one module that defines it (for example `CAPS` in `tools/check_file_size.py`) |

A new kind of statement adds a row here. Other documents link to the home, and quote nothing from
it that can change.

## Rationale

Pointing costs a click; copying costs a silent contradiction later. Holding agent guidance in
`AGENTS.md` alone, with `CLAUDE.md` and `GEMINI.md` importing it, applies the same rule to the three
tools' configuration files.

## Consequences

**Gained**: a reader who finds a statement knows it is the current one.
**Accepted**: documents link more and summarize less; a reader follows links for detail.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| One long README holding everything | Mixes audiences and grows past what anyone reads | 2026-10-13 |
| A copy of agent guidance per tool | Three copies drift | 2026-10-13 |
| A wiki outside the repository | Not versioned with the code it describes | 2026-10-13 |

## Enforcement

- Owner: `docs/`
- Pinned by: nothing -- convention only

## Reversal Conditions

**A statement that fits no row adds a row, not a second copy.** If review finds the same statement
maintained in two homes twice, the two homes merge or the table splits the kind more finely.

## Related

- [Agent config in a single home](agent-config-single-home.md)
- [Prose describes guards; it never licenses changes](prose-describes-guards-never-licenses-changes.md)
