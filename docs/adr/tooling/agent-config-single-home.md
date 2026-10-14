---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-14
---
# Agent configuration in a single home

## Context

Teammates may work with Claude Code, Codex or Gemini CLI. Each reads a different project file:
Claude Code reads `CLAUDE.md`, Codex reads `AGENTS.md`, and Gemini CLI reads `GEMINI.md`. Claude
Code resolves an `@path` import in `CLAUDE.md`, and Gemini CLI's memory import processor resolves
`@./path` in `GEMINI.md`. Each tool also has a project settings file with different merge semantics.

## Decision Drivers

* **Must give every tool the same repository guidance**, without three copies.
* **Must not commit a setting that overrides a developer's own configuration.**
* **Should commit only narrowly scoped permission grants.**

## Decision

- **`AGENTS.md` holds all repository guidance.** `CLAUDE.md` opens with `@AGENTS.md` and
  `GEMINI.md` opens with `@./AGENTS.md`; each adds only notes specific to its tool.
- **`.claude/settings.json` is committed** with a `permissions.allow` list of exact Make targets:
  the gate's (`lint`, `typecheck`, `test`, `sizes`, `synced`, `rebased`, `check`) and `format`.
  Claude Code merges these entries with each developer's own rules. `make setup` and `make sync`
  install packages and are not allowed.
- **`.gitignore` ignores `.claude/settings.local.json` and `.claude/worktrees/`, not all of
  `.claude/`**, so the shared settings file is versioned.
- **No `.codex/config.toml`**: Codex's only project-level levers are the repo-wide
  `approval_policy` and `sandbox_mode`, which are postures for each developer rather than
  per-command grants.
- **No `.gemini/settings.json`**: a project file there overrides the developer's settings rather
  than merging with them.

## Rationale

Importing one file keeps the guidance identical across tools with no drift. A committed allow list
is a standing grant to everyone who clones the repository, so it names exact invocations and no
general-purpose runner (`uv run *`, `make *`). The targets listed run the gate or format the tree;
none installs packages or re-locks, because every one of them runs `uv run --no-sync`.

## Consequences

**Gained**: one place to edit guidance; Claude Code runs the gate without a prompt.
**Accepted**: a tool that stops resolving imports would see only its two-line file until the
guidance is inlined. The allowed targets execute repository code (`make test` runs the suite, and
`format` rewrites files), so the grant reaches whatever code a developer has accepted into the
working tree; an untracked `GNUmakefile` would take precedence over `Makefile` for every entry.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Three full copies of the guidance | They drift | 2026-10-13 |
| `CLAUDE.md` only, as the reference repository does | Codex and Gemini CLI would see nothing | 2026-10-13 |
| Ignoring all of `.claude/` | The shared settings file would not be versioned | 2026-10-13 |
| Wildcard grants such as `Bash(make *)` | A wildcard over a general runner is an arbitrary-execution grant | 2026-10-13 |

## Enforcement

- Owner: `AGENTS.md`
- Pinned by: `tests/test_repo_gate.py`
- `test_claude_and_gemini_import_agents_md` asserts the first line of `CLAUDE.md` and `GEMINI.md`,
  and `test_the_committed_allow_list_is_exactly_the_make_targets` holds the allow list to the
  exact entries above, with no wildcard.

## Reversal Conditions

**Inline the guidance into a tool's file if that tool stops resolving the import**, which its
loaded-memory view shows (`/memory show` in Gemini CLI, `/memory` in Claude Code).

**Commit a `.gemini/settings.json` if Gemini CLI documents merge semantics for project settings.**

## Related

- [One home per kind of statement](one-home-per-kind-of-statement.md)
