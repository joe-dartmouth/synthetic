# Architecture Decision Records

This directory records the significant design decisions for the project, one file per decision, in
[MADR](https://adr.github.io/madr/) format. An ADR captures a single decision: its context, the
choice made, the alternatives weighed, and the consequences that follow.

A record's **id** is its path under `docs/adr/` minus the extension, for example
`tooling/uv-ruff-ty`. There are no ADR numbers.

## There is no index

Discover records by globbing `docs/adr/**/*.md` and reading their frontmatter. Nothing here lists
them: a hand-maintained index turns every concurrent ADR write into a merge conflict on the same
lines, and a shared sequence counter does the same on the next number. Adding a record to an
existing topic creates exactly one file and edits none.

`README.md` and `template.md` match that glob and are not decisions. Exclude them by name wherever
the glob enumerates the log.

[`tests/test_adr_frontmatter.py`](../../tests/test_adr_frontmatter.py) enforces the invariants an
index would make checkable by eye: frontmatter shape, status values, the date rule, section order,
and that no citation anywhere names a record that does not exist.

## Topics

A topic is a single directory level naming the area a decision constrains. The names come from this
repository's own boundaries.

- `tooling`: how the repository is built and worked in: toolchain, package layout, the gate, and
  the conventions code and documents are held to.

- `research`: decisions about the research method itself rather than the system.

Adding a topic is the one case that edits this file.

## Slugs

A kebab-case noun phrase naming the decision, unique within its topic. No number, no date, no
author. Ordering comes from `created`, not from a filename.

## Frontmatter

Every record opens with a flat YAML block:

```yaml
---
status: accepted        # proposed | accepted
topic: tooling
created: 2026-10-14     # first written; immutable
updated: 2026-10-14     # last substantive edit; equals created on a new record
---
```

- **`created` never changes.** It records when the record was first written.
- **`updated` is the date of the last substantive edit**, the timeliness signal a reader starts
  from. Fixing a typo is not substantive; changing what the record asserts is. Nothing checks it
  beyond `created <= updated`.
- **The path is authoritative for `topic`.** The field mirrors the containing directory; on
  mismatch the directory wins.

There are no `supersedes` or `superseded-by` fields and no `superseded`, `deprecated` or `rejected`
status: a record whose decision no longer exists is deleted, so there is no file at either end of
such a link.

## Sections

Nine, in this order, asserted by equality: nothing extra, nothing missing, no reordering.

    Context, Decision Drivers, Decision, Rationale, Consequences,
    Ruled Out, Enforcement, Reversal Conditions, Related

- **`## Ruled Out` is a table of named ideas, never numbered ones**, each row with a one-line reason
  and the ISO date it was weighed. A table row has no anchor, so "Option 2" stops resolving when the
  table is reordered. An argument longer than a cell belongs in `## Rationale`, and the cell points
  at it.
- **`## Enforcement` names an owner and what pins it.** `- Owner:` is the path the decision governs;
  `- Pinned by:` is a real test, or the literal `nothing -- convention only` for a convention
  nothing checks. Every code span in the section that names a path under a tracked top-level
  directory is checked to exist.
- **A pin claim lives in `## Enforcement` and nowhere else in a record** (ADR
  [`tooling/prose-describes-guards-never-licenses-changes`](tooling/prose-describes-guards-never-licenses-changes.md)).
- **`## Reversal Conditions` is checkable.** A future reader must be able to tell whether the
  condition has happened; "if requirements change" cannot be told either way.

## Keeping a record current

A record describes the decision as it stands today. A record that disagrees with the system is a
defect in the record.

| Situation | Action | Result |
|-----------|--------|--------|
| Clarification, corrected detail, or added consequence | Edit the text in place and bump `updated` | stays `accepted` |
| The decision itself changes | Edit `## Decision` and `## Rationale` in place, add the previous approach as a dated `## Ruled Out` row, bump `updated` | stays `accepted` |
| The decision no longer exists | **Delete the file.** Move anything still worth knowing into the successor | file removed |
| One decision splits into two, or two merge into one | Write the new records, carry forward what still applies, delete the old file | file removed |

Name every deletion in the change summary, since nothing else puts it in front of a reviewer. A
successor names its predecessor as unlinked prose in italics with its date, never as a link or a
backticked id, because both of those forms are checked and would fail on a deleted file. Git holds
the history: `git log -p docs/adr/<topic>/<slug>.md`.

## Adding a record

1. Copy `template.md` to `<topic>/<slug>.md`, creating the topic directory (and its bullet above)
   if it is new.
2. Fill in the sections; keep it to the decision, not a tutorial.
3. Set `created` and `updated` to today and `status: proposed`, then `accepted` once agreed.
4. Write `## Reversal Conditions` last, and make it checkable.
