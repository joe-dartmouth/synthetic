---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-14
---
# Prose describes guards; it never licenses changes

## Context

A repository argues from prose about its own guards. A docstring says which mutation a case
catches, a comment names the layer a scope sits in, and an ADR says a test holds a claim. The test
suite runs the code and reads none of those sentences, so a sentence about a guard can be false
while every test stays green. In the reference repository this happened: an ADR argued that a
section could not be deleted because a test read its heading, the test never read the heading, and
the false argument kept the section alive.

## Decision Drivers

* **Must stop a sentence about a guard from substituting for the guard.**
* **Should keep every claim that a test holds a decision where its path is checked.**

## Decision

We will use **an authoring rule, plus one guard that keeps pin claims where they are checked**.

1. **Prose may describe what a guard checks. It never licenses a change.** No change is made, and
   none is refused, because a comment, a docstring or an ADR says a guard covers something. The
   argument for a change is the guard itself: run it, plant the mutation, or read its assertion.
2. **A passive pin claim lives in an ADR's `## Enforcement` section**, where every code span naming
   a path under a tracked top-level directory is checked to exist. Anywhere else in a record, the
   guard refuses it.
3. **The guard matches *pinned* followed by *by*, *in* or *against*,** in any case, with a colon
   allowed after them and the two words split across a line break. It skips the field's bare label
   in a code span, fenced code, and a content digest or a hash of a file's bytes, which fix an
   input's identity rather than naming a guard.

## Rationale

Checking every code location named in prose is not feasible: prose names layers, functions and
behaviors in forms no parser resolves. Moving the one checkable form, the pin claim, into the one
section whose paths are checked turns a class of silent falsehood into a test failure, and the rule
covers the rest by making the guard, not the sentence, the evidence.

## Consequences

**Gained**: a pin claim in an ADR names a path that exists; a reviewer knows to ask for the guard
rather than trust the sentence.
**Accepted**: the active voice ("`tests/x.py` holds this") and a word between the verb and its
preposition pass the guard; the rule alone covers them.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Resolve every code location named in prose | No parser resolves prose references to layers and behaviors | 2026-10-13 |
| No rule | A false sentence about a guard reads as evidence | 2026-10-13 |

## Enforcement

- Owner: `docs/adr/`
- Pinned by: `tests/test_adr_frontmatter.py`
- `test_no_pin_claim_sits_outside_enforcement` refuses the passive claim outside `## Enforcement`,
  and `test_every_enforcement_path_resolves` checks every path the section names.

## Reversal Conditions

**The guard widens to the active voice the first time review finds a false claim written in it.**
Until then the active form stays a documented limit, because widening it would refuse ordinary
descriptive sentences.

## Related

- [One home per kind of statement](one-home-per-kind-of-statement.md)
