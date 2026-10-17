@AGENTS.md

## Claude Code

`.claude/settings.json` allows the gate's Make targets (`lint`, `typecheck`, `test`, `sizes`,
`synced`, `rebased`, `check`) and `format`, which rewrites files. None of them installs packages or
re-locks; `make setup` and `make sync` install and stay behind a prompt. It also allows
`gh pr merge`, so a plan's recorded review-fix-merge flow runs without stopping at the merge. Personal overrides go in
`.claude/settings.local.json`, which is gitignored.
