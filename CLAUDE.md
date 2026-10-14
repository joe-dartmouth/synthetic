@AGENTS.md

## Claude Code

`.claude/settings.json` allows the gate's Make targets (`lint`, `typecheck`, `test`, `sizes`,
`synced`, `rebased`, `check`) and `format`, which rewrites files. None of them installs packages or
re-locks; `make setup` and `make sync` install and stay behind a prompt. Personal overrides go in
`.claude/settings.local.json`, which is gitignored.
