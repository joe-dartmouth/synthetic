@AGENTS.md

## Claude Code

`.claude/settings.json` allows only the read-and-check Make targets (`lint`, `format`,
`typecheck`, `test`, `sizes`, `synced`, `rebased`, `check`). `make setup` and `make sync` install
packages and stay behind a prompt. Personal overrides go in `.claude/settings.local.json`, which is
gitignored.
