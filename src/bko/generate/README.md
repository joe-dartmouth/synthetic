# `bko.generate`: record generation

Generate candidate records for each detected gap through the Anthropic Messages API, guided by the relevant ontology relationships.

`guidance` reads what a request says about each cell at one of six guidance levels, `prompt` renders it, and `calls` asks each cell's quota in chunks through a `Send`, which `api`, the only module that imports `anthropic`, makes real. `decode` turns each reply into candidates, unfiltered and unrepaired, with a placeholder per asked record that did not decode; `spend` prices each call; and `log` writes every request and response with its spend to an append-only JSONL call log. ADR [`research/record-generation-design`](../../../docs/adr/research/record-generation-design.md) records the design, [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#generation) holds the command, and [`docs/EVIDENCE.md`](../../../docs/EVIDENCE.md#g1-record-generation-through-the-messages-api) (G1) holds its paid runs.

- **Dependencies:** The `llm` extra (`pyproject.toml`), and `bko.gaps` with its `data` extra; the command reads releases through `bko.ontology` with its `ontology` extra.
- **Owner:** Shirley Asangi
- **Status:** built.
