# `bko.fusion`: validated records back into the BKO

Join the generated records that pass validation to the training records of the cell each was asked for, list every gap again, and count what closed: the completeness criterion's reading of "after fusion".

`join` adds each requested cell's accepted records to the training split and `closure` reads each gap definition's (cell, statement) pairs before and after. The stage calls no `fuse`: the joined records reach the BKO through the dataset fragments `bko.gaps.detect.represent` fuses, as any dataset's records do. ADR [`research/gap-closure-design`](../../../docs/adr/research/gap-closure-design.md) records which records join and why, [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#gap-closure) holds the command, and [`docs/EVIDENCE.md`](../../../docs/EVIDENCE.md#c1-gap-closure-after-fusion) (C1) holds its runs.

- **Dependencies:** The `data` extra (`pyproject.toml`), `bko.gaps` with `bko.ontology` and its `ontology` extra, `bko.validate`, and `bko.generate.candidates`, which imports no SDK.
- **Owner:** Shirley Asangi
- **Status:** built.
