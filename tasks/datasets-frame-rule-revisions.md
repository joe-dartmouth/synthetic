---
id: datasets-frame-rule-revisions
title: Code implausible zeros as not measured, take tertiles over kept records, and read PhysioNet's MAP and NIMAP together
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/support2-frame, research/nhanes-frame, research/physionet2012-frame, research/ontology-mapping-design
---

# Code implausible zeros as not measured, take tertiles over kept records, and read PhysioNet's MAP and NIMAP together

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Three frame rules were decided on 2026-11-26, each replacing a reading the frame ADRs record:

- **Recorded zeros.** A physiologically implausible 0, in a heart rate, a mean arterial or other
  blood pressure, a temperature, a white cell count or a field like them, codes as not measured.
  ADR [`research/support2-frame`](../docs/adr/research/support2-frame.md) reads SUPPORT2's zeros
  as measured, so a `meanbp` of 0 shows Hypotension and a `wblc` of 0 Decreased total leukocyte
  count; ADR [`research/physionet2012-frame`](../docs/adr/research/physionet2012-frame.md) reads
  PhysioNet 2012's the same way, so a `Temp` of 0 shows Hypothermia and a `MAP` of 0 Hypotension.
  The rule holds for SUPPORT2 and PhysioNet 2012 now and for every frame added after them.
- **Tertile base.** A field coded at tertiles takes them over the records the frame keeps, not
  every record of the published file. ADR
  [`research/nhanes-frame`](../docs/adr/research/nhanes-frame.md) takes `BMXBMI`'s over every
  record of BMX_L, children's values among them, though the frame keeps adults 20 and over; ADR
  [`research/support2-frame`](../docs/adr/research/support2-frame.md) takes its seven tertile
  fields' over the file's 9,105 rows, though the frame drops the one record with a missing
  `scoma`.
- **PhysioNet's `map_min`.** The minimum over the stay's `MAP` and `NIMAP` readings together,
  as every other variable takes its minimum over all its readings and a criterion is met when any
  reading crosses it. The ADR reads `NIMAP` only where the stay has no `MAP`, which leaves 653
  stays without Hypotension though a cuff reading fell below 70.

ADR [`research/ontology-mapping-design`](../docs/adr/research/ontology-mapping-design.md)
states the coded-state rule every frame follows, tertiles over the published file among it.

The set's rule makes a mapping change a new dataset name unless no `RUNS` entry reads the
dataset (`direct-index.md`, What every unit relies on), since old run folders are re-read under
identity checks. The three frames change under their own names by that exception: every run
`bko.evaluate.named_runs.RUNS` names reads `uci`, `synthea`, `diabetes`, `diabetes-codes`,
`tableshift-id` or `tableshift-ood`, so `support2`, `nhanes` and `physionet2012` hold no scored
run folder, and `tests/test_unrun_frames.py` fails if a `RUNS` entry comes to read one. Their gap
listings and contradiction censuses are readings over the training split, rerun here.

## Scope

**In.**

- Recorded zeros, in SUPPORT2's loader and PhysioNet 2012's: a 0 in a listed field codes as not
  measured, state `0`. The lists, read from `bko.datasets.support2` and
  `bko.datasets.physionet2012`:
  - SUPPORT2: `hrt`, `temp`, `wblc`, `meanbp`, `crea`, `bili` and `pafi` of `CITED`, and `sod`,
    `alb`, `ph`, `glucose` and `bun` of `TERTILES`. In the pinned archive `hrt`, `wblc`,
    `meanbp` and `glucose` hold zeros.
  - PhysioNet 2012: the readings of `HR`, `Temp`, `WBC`, `MAP`, `NIMAP`, `Creatinine`,
    `Bilirubin`, `Platelets` and `GCS`, whose scale runs from 3. In the pinned archive `HR`,
    `Temp`, `WBC`, `MAP`, `NIMAP` and `Bilirubin` hold zeros.
  - Left out, since a 0 is plausible: SUPPORT2's `resp` and PhysioNet 2012's `RespRate`, which a
    patient on a ventilator can record with no breath of their own; SUPPORT2's `urine`, an
    anuric day, and `edu`, no schooling; SUPPORT2's `scoma`, whose 0 is the score's normal; the
    counts `num.co` and `hday`; and `MechVent`, a flag.

  In PhysioNet 2012 a 0 reading is not a reading, so a stay's minimum and maximum come from its
  other readings, and a stay with no other reading of the variable is not measured. Each frame
  ADR lists the fields and each field's count of zeros, counted from the pinned archive by a
  command the ADR states.
- Each loader logs, per listed field, the zeros it recoded as not measured, a field with none
  logging 0, so a rule that changes nothing shows on the log.
- Tertiles over kept records: NHANES's `BMXBMI` over the frame's adults, SUPPORT2's `edu`, `sod`,
  `alb`, `ph`, `glucose`, `bun` and `urine` over its kept records once the zero rule has removed
  its zeros, and any other tertile field whose base differs from the records its frame keeps,
  each re-derived by the ADR's command, which prints each field's base count beside its two
  cuts, and its `TERTILES` entry and field description updated. Each loader logs, per tertile
  field, its cuts and the count of kept records measured in it.
- PhysioNet 2012's `map_min` as the minimum over `MAP` and `NIMAP` readings together. The
  `map_from_nimap` count gives way to three counts on the `cohort` line: `map_min_nimap`, the
  records whose minimum is a `NIMAP` reading with no `MAP` reading; `map_min_nimap_lower`, the
  records holding both whose `NIMAP` minimum is the lower; and `map_unmeasured`, the records with
  no `MAP` or `NIMAP` reading once zeros are removed.
- ADR edits, each with `updated` bumped and a Ruled Out row dated 2026-11-26 for the reading it
  replaces: `research/support2-frame` (zeros as measured; tertiles over the published file),
  `research/nhanes-frame` (tertiles over every record of BMX_L), `research/physionet2012-frame`
  (zeros as valid values; `NIMAP` only where no `MAP`, and the 2026-11-25 row that rules out the
  minimum over both leaves). Each of the three records that its frame changed under its own
  name because no `RUNS` entry read it, and names the commit at which the frame changed and the
  frame's new digest beside the old one, so the earlier `d1` and `d2` readings of the same
  dataset name are told apart from the new.
- ADR `research/ontology-mapping-design`'s coded-state rule states both rules as standing rules
  for every frame, once the code matches: tertiles over the records the frame keeps, and an
  implausible 0 not measured, each frame ADR listing its fields.
- The gap listing (`python -m bko.gaps support2`, `nhanes` and `physionet2012`) and the
  contradiction census (`python -m bko.evaluate contradictions --dataset` each) rerun at this
  unit's commit into new run ids, recorded in each frame ADR's Consequences and in the
  [D1](../docs/EVIDENCE.md#d1-graded-grounding-and-the-frequency-aware-level) and
  [D2](../docs/EVIDENCE.md#d2-data-contradictions-census) Runs rows.

**Out.**

- Any frame other than SUPPORT2, NHANES and PhysioNet 2012, and any frame a named run reads.
- New dataset names for the three frames.
- Any change to a cited cutoff, a row's concept or a frame's cohort.

## Acceptance

- For each listed field, a fixture record or stay holding a 0 codes `0`, and a 0 in a field the
  lists leave out, such as SUPPORT2's `urine`, still codes as measured, with the tertiles over
  that field unchanged by the zero rule.
- A fixture stay whose `Temp` readings are 0 and 37 codes `temp_min` and `temp_max` from 37
  alone, and one whose only `MAP` reading is 0, with no `NIMAP`, codes `map_min` not measured.
- A fixture stay with a lowest `MAP` of 75 and a lowest `NIMAP` of 65 codes `map_min` below 70,
  and one with a `MAP` of 0 and a lowest `NIMAP` of 80 does not.
- A fixture in which the dropped records move a tertile codes at the kept records' tertile, a
  literal value.
- Each loader's log holds a zero-count line per listed field, 0 included, and a tertile line
  per tertile field with its cuts and base count; PhysioNet 2012's `cohort` line holds
  `map_min_nimap=`, `map_min_nimap_lower=` and `map_unmeasured=` and no `map_from_nimap=`.
- Each ADR's tertile table and zero counts match what its stated commands print at this unit's
  commit.
- `python -m bko.ontology check` passes, and the three frames' pinned tests pass with
  `--releases`.
- `tests/test_unrun_frames.py` passes: `RUNS` still names no run on `support2`, `nhanes` or
  `physionet2012`.
- `make check` passes.
