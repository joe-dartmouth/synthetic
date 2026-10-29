"""E3's Synthea frame: one row per patient alive and 40 or over, with indicators and the label."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from bko.datasets.cells import LABEL
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, frame_records, load_population, subjects
from bko.evaluate.data import (
    SYNTHEA_ADULTS_FROM,
    SYNTHEA_FIELD_ROWS,
    SYNTHEA_FRAME,
    SYNTHEA_INDICATORS,
    digest,
)
from bko.gaps.observe import observe_population, record_cells
from bko.ontology.mapping import CODE_MAPPINGS
from tests._synthea_frame_fixtures import write_population


REPO: Path = Path(__file__).resolve().parents[1]


def _population(tmp_path: Path):
    return load_population(write_population(tmp_path), snapshot_date=date(2026, 1, 1))


def _frame(population, adults_from: int = SYNTHEA_ADULTS_FROM):
    return frame_records(
        population,
        label=ISCHEMIC_HEART_DISEASE,
        indicators=SYNTHEA_INDICATORS,
        adults_from=adults_from,
    )


def test_the_frame_holds_age_sex_indicators_and_target(tmp_path):
    records = _frame(_population(tmp_path))
    indicators = tuple(SYNTHEA_INDICATORS)
    assert SYNTHEA_FRAME.numeric == ("age",)
    assert SYNTHEA_FRAME.fractional == ()
    assert SYNTHEA_FRAME.edges == (65,)
    assert len(indicators) == len(CODE_MAPPINGS) - 1
    assert tuple(SYNTHEA_INDICATORS.values()) == tuple(row.snomed for row in CODE_MAPPINGS[1:])
    assert SYNTHEA_FRAME.columns == ("age", "sex", *indicators, LABEL)
    assert tuple(records.columns) == SYNTHEA_FRAME.columns
    assert dict(records.dtypes.astype(str)) == dict(SYNTHEA_FRAME.dtypes)
    for name in indicators:
        assert dict(SYNTHEA_FRAME.codebook[name]) == {0: "absent", 1: "present"}
        assert set(records[name]) == {0, 1}
    assert SYNTHEA_FRAME.study
    assert SYNTHEA_FRAME.cohort


def test_the_frame_excludes_the_label_code_the_deceased_and_under_40s(tmp_path):
    population = _population(tmp_path)
    records = _frame(population)
    assert len(records) == 310
    assert records["age"].min() >= 40
    assert ISCHEMIC_HEART_DISEASE not in SYNTHEA_INDICATORS.values()
    people = subjects(population, codes=frozenset({ISCHEMIC_HEART_DISEASE}))
    adults = people.loc[people["age"] >= 40]
    assert records[LABEL].tolist() == adults[LABEL].tolist()
    assert int(records[LABEL].sum()) == 35 + 35 + 65 + 35
    assert len(_frame(population, adults_from=0)) == 325


def test_record_ids_follow_sorted_patient_ids(tmp_path):
    population = _population(tmp_path)
    records = _frame(population)
    people = subjects(population, codes=frozenset({ISCHEMIC_HEART_DISEASE}))
    adults = people.loc[people["age"] >= 40]
    assert records.index.name == "record_id"
    assert records.index.tolist() == list(range(1, len(records) + 1))
    assert list(adults.index) == sorted(adults.index)
    assert records["age"].tolist() == adults["age"].tolist()
    assert records["sex"].tolist() == adults["sex"].tolist()


def test_the_frame_bytes_do_not_depend_on_hash_seed(tmp_path):
    directory = write_population(tmp_path)
    script = (
        "import sys\n"
        "from datetime import date\n"
        "from pathlib import Path\n"
        "from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, frame_records, load_population\n"
        "from bko.evaluate.data import SYNTHEA_ADULTS_FROM, SYNTHEA_INDICATORS, digest\n"
        "pop = load_population(Path(sys.argv[1]), snapshot_date=date(2026, 1, 1))\n"
        "records = frame_records(pop, label=ISCHEMIC_HEART_DISEASE,\n"
        "    indicators=SYNTHEA_INDICATORS, adults_from=SYNTHEA_ADULTS_FROM)\n"
        "print(digest(records))\n"
    )
    digests = []
    for seed in ("1", "2"):
        path = os.pathsep.join([str(REPO / "src"), str(REPO)])
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": path}
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-c", script, str(directory)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        digests.append(done.stdout.strip())
    assert digests[0] == digests[1]
    assert digests[0] == digest(_frame(_population(tmp_path)))


def test_indicator_rows_reproduce_observe_population(tmp_path):
    population = _population(tmp_path)
    records = _frame(population, adults_from=0)
    cells = record_cells(records, SYNTHEA_FIELD_ROWS, name="synthea", edges=SYNTHEA_FRAME.edges)
    observed = observe_population(population, CODE_MAPPINGS, name="synthea")
    assert dict(cells.whole.holders) == dict(observed.holders)
    assert cells.whole.records == observed.records
    assert all(count > 0 for count in observed.holders.values())
