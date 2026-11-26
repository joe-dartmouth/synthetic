"""Offline PhysioNet 2012 inputs: stays, outcomes files and the three sets' archives, in memory.

The real data here is `REAL_STAY` and `REAL_OUTCOME`, set A's stay 134845 and its outcome line
byte for byte, kept so the parser is proven against the real layout. Every other stay is built
by `stay` with invented observations.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from bko.datasets.physionet2012 import HEADER, OUTCOMES_HEADER, SETS, TARRED
from bko.datasets.sources import Source
from tests._datasets_fixtures import pin, plant


REAL_STAY: Final = (
    "Time,Parameter,Value\n"
    "00:00,RecordID,134845\n"
    "00:00,Age,76\n"
    "00:00,Gender,1\n"
    "00:00,Height,-1\n"
    "00:00,ICUType,3\n"
    "00:00,Weight,75\n"
    "01:52,Lactate,1.2\n"
    "02:18,FiO2,0.5\n"
    "02:18,GCS,3\n"
    "02:18,HR,89\n"
    "02:18,MechVent,1\n"
    "02:18,NIDiasABP,53\n"
    "02:18,NIMAP,74.33\n"
    "02:18,NISysABP,117\n"
    "02:18,Temp,35.7\n"
    "02:48,HR,118\n"
    "02:48,NIDiasABP,75\n"
    "02:48,NIMAP,94.67\n"
    "02:48,NISysABP,134\n"
    "03:48,HR,99\n"
    "03:48,NIDiasABP,68\n"
    "03:48,NIMAP,86\n"
    "03:48,NISysABP,122\n"
    "03:48,Urine,120\n"
    "04:05,pH,7.3\n"
    "04:05,Lactate,1.4\n"
    "04:05,PaCO2,45\n"
    "04:05,PaO2,171\n"
    "04:05,SaO2,98\n"
    "04:48,FiO2,0.4\n"
    "04:48,GCS,6\n"
    "04:48,HR,95\n"
    "04:48,MechVent,1\n"
    "04:48,NIDiasABP,64\n"
    "04:48,NIMAP,79\n"
    "04:48,NISysABP,109\n"
    "04:48,Temp,35.8\n"
    "04:48,Urine,120\n"
    "05:44,pH,7.33\n"
    "05:44,PaCO2,44\n"
    "05:44,PaO2,151\n"
    "05:48,HR,69\n"
    "05:48,NIDiasABP,44\n"
    "05:48,NIMAP,57.67\n"
    "05:48,NISysABP,85\n"
    "05:48,Urine,80\n"
    "06:03,DiasABP,49\n"
    "06:03,HR,61\n"
    "06:03,NIDiasABP,43\n"
    "06:03,NIMAP,58.33\n"
    "06:03,NISysABP,89\n"
    "06:03,SysABP,101\n"
    "06:18,DiasABP,48\n"
    "06:18,HR,59\n"
    "06:18,MAP,69\n"
    "06:18,NIDiasABP,51\n"
    "06:18,NIMAP,67.33\n"
    "06:18,NISysABP,100\n"
    "06:18,SysABP,102\n"
    "06:48,DiasABP,55\n"
    "06:48,HR,60\n"
    "06:48,MAP,80\n"
    "06:48,NIDiasABP,44\n"
    "06:48,NIMAP,63\n"
    "06:48,NISysABP,101\n"
    "06:48,SysABP,119\n"
    "06:48,Urine,60\n"
    "07:48,DiasABP,52\n"
    "07:48,HR,61\n"
    "07:48,MAP,77\n"
    "07:48,SysABP,117\n"
    "07:48,Urine,80\n"
    "08:48,DiasABP,53\n"
    "08:48,GCS,6\n"
    "08:48,HR,71\n"
    "08:48,MAP,77\n"
    "08:48,NIDiasABP,55\n"
    "08:48,NIMAP,74\n"
    "08:48,NISysABP,112\n"
    "08:48,SysABP,122\n"
    "08:48,Temp,34.6\n"
    "08:48,Urine,60\n"
    "09:06,ALP,212\n"
    "09:06,ALT,43\n"
    "09:06,AST,68\n"
    "09:06,Bilirubin,1.2\n"
    "09:06,BUN,19\n"
    "09:06,Creatinine,0.7\n"
    "09:06,Glucose,74\n"
    "09:06,HCO3,21\n"
    "09:06,HCT,34.2\n"
    "09:06,Mg,1.4\n"
    "09:06,Platelets,98\n"
    "09:06,K,3.9\n"
    "09:06,Na,136\n"
    "09:06,WBC,6.2\n"
    "09:11,pH,7.35\n"
    "09:11,Lactate,0.9\n"
    "09:11,PaCO2,39\n"
    "09:11,PaO2,156\n"
    "09:11,SaO2,98\n"
    "09:48,DiasABP,47\n"
    "09:48,FiO2,0.4\n"
    "09:48,HR,89\n"
    "09:48,MAP,69\n"
    "09:48,MechVent,1\n"
    "09:48,SysABP,113\n"
    "09:48,Urine,40\n"
    "10:48,DiasABP,46\n"
    "10:48,HR,86\n"
    "10:48,MAP,68\n"
    "10:48,NIDiasABP,81\n"
    "10:48,NIMAP,95.67\n"
    "10:48,NISysABP,125\n"
    "10:48,SysABP,114\n"
    "10:48,Urine,40\n"
    "12:33,DiasABP,70\n"
    "12:33,HR,123\n"
    "12:33,MAP,97\n"
    "12:33,SysABP,144\n"
    "12:33,Urine,180\n"
    "12:48,FiO2,0.4\n"
    "13:03,HCT,34.2\n"
    "13:04,pH,7.33\n"
    "13:04,PaCO2,41\n"
    "13:04,PaO2,102\n"
    "13:18,DiasABP,66\n"
    "13:18,HR,97\n"
    "13:18,MAP,89\n"
    "13:18,SysABP,132\n"
    "13:18,Temp,35.3\n"
    "14:48,DiasABP,67\n"
    "14:48,GCS,15\n"
    "14:48,HR,84\n"
    "14:48,MAP,92\n"
    "14:48,SysABP,140\n"
    "14:48,Urine,140\n"
    "15:48,DiasABP,63\n"
    "15:48,HR,86\n"
    "15:48,MAP,87\n"
    "15:48,SysABP,134\n"
    "16:48,DiasABP,61\n"
    "16:48,HR,81\n"
    "16:48,MAP,85\n"
    "16:48,SysABP,129\n"
    "16:48,Temp,35.9\n"
    "16:48,Urine,160\n"
    "18:48,DiasABP,45\n"
    "18:48,HCT,31.6\n"
    "18:48,HR,100\n"
    "18:48,MAP,64\n"
    "18:48,SysABP,108\n"
    "18:48,Urine,100\n"
    "19:48,DiasABP,49\n"
    "19:48,HR,90\n"
    "19:48,MAP,69\n"
    "19:48,SysABP,107\n"
    "19:48,Urine,130\n"
    "20:48,DiasABP,53\n"
    "20:48,HR,76\n"
    "20:48,MAP,75\n"
    "20:48,SysABP,117\n"
    "20:48,Temp,35.7\n"
    "22:18,HCT,32.5\n"
    "22:48,DiasABP,60\n"
    "22:48,HR,80\n"
    "22:48,MAP,82\n"
    "22:48,SysABP,124\n"
    "22:48,Urine,175\n"
    "24:48,DiasABP,54\n"
    "24:48,HR,78\n"
    "24:48,MAP,75\n"
    "24:48,SysABP,117\n"
    "26:48,DiasABP,48\n"
    "26:48,HR,83\n"
    "26:48,MAP,68\n"
    "26:48,SysABP,106\n"
    "28:48,DiasABP,54\n"
    "28:48,GCS,15\n"
    "28:48,HCT,31.9\n"
    "28:48,HR,84\n"
    "28:48,MAP,72\n"
    "28:48,SysABP,103\n"
    "28:48,Temp,36.9\n"
    "28:48,Urine,340\n"
    "30:48,DiasABP,49\n"
    "30:48,HR,77\n"
    "30:48,MAP,66\n"
    "30:48,SysABP,101\n"
    "32:48,DiasABP,71\n"
    "32:48,HR,78\n"
    "32:48,MAP,86\n"
    "32:48,SysABP,104\n"
    "34:48,DiasABP,58\n"
    "34:48,HCT,32.8\n"
    "34:48,HR,78\n"
    "34:48,MAP,80\n"
    "34:48,Platelets,98\n"
    "34:48,SysABP,122\n"
    "34:48,Urine,340\n"
    "36:48,DiasABP,56\n"
    "36:48,HR,82\n"
    "36:48,SysABP,125\n"
    "36:48,Temp,36.6\n"
    "37:24,BUN,12\n"
    "37:24,Creatinine,0.7\n"
    "37:24,Mg,1.9\n"
    "37:24,K,3.9\n"
    "37:48,Urine,180\n"
    "38:48,DiasABP,75\n"
    "38:48,HR,92\n"
    "38:48,MAP,122\n"
    "38:48,SysABP,150\n"
    "39:18,Urine,120\n"
    "40:48,DiasABP,66\n"
    "40:48,HR,70\n"
    "40:48,MAP,90\n"
    "40:48,NIDiasABP,61\n"
    "40:48,NIMAP,77.67\n"
    "40:48,NISysABP,111\n"
    "40:48,SysABP,136\n"
    "40:48,Urine,80\n"
    "42:48,HR,85\n"
    "44:48,HR,83\n"
    "44:48,NIDiasABP,58\n"
    "44:48,NIMAP,78.33\n"
    "44:48,NISysABP,119\n"
    "44:48,Temp,37\n"
    "44:48,Urine,140\n"
    "46:18,HCT,31.9\n"
)
"""Set A's stay 134845, byte for byte: a man of 76 in a medical ICU, ventilated, who survived."""

REAL_OUTCOME: Final = "134845,16,10,5,223,0"
"""The outcome line of stay 134845 in `Outcomes-a.txt`, byte for byte."""

REAL_ID: Final = 134845


def stay(
    record_id: int,
    *observations: tuple[str, str, str],
    age: str = "60",
    gender: str = "1",
    icu: str = "3",
) -> str:
    """A stay's text: the header, the descriptors at `00:00`, then each `(time, name, value)`."""
    descriptors = (
        ("RecordID", str(record_id)),
        ("Age", age),
        ("Gender", gender),
        ("Height", "-1"),
        ("ICUType", icu),
        ("Weight", "-1"),
    )
    lines = [HEADER, *(f"00:00,{name},{value}" for name, value in descriptors)]
    lines += [",".join(observation) for observation in observations]
    return "".join(f"{line}\n" for line in lines)


def outcomes(*rows: tuple[int, int]) -> str:
    """An outcomes file holding each `(RecordID, death)` with invented severity scores."""
    lines = [OUTCOMES_HEADER, *(f"{record_id},12,4,9,-1,{death}" for record_id, death in rows)]
    return "".join(f"{line}\n" for line in lines)


def zipped(label: str, stays: Mapping[int, str]) -> bytes:
    """Set `label`'s zip: its directory entry, then each stay as `set-x/<RecordID>.txt`."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(zipfile.ZipInfo(f"set-{label}/", date_time=(2012, 5, 8, 1, 5, 0)), "")
        for record_id, text in stays.items():
            info = zipfile.ZipInfo(f"set-{label}/{record_id}.txt", date_time=(2012, 5, 8, 1, 23, 0))
            bundle.writestr(info, text)
    return buffer.getvalue()


def tarred(label: str, stays: Mapping[int, str]) -> bytes:
    """Set `label`'s gzipped tar, its gzip header time 0 so its bytes repeat: its directory entry,
    then each stay as a regular file."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        folder = tarfile.TarInfo(f"set-{label}")
        folder.type = tarfile.DIRTYPE
        bundle.addfile(folder)
        for record_id, text in stays.items():
            body = text.encode("ascii")
            info = tarfile.TarInfo(f"set-{label}/{record_id}.txt")
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    return gzip.compress(buffer.getvalue(), mtime=0)


def archived(label: str, stays: Mapping[int, str]) -> bytes:
    """Set `label`'s archive as PhysioNet serves it: a gzipped tar for set C, else a zip."""
    return tarred(label, stays) if label in TARRED else zipped(label, stays)


def mixed() -> dict[str, dict[int, tuple[str, int]]]:
    """Four stays per set across ages, sexes, ICU types, labels and readings: each set's stays
    by `RecordID`, each its text and its death."""
    sets: dict[str, dict[int, tuple[str, int]]] = {label: {} for label in SETS}
    for n in range(12):
        record_id = 140000 + n
        readings = (
            ("01:00", "HR", str(80 + 5 * n)),
            ("02:00", "Temp", f"{35.5 + 0.25 * n:g}"),
            ("03:00", "WBC", f"{3 + n:g}"),
            ("04:00", "MAP" if n % 2 else "NIMAP", str(60 + 3 * n)),
            ("05:00", "Platelets", str(100 + 10 * n)),
            ("06:00", "GCS", str(15 - n % 4)),
            *((("07:00", "MechVent", "1"),) if n % 3 == 0 else ()),
            *((("08:00", "RespRate", str(14 + n)),) if n % 3 else ()),
        )
        text = stay(
            record_id,
            *readings,
            age=str((40, 55, 70, 85)[n % 4]),
            gender=str(n % 2),
            icu=str(1 + n % 4),
        )
        sets[SETS[n // 4]][record_id] = (text, int(n % 5 == 0))
    return sets


def built(
    sets: Mapping[str, Mapping[int, tuple[str, int]]] | None = None,
    *,
    archives: Mapping[str, bytes] | None = None,
    outcome_texts: Mapping[str, str] | None = None,
) -> list[tuple[Source, bytes]]:
    """The three archives and outcomes files of `sets`, or of `mixed()`, each with its pin.

    `archives` and `outcome_texts` replace a set's built archive or outcomes text.
    """
    sets = mixed() if sets is None else sets
    files: list[tuple[str, bytes]] = []
    for label in SETS:
        held = sets.get(label, {})
        data = (archives or {}).get(label)
        if data is None:
            data = archived(label, {record_id: text for record_id, (text, _) in held.items()})
        files.append((f"physionet2012-set-{label}", data))
    for label in SETS:
        held = sets.get(label, {})
        text = (outcome_texts or {}).get(label)
        if text is None:
            text = outcomes(*((record_id, death) for record_id, (_, death) in held.items()))
        files.append((f"physionet2012-outcomes-{label}", text.encode("utf-8")))
    return [(pin(data, name), data) for name, data in files]


def planted(
    data_dir: Path,
    sets: Mapping[str, Mapping[int, tuple[str, int]]] | None = None,
    *,
    archives: Mapping[str, bytes] | None = None,
    outcome_texts: Mapping[str, str] | None = None,
) -> tuple[Source, ...]:
    """Plant `built`'s files as snapshots under `data_dir` and return their pins."""
    files = built(sets, archives=archives, outcome_texts=outcome_texts)
    for source, data in files:
        plant(data_dir, source, data)
    return tuple(source for source, _ in files)


def real_sets() -> dict[str, dict[int, tuple[str, int]]]:
    """`mixed()` with `REAL_STAY` added to set A, its death read from `REAL_OUTCOME`."""
    sets = mixed()
    sets["a"][REAL_ID] = (REAL_STAY, int(REAL_OUTCOME.split(",")[5]))
    return sets
