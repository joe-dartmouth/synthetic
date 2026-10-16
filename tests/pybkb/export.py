"""Export pybkb's example BKBs and results as the fixtures the `bko.bkb` tests read.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-16; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).

Run by hand under Python 3.11, never by the gate; `docs/DEVELOPMENT.md` (BKB fixtures) has the
commands, and `ruff.toml` holds this file to the 3.11 grammar. PYBKB_ROOT holds pybkb's files at
`COMMIT`, laid out as in its repository, and the interpreter has
`tests/pybkb/oracle-requirements.txt` installed. The script checks every file against `FILES`,
statically scans each pickle's globals against `GLOBALS`, loads the files with pybkb itself,
replays pybkb's own test assertions (`_replay`, from pybkb's `tests/test_bkb_api.py` and
`tests/test_fusion.py`), and writes into OUT_DIR, which must not exist:

- one document per example BKB and per fused BKB, in the format `bko.bkb.serialize` reads, with
  names turned into strings by `str` and I-nodes sorted;
- `queries.json`: what pybkb's `heuristic_update_complete_fused` returns for each query;
- `manifest.json`: the commit, the sha256 of every input, every output and this script, the
  interpreter and library versions, whether each BKB is mutex, the replayed checks and the
  license notice;
- `run.json`: the hash seed, the one value allowed to differ between runs.

Usage: PYTHONHASHSEED=0 python -P -s -B export.py PYBKB_ROOT OUT_DIR
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
import pickletools
import platform
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


COMMIT = "1e1b62a02faa0abc73db4597076c2d38c1313df6"

FILES = (
    ("pybkb/__init__.py", "f38b46a4f01320c2244465fb24d42847648b1353531043b212b1cfba202e5625"),
    ("pybkb/_version.py", "870389b855042adb2088eb7897aa29f6cfa1be2a2f33a4cad6e09db513566381"),
    ("pybkb/bkb.py", "bd42aa36c4864f50127c7c426616f2c5ce286a386345274ea245b87915d0a2e6"),
    ("pybkb/exceptions.py", "446fa4c7d041eb3e7c465e4737b76dfb98a96e98cdae2ccb650b457df11e26a3"),
    ("pybkb/scores.py", "21e92ebb563c210e125e9e936873f61a87bcb58246daab468e92be12ece8f7dd"),
    (
        "pybkb/fusion/__init__.py",
        "c418f57adf0e37ed5e7ef645757a2216f92f787f1a6005f3150da1bb263bf348",
    ),
    (
        "pybkb/mixins/__init__.py",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        "pybkb/mixins/networkx.py",
        "eecc70d9f428b72f63b7e17fedad9c2ae0e355805d7371acdd0b74ed55ef5af6",
    ),
    (
        "pybkb/reason/__init__.py",
        "93a18e8da39c2aa37610b28f92b49159629c365683c1487b01ffe47a7c8de56a",
    ),
    ("pybkb/reason/bkb.py", "d8b6f06adde7f5d6b3f0cf8468c797df70f74df3c6fbfbfc909d3915b43f8043"),
    (
        "pybkb/utils/__init__.py",
        "ffa32d8581fbdcacdbb2cae8a4dc41e9753f911927ebad25d62ddfef342e97d5",
    ),
    (
        "pybkb/utils/probability.py",
        "95b85102ee67112e60f05f42b3365be3bd5620e7e33537f2f419636711764564",
    ),
    ("bkbs/aquatic_eco.bkb", "6099120d31e8770825ec5cece230e541e2e6e15277ae94e1db007896639eea10"),
    ("bkbs/fisherman.bkb", "32f0ac8239b9ca9eedf7bbff440bcbc95d10634f43dc1eb134f41cd071a93d1e"),
    ("bkbs/goldfish.bkb", "d4ccdb6babaee0774de8b3abc64bfd22cc3ac5bb264f92ac854f6bac1950f31a"),
    (
        "bkbs/illegal_dumping_ev.bkb",
        "b8364d45ee419c9e6aab2754e89d451aa02f775f5609b2e3fcff0347a58ead21",
    ),
    (
        "bkbs/illegal_fishing_ev.bkb",
        "5213214b778079f72cc855367b67368ad3934bb8dcc91829c347312347d36933",
    ),
    ("bkbs/pirate.bkb", "15855e0e7c69063c3c994ea16d302d961a9fd31d00c5f3139b5dc7fef39fe193"),
    ("bkbs/tsunami_ev.bkb", "986e2491132449e430d508cef29aed7cdd0bd62f12e217b02a280f3458c9aef2"),
    (
        "tests/test_bkb_lib/test_api_bkb1.bkb",
        "93e2869add04b742847128ec467f650d2ec95c58a8eed4c0d3dd6be0d5b09df4",
    ),
    (
        "tests/test_bkb_lib/test_json.bkb",
        "5c38d398c566e542886395cec6a261dea1876442585ce46366552019a003d05f",
    ),
    (
        "tests/test_bkb_lib/test_json.json",
        "8898cba0317b9c1082f9907fe61efb240ec732ce26bb6975cecc240fb7e29aca",
    ),
    (
        "tests/test_bkb_lib/pirate_scenario_fused.bkb",
        "2c4c79d241ab99076bea25117211caccb42840a18c6a646966d56109dc57a0fa",
    ),
)
"""Every pybkb file the export reads, with its sha256 at `COMMIT`."""

GLOBALS = frozenset(
    {
        ("builtins", "dict"),
        ("copyreg", "_reconstructor"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("scipy.sparse._arrays", "dok_array"),
    }
)
"""The globals pybkb's current-format pickles name."""

INPUTS = {
    "aquatic_eco": "bkbs/aquatic_eco.bkb",
    "fisherman": "bkbs/fisherman.bkb",
    "goldfish": "bkbs/goldfish.bkb",
    "illegal_dumping_ev": "bkbs/illegal_dumping_ev.bkb",
    "illegal_fishing_ev": "bkbs/illegal_fishing_ev.bkb",
    "pirate": "bkbs/pirate.bkb",
    "pirate_scenario_fused": "tests/test_bkb_lib/pirate_scenario_fused.bkb",
    "test_api_bkb1": "tests/test_bkb_lib/test_api_bkb1.bkb",
    "test_json": "tests/test_bkb_lib/test_json.bkb",
    "tsunami_ev": "bkbs/tsunami_ev.bkb",
}
"""Each exported example, by fixture name."""

PIRATE = ("fisherman", "illegal_dumping_ev", "illegal_fishing_ev", "pirate", "tsunami_ev")
"""pybkb's `test_pirate_fusion` fragments, in its order."""

RELIABILITIES = (0.5, 0.8, 0.2, 0.3, 0.9)
"""Non-uniform reliabilities for the pirate fragments, which pybkb's tests never use."""

SAMPLED = {"pirate_fused_r1": 48, "pirate_fused_mixed": 16, "pirate_fused_mixed_collapsed": 16}
"""How many sampled complete-evidence queries each fused BKB gets."""

NON_MUTEX = (
    ((1, True, 0.3, []), (1, True, 0.4, [])),
    ((1, True, 0.3, [(2, True)]), (1, True, 0.4, [])),
    ((1, True, 0.3, [(2, True)]), (1, True, 0.4, [(3, False)])),
)
"""pybkb's `test_is_mutex` cases that must raise `BKBNotMutexError`."""

DISTRIBUTIONS = (
    "compress-pickle",
    "ipycytoscape",
    "llvmlite",
    "lz4",
    "networkx",
    "numba",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "tqdm",
)

NOTICE = (
    "Converted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed"
    " under the GNU General Public License, version 3, at the commit this manifest names."
    " Released under the GNU General Public License, version 3 only."
)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.exit("usage: export.py PYBKB_ROOT OUT_DIR")
    root, out = Path(argv[1]), Path(argv[2])
    seed = os.environ.get("PYTHONHASHSEED", "")
    if not seed.isdigit():
        sys.exit("set PYTHONHASHSEED to an integer before starting the interpreter")
    _verify(root)
    sys.path.insert(0, str(root))
    pybkb = {
        name: importlib.import_module(f"pybkb.{name}")
        for name in ("bkb", "exceptions", "fusion", "reason.bkb")
    }
    loaded = {name: pybkb["bkb"].BKB.load(str(root / path)) for name, path in INPUTS.items()}
    checks = _replay(pybkb, root, loaded)
    bkbs = {**loaded, **_fuse(pybkb["fusion"], loaded)}
    out.mkdir(parents=True)
    outputs = {
        f"{name}.json": _write(out / f"{name}.json", _document(bkb)) for name, bkb in bkbs.items()
    }
    outputs["queries.json"] = _write(out / "queries.json", _queries(pybkb["reason.bkb"], bkbs))
    _write(
        out / "manifest.json",
        {
            "checks": checks,
            "commit": COMMIT,
            "exporter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": dict(FILES),
            "mutex": {name: _is_mutex(pybkb["exceptions"], bkb) for name, bkb in bkbs.items()},
            "notice": NOTICE,
            "outputs": outputs,
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in DISTRIBUTIONS},
        },
    )
    _write(out / "run.json", {"hash_seed": seed})
    return 0


def _verify(root: Path) -> None:
    lz4_frame = importlib.import_module("lz4.frame")
    for path, digest in FILES:
        data = (root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            sys.exit(f"{path} does not match its sha256 at {COMMIT}")
        if path.endswith(".bkb"):
            unexpected = _globals(lz4_frame.decompress(data)) - GLOBALS
            if unexpected:
                sys.exit(f"{path} names globals outside the allow-list: {sorted(unexpected)}")


def _globals(raw: bytes) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    strings: list[str] = []
    memo: dict[int, object] = {}
    last: object = None
    for opcode, arg, _ in pickletools.genops(raw):
        name = opcode.name
        if name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}:
            last = str(arg)
            strings.append(last)
        elif name == "MEMOIZE":
            memo[len(memo)] = last
        elif name in {"PUT", "BINPUT", "LONG_BINPUT"}:
            memo[_memo_index(arg)] = last
        elif name in {"GET", "BINGET", "LONG_BINGET"}:
            last = memo.get(_memo_index(arg))
            if isinstance(last, str):
                strings.append(last)
        elif name == "STACK_GLOBAL":
            found.add((strings[-2], strings[-1]))
            last = None
        elif name in {"GLOBAL", "INST"}:
            module, attribute = str(arg).split(" ", 1)
            found.add((module, attribute))
            last = None
        elif name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}:
            found.add(("refused", name))
        else:
            last = None
    return found


def _memo_index(arg: object) -> int:
    if not isinstance(arg, int):
        sys.exit(f"a memo opcode carries {arg!r}, not an index")
    return arg


def _replay(pybkb: dict[str, Any], root: Path, loaded: dict[str, Any]) -> dict[str, bool]:
    bkb_class, fusion, errors = pybkb["bkb"].BKB, pybkb["fusion"], pybkb["exceptions"]
    built = bkb_class()
    for component, state in [("A", 1), ("A", "2"), (1, 1), (1, "B")]:
        built.add_inode(component, state)
    built.add_snode("A", 1, 0.003)
    built.add_snode(1, 1, 0.9, [("A", 1)])
    built.add_snode(1, "B", 0.56)
    built.add_snode("A", "2", 0.97, [(1, 1)])
    saved = loaded["test_api_bkb1"]
    reloaded = bkb_class.load(str(root / INPUTS["test_api_bkb1"]))
    expected = json.loads((root / "tests/test_bkb_lib/test_json.json").read_text(encoding="utf-8"))
    aquatic = loaded["aquatic_eco"]
    normal = [("pH_Value", "Normal")]
    counts = [
        len(aquatic.find_snodes("ReproduceAbility", "Low")),
        len(aquatic.find_snodes("ReproduceAbility", "Low", tail_subset=normal)),
        len(aquatic.find_snodes("ReproduceAbility", "Low", prob=0.1, tail_subset=normal)),
    ]
    pirate = fusion.fuse([loaded[name] for name in PIRATE], [1] * 5, collapse=False)
    stored = loaded["pirate_scenario_fused"]
    goldfish, sources = [loaded["goldfish"]] * 10, [str(index) for index in range(10)]
    return {
        "find_snodes_counts": counts == [2, 1, 1],
        "goldfish_collapsed_is_mutex": _is_mutex(
            errors, fusion.fuse(goldfish, [1] * 10, source_names=sources, collapse=True)
        ),
        "goldfish_is_mutex": _is_mutex(
            errors, fusion.fuse(goldfish, [1] * 10, source_names=sources, collapse=False)
        ),
        "json_matches_to_dict": expected == loaded["test_json"].to_dict(),
        "non_mutex_cases_are_refused": not any(
            _is_mutex(errors, _non_mutex(bkb_class, case)) for case in NON_MUTEX
        ),
        "pirate_fresh_equals_stored": bool(pirate == stored),
        "pirate_is_mutex": _is_mutex(errors, pirate),
        "pirate_stored_equals_fresh": bool(stored == pirate),
        "saved_equals_reload": bool(saved == reloaded),
        "saved_is_mutex": _is_mutex(errors, saved),
        "simple_build_equals_saved": bool(built == saved),
    }


def _non_mutex(bkb_class: Any, case: tuple[Any, ...]) -> Any:
    bkb = bkb_class()
    for component in (1, 2, 3):
        for state in (True, False):
            bkb.add_inode(component, state)
    for component, state, prob, tail in case:
        bkb.add_snode(component, state, prob, tail)
    return bkb


def _is_mutex(errors: Any, bkb: Any) -> bool:
    try:
        return bool(bkb.is_mutex())
    except errors.BKBNotMutexError:
        return False


def _fuse(fusion: Any, loaded: dict[str, Any]) -> dict[str, Any]:
    pirate, mixed = [loaded[name] for name in PIRATE], list(RELIABILITIES)
    goldfish, sources = [loaded["goldfish"]] * 10, [str(index) for index in range(10)]
    return {
        "goldfish_fused_collapsed": fusion.fuse(
            goldfish, [1] * 10, source_names=sources, collapse=True
        ),
        "pirate_fused_mixed": fusion.fuse(pirate, mixed, collapse=False),
        "pirate_fused_mixed_collapsed": fusion.fuse(pirate, mixed, collapse=True),
        "pirate_fused_r1": fusion.fuse(pirate, [1] * 5, collapse=False),
    }


def _name(value: Any) -> str:
    if isinstance(value, str | int):
        return str(value)
    sys.exit(f"cannot turn the name {value!r} into a string")


def _state(value: Any) -> str | frozenset[str]:
    if isinstance(value, frozenset):
        return frozenset(_name(member) for member in value)
    return _name(value)


def _inode_key(inode: tuple[str, str | frozenset[str]]) -> tuple[str, int, str, tuple[str, ...]]:
    component, state = inode
    if isinstance(state, str):
        return (component, 0, state, ())
    return (component, 1, "", tuple(sorted(state)))


def _document(bkb: Any) -> dict[str, Any]:
    originals = [tuple(inode) for inode in bkb.inodes]
    converted = [(_name(component), _state(state)) for component, state in originals]
    if len(set(converted)) != len(converted):
        sys.exit(f"the names of {bkb.name!r} collide once turned into strings")
    order = sorted(converted, key=_inode_key)
    position = {inode: index for index, inode in enumerate(order)}
    lookup = dict(zip(originals, converted, strict=True))
    snodes = []
    for index, prob in enumerate(bkb.snode_probs):
        head = lookup[tuple(bkb.get_snode_head(index))]
        tail = sorted(position[lookup[tuple(member)]] for member in bkb.get_snode_tail(index))
        snodes.append({"head": position[head], "prob": float(prob), "tail": tail})
    return {
        "description": bkb.description,
        "inodes": [[c, s if isinstance(s, str) else sorted(s)] for c, s in order],
        "name": _name(bkb.name),
        "snodes": snodes,
    }


def _components(bkb: Any) -> list[Any]:
    return sorted((c for c in bkb.inodes_map if "__Source__" not in str(c)), key=str)


def _states(bkb: Any, component: Any) -> list[Any]:
    return sorted(bkb.inodes_map[component], key=str)


def _exhaustive(bkb: Any) -> Iterator[tuple[Any, dict[Any, Any]]]:
    components = _components(bkb)
    for target in components:
        others = [c for c in components if c != target]
        for states in itertools.product(*(_states(bkb, c) for c in others)):
            yield target, dict(zip(others, states, strict=True))


def _sampled(bkb: Any, count: int) -> Iterator[tuple[Any, dict[Any, Any]]]:
    components = _components(bkb)
    for index in range(count):
        target = components[index % len(components)]
        yield target, {c: _pick(index, c, _states(bkb, c)) for c in components if c != target}


def _pick(index: int, component: Any, states: list[Any]) -> Any:
    digest = hashlib.sha256(f"{index}/{component}".encode()).hexdigest()
    return states[int(digest, 16) % len(states)]


def _queries(reason: Any, bkbs: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = [("aquatic_eco", _exhaustive(bkbs["aquatic_eco"]))]
    pairs += [(name, _sampled(bkbs[name], count)) for name, count in SAMPLED.items()]
    return [
        _query(reason, name, bkbs[name], target, evidence)
        for name, chosen in pairs
        for target, evidence in chosen
    ]


def _query(reason: Any, name: str, bkb: Any, target: Any, evidence: dict[Any, Any]) -> Any:
    updates = reason.BKBReasoner(bkb=bkb).heuristic_update_complete_fused(target, dict(evidence))
    return {
        "bkb": name,
        "evidence": {_name(c): _name(s) for c, s in evidence.items()},
        "result": {_name(state): _value(value) for (_, state), value in updates.items()},
        "target": _name(target),
    }


def _value(value: Any) -> float | str | None:
    if value is None:
        return None
    number = float(value)
    return "-inf" if number == -math.inf else number


def _write(path: Path, payload: Any) -> str:
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
