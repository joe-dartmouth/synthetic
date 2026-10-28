"""A paid `score` run's world for `python -m bko.evaluate` tests: data, releases, scripted replies.

Every call goes to a `ScriptedSend`; no test reaches the API.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Final

from bko.evaluate.__main__ import Context
from bko.generate.messages import Reply, Send
from tests._datasets_fixtures import pin
from tests._evaluate_fixtures import FAST_ARMS, SMALL
from tests._fusion_fixtures import young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._generate_fixtures import SIZES, ScriptedSend, reply, uci
from tests._ontology_fixtures import RoutingOpener
from tests._validate_fixtures import PLANTED, planted


NARROW: Final = dataclasses.replace(
    SMALL, arms=("none", "bko"), candidate="bko", comparators=("none",), replica_arms=()
)
"""E1 on two holdouts with one baseline and the candidate."""

UCI_E1: Final = uci({(40, 0): "4.0"}, "1.0", {**SIZES, (40, 0, 1): 5})
"""`UCI_X` with five young women with the condition, so every holdout tests one of them.

Each holdout's gap request is then 12 rows of `("<55", "female", 0)` and 16 of
`("<55", "female", 1)`, from 8 and 4 training records."""

PIN: Final = pin(UCI_E1)


def scripted() -> ScriptedSend:
    """Both holdouts' replies: each fills its two cells in one call apiece."""
    holdout = (reply([young(0)] * 12), reply([young(1)] * 16))
    return ScriptedSend(*holdout, *holdout)


def context(send: Send, *, arms=None) -> tuple[Context, RoutingOpener]:
    """A context over `UCI_E1` and the ladder releases, with `send` for every call."""
    pins, served = ladder_releases()
    opener = RoutingOpener({**served, PIN.url: UCI_E1})
    built = Context(
        open_url=opener,
        uci=PIN,
        experiment=NARROW,
        arms=arms or {"none": FAST_ARMS["none"]},
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        send=lambda: send,
    )
    return built, opener


THREE: Final = dataclasses.replace(NARROW, arms=("none", "bko", "bko-population", "bko-frequency"))
"""`NARROW` with all three BKO arms on one call log."""

BAD: Final = young(1, chol=234)
"""A candidate that passes `bko.validate` and is dropped for range: no training record has it."""


def holdout_replies(arm: str) -> tuple[Reply, Reply]:
    """One holdout's two calls for `arm`, asking 12 rows of target 0 then 16 of target 1.

    Each arm's replies differ, so each arm's counts show whose calls they came from:
    `bko-population` gets one record dropped for range per holdout, `bko-frequency` two past the
    quota.
    """
    well: list[object] = [young(0)] * 12
    sick: list[object] = [young(1)] * 16
    if arm == "bko-population":
        sick = [*sick, BAD]
    if arm == "bko-frequency":
        well = [*well, young(0), young(0)]
    return reply(well), reply(sick)


def three_arm_script() -> ScriptedSend:
    """Both holdouts' calls in the order the harness makes them: holdout, then arm, then cell."""
    arms = THREE.arms[1:]
    return ScriptedSend(
        *(r for _ in THREE.holdout_seeds for arm in arms for r in holdout_replies(arm))
    )


REPLICAS: Final = dataclasses.replace(THREE, replica_arms=THREE.arms[1:])
"""`THREE` with every BKO arm also filling a replica, as E1 asks of every row-adding arm."""

REPLICA_QUOTA: Final = (
    (40, 0, 0, 8),
    (40, 0, 1, 4),
    (40, 1, 0, 20),
    (40, 1, 1, 20),
    (63, 0, 0, 20),
    (63, 0, 1, 20),
    (63, 1, 0, 20),
    (63, 1, 1, 20),
)
"""`UCI_E1`'s training cells in grid order, as age, sex, target and records: one in five of
`SIZES` held out. Each is one replica call of that many rows."""


def replica_replies(arm: str) -> tuple[Reply, ...]:
    """One holdout's eight replica calls for `arm`, with cp 2 so no record copies a test record.

    Each arm's replica replies differ from its gap replies, so a count under the wrong source
    shows: `bko` gets one candidate that fails validation, `bko-population` three dropped for
    range, and `bko-frequency` one past the quota and one copy of a test record.
    """
    calls: list[list[object]] = []
    for age, sex, target, rows in REPLICA_QUOTA:
        call: list[object] = [planted(age=age, sex=sex, target=target, cp=2)] * rows
        calls.append(call)
    if arm == "bko":
        calls[0] = [*calls[0], PLANTED["code:cp"]]
    if arm == "bko-population":
        calls[0] = [*calls[0], BAD, BAD, BAD]
    if arm == "bko-frequency":
        calls[0] = [*calls[0], calls[0][0]]
        calls[6] = [planted(age=63, sex=1, target=0), *calls[6][1:]]
    return tuple(reply(call) for call in calls)


def replica_script() -> ScriptedSend:
    """Both holdouts' calls in harness order: each holdout's gap fills, then its replica fills."""
    arms = REPLICAS.arms[1:]
    return ScriptedSend(
        *(
            r
            for _ in REPLICAS.holdout_seeds
            for r in (
                *(g for arm in arms for g in holdout_replies(arm)),
                *(p for arm in REPLICAS.replica_arms for p in replica_replies(arm)),
            )
        )
    )


def candidates_for(log: Path) -> Path:
    """The candidates file a test run beside the call log `log` writes."""
    return log.with_name(f"{log.stem}.candidates.jsonl")


def score_argv(log: Path, tmp_path: Path, *flag: str) -> list[str]:
    """`score` with `flag`, the call log `log`, its candidates and a data root in `tmp_path`.

    The candidates file sits beside the log, as `candidates_for` names it.
    """
    candidates = ["--candidates", str(candidates_for(log))]
    return ["score", *flag, "--log", str(log), *candidates, "--data-dir", str(tmp_path / "data")]
