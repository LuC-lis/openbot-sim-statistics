"""Run policies over a design and log one tidy row per episode.

The output is deliberately boring: a CSV with one row per (policy, condition,
replicate) and columns for the binary outcome, the continuous outcome and the
termination reason.  Everything downstream — Wilson intervals, bootstrap
intervals, permutation tests, the logistic regression — reads that CSV, so any
number in a report can be traced back to a logged episode.

Definition of the binary outcome
--------------------------------
``success = 1`` iff the episode ended by **truncation** (the vehicle stayed on
the road and clear of obstacles for the full horizon) and ``0`` otherwise.
Collisions, off-road exits and spin-outs are all failures; ``reason`` keeps them
distinguishable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from openbot_sim.env import OpenBotEnv
from openbot_sim.policies import RandomPolicy, TeacherPolicy

from .design import Condition, ConditionDesign

__all__ = ["FIELDS", "run_episode", "run_design", "baseline_policies", "write_csv", "read_csv"]

#: column order of the outcomes CSV
FIELDS = [
    "policy",
    "replicate",
    "condition_id",
    "seed",
    "cmd",
    "lateral0",
    "heading0",
    "obstacle_offset",
    "appearance_seed",
    "perturb_rate",
    "road_half_width",
    "success",
    "steps",
    "track_err",
    "return",
    "reason",
]


def run_episode(
    env: OpenBotEnv,
    policy,
    cond: Condition,
    max_steps: int = 400,
    disturb: float = 0.0,
) -> dict:
    """Play one episode under one condition and return a row of :data:`FIELDS`.

    ``disturb`` is *added* to the condition's own perturbation rate; pass 0 to
    use exactly what the design specifies.
    """
    obs, info = env.reset(
        seed=cond.seed,
        start=(cond.lateral0, cond.heading0),
        appearance=cond.appearance,
        obstacles=cond.obstacles,
    )
    env.cmd = float(cond.cmd)
    env.max_steps = int(max_steps)
    obs, info = env.observe()

    total_r = 0.0
    track: list[float] = []
    reason = "timeout"
    t = 0
    for t in range(max_steps):
        action = policy.act(obs, info)
        obs, rew, term, trunc, info = env.step(action)
        total_r += float(rew)
        track.append(abs(info["lateral"] - info["target_lateral"]))
        if disturb > 0 and env.rng.random() < disturb and not (term or trunc):
            env.perturb()
            obs, info = env.observe()
        if term or trunc:
            reason = info["components"]["reason"] if term else "timeout"
            break

    return {
        "replicate": cond.replicate,
        "condition_id": cond.condition_id,
        "seed": cond.seed,
        "cmd": cond.cmd,
        "lateral0": round(cond.lateral0, 6),
        "heading0": round(cond.heading0, 6),
        "obstacle_offset": round(cond.obstacle_offset, 6),
        "appearance_seed": cond.appearance_seed,
        "perturb_rate": round(cond.perturb_rate, 6),
        "road_half_width": cond.road_half_width,
        "success": int(reason == "timeout"),
        "steps": t + 1,
        "track_err": float(np.mean(track)) if track else float("nan"),
        "return": total_r,
        "reason": reason,
    }


def run_design(
    policies: dict[str, Callable[[np.random.Generator], object]],
    design: ConditionDesign | None = None,
    conditions: Sequence[Condition] | None = None,
    max_steps: int = 400,
    disturb: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Run every policy over every condition; return the tidy rows.

    All policies see the same conditions in the same order with the same seeds,
    which is what makes the downstream comparison paired.
    """
    if conditions is None:
        conditions = (design or ConditionDesign()).build()

    rows: list[dict] = []
    for name, factory in policies.items():
        env = OpenBotEnv(seed=seed, road_half_width=conditions[0].road_half_width)
        policy = factory(np.random.default_rng(seed))
        for cond in conditions:
            row = run_episode(env, policy, cond, max_steps=max_steps, disturb=disturb)
            row["policy"] = name
            rows.append(row)
        if verbose:
            n = sum(r["success"] for r in rows if r["policy"] == name)
            print(f"  {name:<22} {n}/{len(conditions)} episodes survived the horizon")
    return rows


def baseline_policies() -> dict[str, Callable[[np.random.Generator], object]]:
    """The two always-available baselines: a random walk and the privileged expert."""
    return {
        "random": lambda rng: RandomPolicy(rng),
        "expert": lambda _rng: TeacherPolicy(),
    }


def write_csv(rows: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    return path


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["success"] = int(row["success"])
        row["steps"] = int(row["steps"])
        for key in ("cmd", "replicate", "condition_id", "seed", "appearance_seed"):
            row[key] = int(row[key])
        for key in ("lateral0", "heading0", "obstacle_offset", "perturb_rate",
                    "road_half_width", "track_err", "return"):
            row[key] = float(row[key])
    return rows
