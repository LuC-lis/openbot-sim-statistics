"""The condition design: what gets varied, and how the seeds are paired.

An evaluation that draws its conditions from whatever the environment happens
to sample gives you a success rate for an unspecified mixture of conditions.
This module replaces that with a **designed experiment**:

* the condition space is sampled by **Latin hypercube** — every factor is split
  into ``n`` strata and each stratum is used exactly once, so a small number of
  conditions still covers the range of every factor (better than independent
  uniform draws, which clump);
* the same condition table is replayed for **every** policy, with **identical
  seeds** — so the comparison between two policies is *paired* and the
  between-condition variance cancels;
* ``replicates`` independent tables are generated (different master seeds), so
  the whole comparison can be repeated and the variation across replicates
  reported.

Factors
-------
===================  ==========================================================
``lateral0``         initial lateral offset from the centre line (m)
``heading0``         initial heading error (rad)
``cmd``              commanded turn {-1, 0, +1}
``obstacle_offset``  lateral centre of the obstacle block on the road (m)
``appearance_seed``  seed of the floor texture / illumination / sensor noise
``perturb_rate``     per-step probability of a state disturbance
===================  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from openbot_sim.appearance import AppearanceSpec, sample_appearance

__all__ = ["latin_hypercube", "Condition", "ConditionDesign"]

#: obstacle block geometry (metres): x span, and lateral half-width
OBSTACLE_X = (2.5, 3.0)
OBSTACLE_HALF_WIDTH = 0.18


def latin_hypercube(n: int, d: int, seed: int = 0) -> np.ndarray:
    """``(n, d)`` Latin-hypercube sample of the unit cube.

    Each of the ``d`` columns is the ``n`` strata ``(i/n, (i+1)/n)`` in a random
    permutation, with the point placed uniformly inside its stratum.  The result
    is a *balanced* sample: every column has exactly one point per stratum.

    >>> import numpy as np
    >>> s = latin_hypercube(4, 2, seed=0)
    >>> s.shape
    (4, 2)
    >>> sorted(np.floor(s[:, 0] * 4).astype(int).tolist())
    [0, 1, 2, 3]
    """
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    rng = np.random.default_rng(seed)
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        cut[:, j] = cut[rng.permutation(n), j]
    return cut


@dataclass(frozen=True)
class Condition:
    """One fully specified evaluation condition (a row of the design table)."""

    condition_id: int
    replicate: int
    seed: int
    lateral0: float
    heading0: float
    cmd: int
    road_half_width: float
    obstacle_offset: float
    appearance_seed: int
    perturb_rate: float

    @property
    def obstacles(self) -> list[tuple[float, float, float, float]]:
        y0 = self.obstacle_offset - OBSTACLE_HALF_WIDTH
        y1 = self.obstacle_offset + OBSTACLE_HALF_WIDTH
        return [(OBSTACLE_X[0], OBSTACLE_X[1], y0, y1)]

    @property
    def appearance(self) -> AppearanceSpec:
        return sample_appearance(self.appearance_seed)

    def to_row(self) -> dict:
        return asdict(self)


class ConditionDesign:
    """A Latin-hypercube condition table, replayable across policies.

    Parameters
    ----------
    n_conditions
        Number of distinct conditions per replicate.
    replicates
        Number of independent tables (master seeds).
    seed
        Master seed; replicate ``r`` uses ``seed + r``.
    road_half_width
        Road half width used for every condition (kept fixed so that the
        comparison is not confounded by a change of geometry).
    """

    FACTORS = (
        "lateral0",
        "heading0",
        "cmd",
        "obstacle_offset",
        "appearance_seed",
        "perturb_rate",
    )

    def __init__(
        self,
        n_conditions: int = 24,
        replicates: int = 2,
        seed: int = 20261009,
        road_half_width: float = 0.9,
        lateral_fraction: float = 0.85,
        heading_range: float = 0.35,
        obstacle_span: float = 0.55,
        perturb_max: float = 0.02,
    ):
        self.n_conditions = int(n_conditions)
        self.replicates = int(replicates)
        self.seed = int(seed)
        self.road_half_width = float(road_half_width)
        self.lateral_fraction = float(lateral_fraction)
        self.heading_range = float(heading_range)
        self.obstacle_span = float(obstacle_span)
        self.perturb_max = float(perturb_max)

    # ------------------------------------------------------------------
    def build(self) -> list[Condition]:
        conditions: list[Condition] = []
        half = self.road_half_width
        for rep in range(self.replicates):
            master = self.seed + rep
            unit = latin_hypercube(self.n_conditions, len(self.FACTORS), seed=master)
            for i in range(self.n_conditions):
                u = unit[i]
                cmds = (-1, 0, 1)
                conditions.append(
                    Condition(
                        condition_id=i,
                        replicate=rep,
                        # episode seed is a deterministic function of the
                        # condition, so policy A and policy B replay the *same*
                        # disturbance sequence and the same command schedule.
                        seed=int(master * 100003 + i * 7919),
                        lateral0=float((2.0 * u[0] - 1.0) * half * self.lateral_fraction),
                        heading0=float((2.0 * u[1] - 1.0) * self.heading_range),
                        cmd=int(cmds[min(2, int(u[2] * 3))]),
                        road_half_width=half,
                        obstacle_offset=float((2.0 * u[3] - 1.0) * self.obstacle_span),
                        appearance_seed=int(master * 7919 + i * 104729 + 1),
                        perturb_rate=float(u[5] * self.perturb_max),
                    )
                )
        return conditions

    def to_rows(self) -> list[dict]:
        return [c.to_row() for c in self.build()]
