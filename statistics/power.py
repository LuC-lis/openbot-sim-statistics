"""Sample-size planning, so the design is chosen *before* the data is seen.

Two questions, two formulas:

``n_for_two_proportions``
    How many episodes per policy are needed to detect a difference between two
    survival rates with a two-sided test at level ``alpha`` and power
    ``1 - beta``?  Uses the standard pooled-variance normal approximation, with
    the arcsine-free form that behaves sensibly at the rates a closed-loop
    experiment actually produces (0.6 vs 0.4, not 0.001 vs 0.0009).

``n_for_paired_effect``
    The same for a paired comparison on a continuous outcome (tracking error),
    expressed as an effect size ``dz = mean difference / sd of the difference``.

``minimum_detectable_difference``
    The inverse question, which is the one to quote after the fact: given the
    number of episodes actually run, how large a difference could this design
    have detected?  A null result is only informative alongside this number.

All three are closed-form; no simulation and no SciPy.
"""

from __future__ import annotations

import math

__all__ = [
    "z_alpha_two_sided",
    "z_power",
    "n_for_two_proportions",
    "n_for_paired_effect",
    "minimum_detectable_difference",
    "minimum_detectable_effect",
]

_Z_95 = 1.959963984540054


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF by bisection (accurate to ~1e-12)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def z_alpha_two_sided(alpha: float = 0.05) -> float:
    """``z_{1 - alpha/2}`` (1.959964 at alpha = 0.05)."""
    return _normal_ppf(1.0 - alpha / 2.0)


def z_power(power: float = 0.8) -> float:
    """``z_{1 - beta}`` (0.841621 at power = 0.8)."""
    return _normal_ppf(power)


def n_for_two_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
) -> int:
    """Episodes needed **per group** to detect ``p1`` vs ``p2``.

    Pooled-variance normal approximation with an unpooled alternative term:

        n = ( z_{1-a/2} * sqrt(2 p̄ (1-p̄)) + z_{1-b} * sqrt(p1(1-p1) + p2(1-p2)) )^2
            / (p1 - p2)^2

    The result is rounded up.  ``allocation`` is the ratio n2/n1 (1 = equal
    groups); for unequal allocation the returned value is n1.
    """
    if not 0.0 < p1 < 1.0 or not 0.0 < p2 < 1.0:
        raise ValueError("p1 and p2 must be strictly inside (0, 1)")
    if p1 == p2:
        raise ValueError("there is no difference to detect")
    p_bar = 0.5 * (p1 + p2)
    za, zb = z_alpha_two_sided(alpha), z_power(power)
    numerator = za * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + zb * math.sqrt(
        p1 * (1.0 - p1) + p2 * (1.0 - p2) / allocation
    )
    return int(math.ceil(numerator ** 2 / (p1 - p2) ** 2))


def n_for_paired_effect(dz: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Pairs needed to detect a standardised paired difference ``dz``."""
    if dz <= 0:
        raise ValueError("dz must be positive")
    za, zb = z_alpha_two_sided(alpha), z_power(power)
    return int(math.ceil(((za + zb) / dz) ** 2))


def minimum_detectable_difference(
    n: int,
    p_reference: float,
    alpha: float = 0.05,
    power: float = 0.8,
    lo: float = 1e-4,
    hi: float | None = None,
) -> float:
    """Smallest detectable absolute difference in proportions at sample size ``n``.

    Solves :func:`n_for_two_proportions` for ``|p1 - p2|`` by bisection, keeping
    the reference rate fixed.  This is the number that belongs next to a null
    result: *"with 120 episodes per policy this design could have detected a
    difference of about X percentage points; smaller differences remain open."*
    """
    if not 0 < p_reference < 1:
        raise ValueError("p_reference must be inside (0, 1)")
    hi = hi if hi is not None else min(p_reference, 1.0 - p_reference)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p2 = p_reference + mid if p_reference + mid < 1.0 else p_reference - mid
        try:
            need = n_for_two_proportions(p_reference, p2, alpha=alpha, power=power)
        except ValueError:
            need = n + 1
        if need <= n:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def minimum_detectable_effect(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest detectable ``dz`` for a paired design with ``n`` pairs."""
    if n < 1:
        raise ValueError("n must be positive")
    za, zb = z_alpha_two_sided(alpha), z_power(power)
    return (za + zb) / math.sqrt(n)
