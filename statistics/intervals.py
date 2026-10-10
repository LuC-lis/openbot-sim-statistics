"""Interval estimation for a proportion and for a paired difference.

Two estimators are used throughout the evaluation:

``wilson_interval``
    Score interval for a binomial proportion.  Preferred over the Wald
    ("normal approximation") interval because it stays inside [0, 1] and is
    still honest when a policy scores 0/20 or 20/20 — which is exactly the
    regime a closed-loop recovery experiment lives in.

``bootstrap_ci`` / ``paired_bootstrap_diff``
    Non-parametric percentile bootstrap.  Used for continuous outcomes such as
    tracking error, where no closed form is available, and for the difference
    between two policies evaluated on *the same* seeds (paired design), which
    removes the between-condition variance from the comparison.

Every function takes an explicit ``seed`` and is a pure function of its inputs,
so a reported interval is reproducible from the raw numbers alone.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

__all__ = ["wilson_interval", "bootstrap_ci", "paired_bootstrap_diff", "bootstrap_difference"]

_Z = 1.959963984540054  # 97.5th percentile of the standard normal


def _z_for(confidence: float) -> float:
    """Two-sided normal quantile for a confidence level (0.95 -> 1.959964)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if abs(confidence - 0.95) < 1e-12:
        return _Z
    # inverse erf via bisection on the normal CDF, accurate to ~1e-12
    target = 0.5 * (1.0 + confidence)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` trials.

    The interval is the set of null proportions ``p`` that a two-sided score
    test would not reject, i.e. it inverts the score statistic rather than
    approximating the sampling distribution of ``k/n`` by a normal.

    >>> wilson_interval(0, 20)[0]
    0.0
    >>> round(wilson_interval(0, 20)[1], 4)
    0.1611
    >>> round(wilson_interval(20, 20)[0], 4)
    0.8389
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    z = _z_for(confidence)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def bootstrap_ci(
    sample: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic`` applied to ``sample``.

    Resamples with replacement ``n_resamples`` times; the interval is the
    ``[alpha/2, 1 - alpha/2]`` quantiles of the resampled statistics.  Reported
    as ``(lo, hi, point)``-style values elsewhere; here the point estimate is
    returned by :func:`bootstrap_ci` callers via ``statistic(sample)``.
    """
    x = np.asarray(sample, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("sample is empty")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    alpha = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    stats = np.array([statistic(x[i]) for i in idx], dtype=float)
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap interval for ``mean(a - b)`` on paired observations.

    ``a`` and ``b`` must be aligned: entry ``i`` of each must come from the same
    condition and the same seed, otherwise the pairing — and with it the whole
    point of the design — is lost.  Returns ``(point, lo, hi)``.
    """
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired samples must have the same length")
    d = a_arr - b_arr
    lo, hi = bootstrap_ci(d, np.mean, n_resamples=n_resamples, confidence=confidence, seed=seed)
    return float(d.mean()), lo, hi


def bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap interval for ``mean(a) - mean(b)`` (unpaired)."""
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a_arr.size, size=(n_resamples, a_arr.size))
    ib = rng.integers(0, b_arr.size, size=(n_resamples, b_arr.size))
    stats = np.array([a_arr[i].mean() - b_arr[j].mean() for i, j in zip(ia, ib)])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(a_arr.mean() - b_arr.mean()), float(lo), float(hi)
