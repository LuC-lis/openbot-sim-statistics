"""Permutation tests.

The evaluation of a learned control policy raises hypotheses that are *not*
about a mean: "does this policy's command depend on its camera input at all?"
and "are these two policies distinguishable on the same conditions?".  A
permutation test is used because it makes the exchangeability assumption
explicit, needs no distributional assumption, and — with a fixed seed — is
exactly reproducible.

Two entry points:

``permutation_test``
    Generic.  Give it two samples (unpaired) or one paired sample and
    ``paired=True``; it recomputes the statistic under every relabelling of the
    data implied by the null hypothesis.

``observation_channel_test``
    The input-blindness null hypothesis: replacing the camera image with noise
    leaves the commanded turn unchanged.  Implemented as a paired permutation
    test on the per-condition turn span, where the pairing is the condition.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = ["permutation_test", "observation_channel_test"]


def permutation_test(
    a: Sequence[float],
    b: Sequence[float] | None = None,
    statistic: Callable[[np.ndarray, np.ndarray | None], float] | None = None,
    n_permutations: int = 2000,
    paired: bool = False,
    seed: int = 0,
) -> dict:
    """Permutation test for a two-sample (or paired) statistic.

    Parameters
    ----------
    a, b
        Samples.  If ``b`` is ``None`` a one-sample test against zero is run.
    statistic
        ``f(a, b) -> float``.  Defaults to the difference of means (unpaired)
        or the mean paired difference (paired).  A signed statistic is assumed:
        the p-value is two-sided, ``2 * min(P(T* >= t), P(T* <= t))`` capped at 1.
    paired
        If ``True``, permute the *sign* of each paired difference instead of
        relabelling between groups.  This is the correct null when the two
        measurements of a pair belong to the same condition.
    seed
        Seed for the permutation RNG.

    Returns a dict with the observed statistic, the p-value, the number of
    permutations and the null distribution's standard deviation.
    """
    a_arr = np.asarray(a, dtype=float).ravel()
    if b is None:
        b_arr = None
        if statistic is None:
            statistic = lambda x, _y: float(np.mean(x))  # noqa: E731
    else:
        b_arr = np.asarray(b, dtype=float).ravel()
        if statistic is None:
            if paired:
                statistic = lambda x, y: float(np.mean(x - y))  # noqa: E731
            else:
                statistic = lambda x, y: float(np.mean(x) - np.mean(y))  # noqa: E731

    observed = float(statistic(a_arr, b_arr))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)

    if b_arr is None:
        for i in range(n_permutations):
            signs = rng.choice([-1.0, 1.0], size=a_arr.size)
            null[i] = statistic(a_arr * signs, None)
    elif paired:
        if a_arr.shape != b_arr.shape:
            raise ValueError("paired samples must have the same length")
        d = a_arr - b_arr
        zeros = np.zeros_like(d)
        for i in range(n_permutations):
            signs = rng.choice([-1.0, 1.0], size=d.size)
            # sign flips are the exchangeable operation under H0; the statistic
            # is evaluated through the same call signature as the observed one.
            null[i] = statistic(d * signs, zeros)
    else:
        pooled = np.concatenate([a_arr, b_arr])
        n_a = a_arr.size
        for i in range(n_permutations):
            perm = rng.permutation(pooled)
            null[i] = statistic(perm[:n_a], perm[n_a:])

    p_hi = (np.sum(null >= observed) + 1.0) / (n_permutations + 1.0)
    p_lo = (np.sum(null <= observed) + 1.0) / (n_permutations + 1.0)
    return {
        "statistic": observed,
        "p_value": float(min(1.0, 2.0 * min(p_hi, p_lo))),
        "n_permutations": int(n_permutations),
        "null_sd": float(null.std(ddof=1)) if n_permutations > 1 else float("nan"),
        "null_mean": float(null.mean()),
    }


def observation_channel_test(
    span_real: Sequence[float],
    span_noise: Sequence[float],
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict:
    """Test the null hypothesis that the policy is input-blind.

    ``span_real`` and ``span_noise`` are the per-condition turn spans (the
    difference between the most-left and most-right command) measured with the
    real camera image and with the image replaced by noise, on *the same*
    conditions and seeds.

    Under the null the policy's output depends only on the command channel, so
    swapping the two images within a pair leaves the paired difference
    exchangeable.  A large observed difference with a small p-value is evidence
    that the image is being used.

    Returns the same structure as :func:`permutation_test` plus
    ``mean_difference`` and a standardised effect size (Cohen's dz).
    """
    a = np.asarray(span_real, dtype=float).ravel()
    b = np.asarray(span_noise, dtype=float).ravel()
    res = permutation_test(a, b, n_permutations=n_permutations, paired=True, seed=seed)
    d = a - b
    sd = float(d.std(ddof=1)) if d.size > 1 else float("nan")
    res["mean_difference"] = float(d.mean())
    res["effect_size_dz"] = float(d.mean() / sd) if sd and sd > 0 else float("nan")
    return res
