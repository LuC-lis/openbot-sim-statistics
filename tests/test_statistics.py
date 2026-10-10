"""Unit tests for the analysis layer (no pytest required).

    python -m tests.test_statistics
"""

import math

import numpy as np

from statistics.design import ConditionDesign, latin_hypercube
from statistics.intervals import bootstrap_ci, paired_bootstrap_diff, wilson_interval
from statistics.regression import logistic_regression
from statistics.tests import observation_channel_test, permutation_test


def test_latin_hypercube_is_stratified():
    n, d = 16, 5
    s = latin_hypercube(n, d, seed=1)
    assert s.shape == (n, d)
    assert np.all((s >= 0) & (s < 1))
    for j in range(d):
        strata = np.floor(s[:, j] * n).astype(int)
        assert sorted(strata.tolist()) == list(range(n)), "each stratum used exactly once"


def test_wilson_endpoints():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.15 < hi < 0.17
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and 0.83 < lo < 0.85
    lo, hi = wilson_interval(10, 20)
    assert lo < 0.5 < hi
    # narrower than the Wald interval at the boundary and always inside [0, 1]
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_covers_the_truth():
    # Wald would fail here; Wilson should not be systematically short.
    p, n, trials = 0.5, 40, 400
    rng = np.random.default_rng(0)
    covered = 0
    for _ in range(trials):
        k = int(rng.binomial(n, p))
        lo, hi = wilson_interval(k, n)
        covered += lo <= p <= hi
    assert covered / trials > 0.90, covered / trials


def test_bootstrap_ci_of_a_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(3.0, 1.0, size=500)
    lo, hi = bootstrap_ci(x, np.mean, n_resamples=500, seed=0)
    assert lo < 3.0 < hi
    assert hi - lo < 0.5


def test_paired_bootstrap_removes_between_condition_variance():
    # Same difference everywhere, but huge condition-to-condition spread:
    # the paired interval must be far tighter than the unpaired one.
    base = np.arange(50, dtype=float) * 10.0
    a = base + 1.0
    b = base
    _, lo_p, hi_p = paired_bootstrap_diff(a, b, n_resamples=400, seed=0)
    assert abs(hi_p - lo_p) < 0.5
    lo_u, hi_u = bootstrap_ci(a, np.mean, n_resamples=400, seed=0)
    assert (hi_u - lo_u) > 5.0


def test_permutation_test_null_is_calibrated():
    rng = np.random.default_rng(3)
    pvals = []
    for i in range(200):
        a = rng.normal(0, 1, 12)
        b = rng.normal(0, 1, 12)
        pvals.append(permutation_test(a, b, n_permutations=200, seed=i)["p_value"])
    # under the null a 5% test should reject ~5% of the time
    assert 0.01 < np.mean(np.array(pvals) < 0.05) < 0.15


def test_permutation_test_detects_a_real_shift():
    rng = np.random.default_rng(4)
    a = rng.normal(1.0, 1.0, 20)
    b = rng.normal(0.0, 1.0, 20)
    res = permutation_test(a, b, n_permutations=2000, seed=0)
    assert res["statistic"] > 0.5
    assert res["p_value"] < 0.05


def test_observation_channel_test():
    rng = np.random.default_rng(5)
    # input-blind policy: real and noise images give the same span
    blind = rng.normal(0.0, 0.05, 12)
    res = observation_channel_test(blind, rng.normal(0.0, 0.05, 12), n_permutations=1000, seed=0)
    assert res["p_value"] > 0.05

    # vision-driven policy: noise removes the span
    real = rng.normal(-0.47, 0.03, 12)
    noise = rng.normal(0.0, 0.03, 12)
    res = observation_channel_test(real, noise, n_permutations=1000, seed=0)
    assert res["p_value"] < 0.01
    assert res["effect_size_dz"] < -5.0


def test_logistic_regression_recovers_a_known_coefficient():
    rng = np.random.default_rng(6)
    n = 4000
    x = rng.normal(0.0, 1.0, n)
    beta0, beta1 = -0.4, 1.2
    p = 1.0 / (1.0 + np.exp(-(beta0 + beta1 * x)))
    y = (rng.random(n) < p).astype(float)
    fit = logistic_regression(x[:, None], y, names=["x"], standardise=False)
    assert abs(fit.coef[0] - beta0) < 0.15
    assert abs(fit.coef[1] - beta1) < 0.15
    assert fit.p[1] < 1e-6
    assert fit.odds_ratio[1] > 1.0


def test_logistic_regression_flags_separation():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    fit = logistic_regression(x[:, None], y, standardise=False)
    assert fit.separated or abs(fit.coef[1]) > 3.0


def test_design_pairs_seeds_across_policies():
    d = ConditionDesign(n_conditions=8, replicates=2, seed=11)
    conditions = d.build()
    assert len(conditions) == 16
    # every condition carries an explicit start and its own appearance seed
    assert all(c.appearance_seed > 0 for c in conditions)
    assert len({c.appearance_seed for c in conditions}) == len(conditions)
    assert len({c.seed for c in conditions}) == len(conditions)
    # obstacles are real rectangles on the road
    for c in conditions:
        (x0, x1, y0, y1), = c.obstacles
        assert x0 < x1 and y0 < y1
        assert abs(y0) < 1.0 and abs(y1) < 1.0
    # determinism
    assert [c.to_row() for c in ConditionDesign(n_conditions=8, replicates=2, seed=11).build()] == \
           [c.to_row() for c in conditions]


def main() -> None:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as exc:  # pragma: no cover
            failures.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # pragma: no cover
            failures.append((name, exc))
            print(f"  ERR  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
