# Statistical protocol

This document describes how a number in a report produced by this repository was
obtained. Every step is reproducible from the logged episodes: the design table
is deterministic given a master seed, and every interval, test and coefficient
is a pure function of the outcomes CSV plus an explicit resampling seed.

Nothing here depends on SciPy, R or a notebook. `numpy` is the only requirement,
and the whole analysis layer is unit-tested (`python -m tests.test_statistics`).

---

## 1 · What is varied, and what is held fixed

**Fixed** across the whole evaluation: road geometry (straight lane,
`road_half_width = 0.9 m`), camera intrinsics, image resolution (96 × 256),
control horizon (400 steps), and the task definition. A comparison is only
interpretable if these do not move between the two policies being compared.

**Varied by design** (the six factors in `statistics/design.py`):

| factor | range | why it is in the design |
|---|---|---|
| initial lateral offset | ±0.85 × road half width | the recovery problem: how far off the centre line the vehicle starts |
| initial heading error | ±0.35 rad | coupled with the lateral offset; a spin-out is the other failure mode |
| commanded turn | −1 / 0 / +1 | the task is conditional; a policy can look fine at cmd = 0 and fail at cmd = ±1 |
| obstacle lateral offset | ±0.55 m | a blocked lane forces a decision that pure lane-keeping never has to make |
| appearance seed | — | floor texture, illumination, colour temperature, vignetting and sensor noise |
| disturbance rate | 0 – 0.02 per step | stochastic state perturbations, i.e. the "someone bumped the robot" case |

Sampling is **Latin hypercube**: each factor is split into `n` equal strata and
each stratum is used exactly once per replicate (see
`results/design_coverage.png`). With independent uniform draws, 24 conditions
leave visible gaps in a factor's range and clumps elsewhere; with a Latin
hypercube, every factor covers its range evenly, which is what makes a small
experiment informative.

**Replicates.** A design of `n` conditions is repeated with independent master
seeds. Conditions from different replicates are never mixed when computing a
paired quantity.

**Pairing.** The episode seed is a deterministic function of
`(replicate, condition_id)`, so policy *A* and policy *B* face the same starting
pose, the same obstacle, the same appearance seed and the same command schedule
— *the same scene twice*. Differences are therefore computed **within**
conditions.

---

## 2 · Outcomes

One row per episode, written to CSV (`statistics/outcomes.py`):

- `success` — 1 iff the episode reached the horizon (on-road and clear of
  obstacles at every step), else 0;
- `reason` — `timeout`, `off_road`, `turned` (spin-out) or `collision`, so the
  failure modes stay distinguishable rather than collapsing into "failed";
- `track_err` — mean |lateral − target| over the episode (m);
- `steps`, `return`, plus the full condition vector for the regression.

---

## 3 · Estimating a single policy's success rate

The headline number is a proportion — `k` survivors out of `n` episodes — and it
is reported as a **Wilson score interval** (`statistics/intervals.py`), not a
Wald interval.

Reason: the interesting policies sit at the edge of the outcome space. A policy
that survives 0 of 120 episodes has a Wald interval of width zero, which is
obviously wrong; the Wilson interval correctly reports "somewhere between 0 and
3 %". Both endpoints stay inside [0, 1] by construction, and the interval is
obtained by inverting a score test rather than by assuming the sample proportion
is normal, which it is not at these counts.

Continuous outcomes (tracking error) get a **percentile bootstrap** interval
with 2 000 resamples and a fixed seed.

---

## 4 · Comparing two policies

Two policies evaluated on the same conditions are analysed as a **paired**
design:

1. **Paired bootstrap** of the difference in means (`paired_bootstrap_diff`):
   resample conditions with replacement, recompute `mean(a − b)` within each
   resample, take the 2.5 % and 97.5 % quantiles. Resampling pairs — not
   independent samples — removes the condition-to-condition variance from the
   interval. In the units test this is a factor of > 10 in interval width: with
   a spread of ~10 m between conditions and a true difference of 1 m, the
   unpaired interval is several metres wide and the paired interval is under
   0.5 m.
2. **Paired permutation test** (`permutation_test(..., paired=True)`): under
   H₀ the two policies are exchangeable within a condition, so the sign of each
   paired difference is exchangeable; the p-value is the fraction of 2 000 sign
   flips whose mean difference is at least as extreme as the observed one
   (two-sided, with a `+1` correction so a p-value of exactly 0 is never
   reported).

The interval says how large the difference plausibly is; the permutation test
says whether the pairing leaves any room for it to be zero.

---

## 5 · Which conditions predict failure

A success rate is one number; the question worth asking is *what makes an
episode fail*. That is a **logistic regression** of the failure indicator
(`1 − success`) on the condition factors, fitted by IRLS (Newton–Raphson on the
log-likelihood) in `statistics/regression.py`.

Reported with each coefficient: the Wald standard error from the inverse
observed Fisher information, the normal `z`, a two-sided p-value, and the odds
ratio with a 95 % interval. Continuous predictors are standardised, so each
coefficient means "per one standard deviation of this factor" and magnitudes are
comparable across factors measured in different units.

Two guards, because a logistic fit fails loudly in practice in exactly two ways:

- **Separation.** If the outcome is perfectly predicted (for example by a policy
  indicator when the baseline never survives), the likelihood has no interior
  maximum and every standard error is meaningless. The fit detects this, flags
  it, and the reporting code fits one model *per policy* by default so that the
  question is asked where the outcome actually varies.
- **Collinearity.** The condition number of the information matrix is printed
  with every fit. A large value means the standard errors should not be trusted
  regardless of the p-values.

McFadden's pseudo-R² (1 − ℓ/ℓ₀) is reported as a descriptive measure; it is not
interpreted as a variance explained, and no p-value is attached to it.

---

## 6 · Turning a finding into a claim about the policy

The dropout sweep in the original study located a cliff between p = 0.4 and
p = 0.5. A cliff is a statement about **effect size**, not about significance:
the question is not "is the span at p = 0.5 different from the span at p = 0.4"
(any difference is, with enough episodes) but "has the span collapsed to the
value an input-blind policy would show". The report therefore compares the
observed span against the span of the *noise-image* control, where the null
value is known, and the dropout is set at the largest value whose effect size
against the noise control is still large.

The same logic applies to the input-blindness claim itself, which is why
`experiments/probe_permutation.py` exists: it states the null hypothesis
explicitly, substitutes a noise image for the real one within a condition, and
reports the mean paired difference and Cohen's dz alongside the p-value.

---

## 7 · What this analysis cannot tell you

- A 95 % interval covers the truth in 95 % of *repeated experiments*; it is not
  a statement about the current data.
- Conditioning on the same fixed condition table across policies removes
  between-condition variance but does not make the conditions a random sample of
  the real world. Sim-to-real transfer is not tested here.
- The design is over six factors; interactions between them are only visible to
  the extent that the design's strata cross, which for 40 conditions is partial.
- Latin hypercube sampling has no replication of the *same* condition, so
  pure-noise (aleatoric) variance is estimated across strata, not within.
