# Designed evaluation — statistical report

Source: `results/outcomes_design.csv`  

Bootstrap/permutation resamples: **2000** (fixed seeds, so the numbers below are reproducible).

## Success rate (Wilson score interval)

| policy | episodes | survived | success | 95% CI | mean track err | 95% CI |
|---|---:|---:|---:|---|---:|---|
| expert | 120 | 75 | 62.5% | [53.6%, 70.6%] | 0.309 | [0.287, 0.332] |
| random | 120 | 0 | 0.0% | [0.0%, 3.1%] | 0.584 | [0.521, 0.648] |

Termination reasons:

- **expert**: timeout 75, collision 45
- **random**: turned 119, off_road 1

## Paired comparison against `random`

Same conditions, same seeds, so the difference is bootstrapped on pairs; the p-value comes from a paired permutation test (sign flips within pairs).

| policy | pairs | Δ success | 95% CI | permutation p | Δ track err | 95% CI |
|---|---:|---:|---|---:|---:|---|
| expert | 120 | +62.5 pp | [+53.3, +71.7] | 0.0010 | -0.275 | [-0.335, -0.216] |

## Logistic regression on the failure indicator — expert

n = 120, failures = 45, McFadden pseudo-R² = 0.020, IRLS iterations = 5, information-matrix condition number = 1.74

Coefficients are per one standard deviation of the predictor (continuous predictors are standardised); odds ratio > 1 means the factor makes failure more likely.

| term | coef | SE | z | p | odds ratio | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| intercept | -0.525 | 0.192 | -2.74 | 0.0062 | 0.59 | [0.41, 0.86] |
| initial lateral offset (m) | -0.040 | 0.198 | -0.20 | 0.8406 | 0.96 | [0.65, 1.42] |
| initial heading error (rad) | -0.020 | 0.192 | -0.11 | 0.9150 | 0.98 | [0.67, 1.43] |
| obstacle lateral offset (m) | -0.210 | 0.199 | -1.06 | 0.2897 | 0.81 | [0.55, 1.20] |
| disturbance rate (per step) | +0.277 | 0.196 | +1.42 | 0.1564 | 1.32 | [0.90, 1.94] |
