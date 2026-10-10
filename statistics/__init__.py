"""statistics — an analysis layer for closed-loop robot-policy evaluations.

This package turns a simulator that *produces* numbers into an experiment that
*reports* them with stated uncertainty.  It is deliberately dependency-light
(``numpy`` only, no SciPy, no TensorFlow) so that every interval, test and
coefficient in the accompanying report can be recomputed from the logged
per-episode outcomes on any machine.

Modules
-------
``design``      Latin-hypercube condition design + paired seed sets
``outcomes``    run a policy over a design, log one tidy row per episode
``intervals``   Wilson score interval, percentile bootstrap, paired bootstrap
``tests``       permutation tests (unpaired and paired)
``regression``  logistic regression on the failure indicator (IRLS + Wald SEs)
``power``       a-priori sample size and post-hoc minimum detectable effect
``analyze``     CLI: outcomes CSV -> interval table, model table, markdown report

Typical use::

    python -m statistics.evaluate_design --out results/outcomes_design.csv
    python -m statistics.analyze results/outcomes_design.csv --report results/report_design.md
"""

from .design import ConditionDesign, latin_hypercube  # noqa: F401
from .intervals import (  # noqa: F401
    bootstrap_ci,
    paired_bootstrap_diff,
    wilson_interval,
)
from .power import (  # noqa: F401
    minimum_detectable_difference,
    minimum_detectable_effect,
    n_for_paired_effect,
    n_for_two_proportions,
)
from .regression import LogisticFit, logistic_regression  # noqa: F401
from .tests import permutation_test  # noqa: F401

__all__ = [
    "ConditionDesign",
    "latin_hypercube",
    "wilson_interval",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "permutation_test",
    "logistic_regression",
    "LogisticFit",
    "n_for_two_proportions",
    "n_for_paired_effect",
    "minimum_detectable_difference",
    "minimum_detectable_effect",
]

__version__ = "1.0.0"
