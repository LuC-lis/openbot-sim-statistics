"""Logistic regression on the failure indicator.

Closed-loop evaluation produces a binary outcome per episode (recovered vs.
did not).  A success rate is a one-number summary; the question worth asking is
*which conditions predict failure*.  This module fits

    logit P(fail_i) = beta_0 + sum_j beta_j x_ij

by iteratively reweighted least squares (Newton-Raphson on the log-likelihood),
which for a logistic model is both fast and exact to machine precision.  It is
implemented in numpy so the fit is reproducible without SciPy.

Reported alongside each coefficient: the Wald standard error from the inverse
observed Fisher information, the normal z statistic, a two-sided p-value, and
the odds ratio with its interval.  Coefficients are reported with the sample
standard deviation of the predictor, so magnitudes are comparable across
predictors measured in different units.

Guard rails: perfect separation is detected (the fitted probabilities collapse
to 0/1) and reported instead of silently returning a huge coefficient, and the
condition number of the information matrix is returned so a caller can see
collinearity in the design before trusting the standard errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["LogisticFit", "logistic_regression", "normal_p_value"]


def normal_p_value(z: float) -> float:
    """Two-sided p-value for a standard normal deviate."""
    return math.erfc(abs(z) / math.sqrt(2.0))


@dataclass
class LogisticFit:
    """Result of :func:`logistic_regression`."""

    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    p: np.ndarray
    odds_ratio: np.ndarray
    or_lo: np.ndarray
    or_hi: np.ndarray
    n: int
    n_events: int
    log_likelihood: float
    null_log_likelihood: float
    iterations: int
    converged: bool
    separated: bool
    cond_number: float
    standardised: bool = False
    scales: dict[str, float] = field(default_factory=dict)

    @property
    def mcfadden_r2(self) -> float:
        """McFadden pseudo-R^2, ``1 - ll / ll_null``."""
        if self.null_log_likelihood == 0:
            return float("nan")
        return 1.0 - self.log_likelihood / self.null_log_likelihood

    def to_rows(self) -> list[dict]:
        rows = []
        for i, name in enumerate(self.names):
            rows.append(
                {
                    "term": name,
                    "coef": float(self.coef[i]),
                    "se": float(self.se[i]),
                    "z": float(self.z[i]),
                    "p": float(self.p[i]),
                    "odds_ratio": float(self.odds_ratio[i]),
                    "or_lo": float(self.or_lo[i]),
                    "or_hi": float(self.or_hi[i]),
                }
            )
        return rows


def logistic_regression(
    X: np.ndarray,
    y: Sequence[int],
    names: Sequence[str] | None = None,
    standardise: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
    ridge: float = 1e-9,
) -> LogisticFit:
    """Fit a logistic regression by IRLS.

    Parameters
    ----------
    X
        ``(n, p)`` design matrix *without* an intercept column; one is added.
    y
        ``n`` binary outcomes (1 = failure, 0 = success).
    names
        Optional predictor names; defaults to ``x0 ... x_{p-1}``.
    standardise
        Centre and scale predictors by their sample standard deviation.  This
        keeps Newton steps well conditioned and makes coefficient magnitudes
        comparable; the fitted ``scales`` are recorded so a coefficient can be
        read as "per one standard deviation".
    ridge
        Tikhonov term added to the information matrix, so a perfectly separated
        or collinear design returns a finite (flagged) fit instead of raising.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if y.size != n:
        raise ValueError("X and y disagree on the number of observations")
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("y must be binary (0/1)")

    if names is None:
        names = [f"x{i}" for i in range(p)]
    names = list(names)

    scales = {}
    if standardise:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        scales = {name: float(s) for name, s in zip(names, sd)}
    else:
        Xs = X

    design = np.column_stack([np.ones(n), Xs])
    beta = np.zeros(p + 1, dtype=float)
    separated = False
    cond_number = float("nan")

    for it in range(1, max_iter + 1):
        eta = design @ beta
        eta = np.clip(eta, -500.0, 500.0)
        mu_i = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu_i * (1.0 - mu_i), 1e-12)
        grad = design.T @ (y - mu_i)
        info = design.T @ (design * w[:, None]) + ridge * np.eye(p + 1)
        cond_number = float(np.linalg.cond(info))
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge prevents this
            separated = True
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    eta = np.clip(design @ beta, -500.0, 500.0)
    mu_i = 1.0 / (1.0 + np.exp(-eta))
    if np.all(mu_i > 1 - 1e-6) or np.all(mu_i < 1e-6):
        separated = True
    w = np.maximum(mu_i * (1.0 - mu_i), 1e-12)
    info = design.T @ (design * w[:, None]) + ridge * np.eye(p + 1)
    try:
        cov = np.linalg.inv(info)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:  # pragma: no cover
        se = np.full(p + 1, np.nan)

    eps = 1e-12
    log_lik = float(np.sum(y * np.log(mu_i + eps) + (1 - y) * np.log(1 - mu_i + eps)))
    p_bar = float(np.clip(y.mean(), eps, 1 - eps))
    null_ll = float(n * (p_bar * math.log(p_bar) + (1 - p_bar) * math.log(1 - p_bar)))

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, np.nan)
        pvals = np.array([normal_p_value(v) if np.isfinite(v) else np.nan for v in z])
        or_ = np.exp(np.clip(beta, -500.0, 500.0))
        z_crit = 1.959963984540054
        or_lo = np.exp(np.clip(beta - z_crit * se, -500.0, 500.0))
        or_hi = np.exp(np.clip(beta + z_crit * se, -500.0, 500.0))

    return LogisticFit(
        names=["intercept"] + names,
        coef=beta,
        se=se,
        z=z,
        p=pvals,
        odds_ratio=or_,
        or_lo=or_lo,
        or_hi=or_hi,
        n=n,
        n_events=int(y.sum()),
        log_likelihood=log_lik,
        null_log_likelihood=null_ll,
        iterations=it,
        converged=it < max_iter,
        separated=separated,
        cond_number=cond_number,
        standardised=standardise,
        scales=scales,
    )
