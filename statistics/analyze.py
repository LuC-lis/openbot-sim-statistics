"""CLI: turn an outcomes CSV into intervals, a paired comparison and a model.

    python -m statistics.analyze results/outcomes_design.csv --report results/report_design.md

Everything printed here is derived from the logged episodes; nothing is
hard-coded.  Three sections:

1. **Success rate with a Wilson score interval** — for every policy.
2. **Paired comparison against a baseline** — both policies are replayed on the
   same conditions with the same seeds, so the difference is bootstrapped on
   pairs and the permutation test permutes within pairs.
3. **Logistic regression on the failure indicator** — which *conditions* predict
   failure, with Wald standard errors, odds ratios and McFadden's pseudo-R^2.
   This is the part a success rate cannot tell you.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .intervals import bootstrap_ci, paired_bootstrap_diff, wilson_interval
from .outcomes import read_csv
from .power import minimum_detectable_difference, minimum_detectable_effect
from .regression import logistic_regression
from .tests import permutation_test

__all__ = ["analyze", "to_markdown", "main"]

_REGRESSORS = (
    ("lateral0", "initial lateral offset (m)"),
    ("heading0", "initial heading error (rad)"),
    ("obstacle_offset", "obstacle lateral offset (m)"),
    ("perturb_rate", "disturbance rate (per step)"),
)


def _by_policy(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["policy"], []).append(row)
    return out


def _paired(rows_a: list[dict], rows_b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Align two policies' rows on (replicate, condition_id)."""
    key = lambda r: (r["replicate"], r["condition_id"])  # noqa: E731
    b_index = {key(r): r for r in rows_b}
    a_aligned, b_aligned = [], []
    for ra in sorted(rows_a, key=key):
        rb = b_index.get(key(ra))
        if rb is not None:
            a_aligned.append(ra)
            b_aligned.append(rb)
    return a_aligned, b_aligned


def analyze(rows: list[dict], baseline: str = "random", seed: int = 0,
            n_resamples: int = 2000, model_per_policy: bool = True) -> dict:
    """Compute the whole report as a plain dict (so it can be unit-tested)."""
    groups = _by_policy(rows)
    policies = sorted(groups)

    summary = []
    for name in policies:
        rs = groups[name]
        k = sum(r["success"] for r in rs)
        n = len(rs)
        lo, hi = wilson_interval(k, n)
        track = [r["track_err"] for r in rs if np.isfinite(r["track_err"])]
        t_lo, t_hi = bootstrap_ci(track, np.mean, n_resamples=n_resamples, seed=seed)
        summary.append(
            {
                "policy": name,
                "n": n,
                "successes": k,
                "success": k / n if n else float("nan"),
                "success_lo": lo,
                "success_hi": hi,
                "track_err": float(np.mean(track)) if track else float("nan"),
                "track_lo": t_lo,
                "track_hi": t_hi,
                "reasons": dict(Counter(r["reason"] for r in rs)),
            }
        )

    comparisons = []
    if baseline in groups:
        for name in policies:
            if name == baseline:
                continue
            a, b = _paired(groups[name], groups[baseline])
            if not a:
                continue
            s_a = [r["success"] for r in a]
            s_b = [r["success"] for r in b]
            d, d_lo, d_hi = paired_bootstrap_diff(s_a, s_b, n_resamples=n_resamples, seed=seed)
            perm = permutation_test(s_a, s_b, paired=True, n_permutations=n_resamples, seed=seed)
            t_a = [r["track_err"] for r in a]
            t_b = [r["track_err"] for r in b]
            td, td_lo, td_hi = paired_bootstrap_diff(t_a, t_b, n_resamples=n_resamples, seed=seed)
            comparisons.append(
                {
                    "policy": name,
                    "vs": baseline,
                    "n_pairs": len(a),
                    "success_diff": d,
                    "success_diff_lo": d_lo,
                    "success_diff_hi": d_hi,
                    "perm_p": perm["p_value"],
                    "track_diff": td,
                    "track_diff_lo": td_lo,
                    "track_diff_hi": td_hi,
                }
            )

    # --- failure model -------------------------------------------------
    # A policy indicator is a *perfect* predictor whenever one policy never
    # survives, which drives the IRLS fit to separation and makes every
    # standard error meaningless.  So the default is to fit one model per
    # policy: the question "which conditions predict failure?" is then answered
    # within a policy, where the outcome actually varies.
    fits: dict[str, object] = {}
    if model_per_policy:
        for name in policies:
            sub = groups[name]
            y_sub = [1 - r["success"] for r in sub]
            if 0 < sum(y_sub) < len(y_sub):
                X_sub = [[r[key] for key, _ in _REGRESSORS] for r in sub]
                fits[name] = logistic_regression(
                    np.asarray(X_sub, dtype=float), y_sub,
                    names=[label for _, label in _REGRESSORS],
                )
    else:
        policy_names = [p for p in policies if p != baseline]
        X, y = [], []
        for row in rows:
            X.append([row[key] for key, _ in _REGRESSORS] + [1.0 if row["policy"] in policy_names else 0.0])
            y.append(1 - row["success"])
        names = [label for _, label in _REGRESSORS] + [f"policy != {baseline}"]
        fits["pooled"] = logistic_regression(np.asarray(X, dtype=float), y, names=names)

    return {
        "summary": summary,
        "comparisons": comparisons,
        "baseline": baseline,
        "n_resamples": n_resamples,
        "regression": fits,
        "regressors": [label for _, label in _REGRESSORS],
    }


def to_markdown(result: dict, source: str = "") -> str:
    s = result["summary"]
    lines = ["# Designed evaluation — statistical report", ""]
    if source:
        lines += [f"Source: `{source}`  ", ""]
    lines += [
        f"Bootstrap/permutation resamples: **{result['n_resamples']}** (fixed seeds, so the "
        "numbers below are reproducible).",
        "",
        "## Success rate (Wilson score interval)",
        "",
        "| policy | episodes | survived | success | 95% CI | mean track err | 95% CI |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in s:
        lines.append(
            f"| {row['policy']} | {row['n']} | {row['successes']} | {row['success']*100:.1f}% | "
            f"[{row['success_lo']*100:.1f}%, {row['success_hi']*100:.1f}%] | "
            f"{row['track_err']:.3f} | [{row['track_lo']:.3f}, {row['track_hi']:.3f}] |"
        )
    lines += ["", "Termination reasons:", ""]
    for row in s:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(row["reasons"].items(), key=lambda kv: -kv[1]))
        lines.append(f"- **{row['policy']}**: {reasons}")
    lines += [""]

    if result["comparisons"]:
        lines += [
            f"## Paired comparison against `{result['baseline']}`",
            "",
            "Same conditions, same seeds, so the difference is bootstrapped on pairs; "
            "the p-value comes from a paired permutation test (sign flips within pairs).",
            "",
            "| policy | pairs | Δ success | 95% CI | permutation p | Δ track err | 95% CI |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
        for c in result["comparisons"]:
            lines.append(
                f"| {c['policy']} | {c['n_pairs']} | {c['success_diff']*100:+.1f} pp | "
                f"[{c['success_diff_lo']*100:+.1f}, {c['success_diff_hi']*100:+.1f}] | "
                f"{c['perm_p']:.4f} | {c['track_diff']:+.3f} | "
                f"[{c['track_diff_lo']:+.3f}, {c['track_diff_hi']:+.3f}] |"
            )
        lines += [""]

    lines += ["## What this design could have detected", ""]
    for row in s:
        if row["n"] and 0 < row["success"] < 1:
            mde = minimum_detectable_difference(row["n"], min(max(row["success"], 0.05), 0.95))
            lines.append(
                f"- **{row['policy']}** (n = {row['n']}, observed {row['success']*100:.1f} %): "
                f"a difference of about **{mde*100:.1f} percentage points** against a comparison "
                "policy would have been detected with 80 % power at alpha = 0.05. "
                "Smaller differences remain open."
            )
    if result["comparisons"]:
        n_pairs = result["comparisons"][0]["n_pairs"]
        lines.append(
            f"- Paired tracking-error comparisons ({n_pairs} pairs): the smallest detectable "
            f"standardised effect is dz = {minimum_detectable_effect(n_pairs):.3f}."
        )
    lines += [""]

    fits = result["regression"]
    for name, fit in fits.items():
        title = name if name != "pooled" else "pooled (with policy indicator)"
        lines += [
            f"## Logistic regression on the failure indicator — {title}",
            "",
            f"n = {fit.n}, failures = {fit.n_events}, McFadden pseudo-R² = {fit.mcfadden_r2:.3f}, "
            f"IRLS iterations = {fit.iterations}, information-matrix condition number = {fit.cond_number:.2f}"
            + (", **flagged: separation**" if fit.separated else ""),
            "",
            "Coefficients are per one standard deviation of the predictor (continuous predictors "
            "are standardised); odds ratio > 1 means the factor makes failure more likely.",
            "",
            "| term | coef | SE | z | p | odds ratio | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in fit.to_rows():
            lines.append(
                f"| {row['term']} | {row['coef']:+.3f} | {row['se']:.3f} | {row['z']:+.2f} | "
                f"{row['p']:.4f} | {row['odds_ratio']:.2f} | "
                f"[{row['or_lo']:.2f}, {row['or_hi']:.2f}] |"
            )
        lines += [""]
    return "\n".join(lines)


def write_figure(rows: list[dict], result: dict, path: str) -> None:
    """Two-panel summary: success rate with its interval, and the paired spread.

    Panel (a) is the headline number with the Wilson interval drawn on it, so a
    reader can see how much of the vertical axis the uncertainty actually
    occupies.  Panel (b) shows every episode's tracking error, with the paired
    differences to the baseline next to it and a zero line — the design's whole
    point is that those differences are computed within a condition.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _by_policy(rows)
    policies = [s["policy"] for s in result["summary"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    xs = np.arange(len(policies))
    means = [next(s for s in result["summary"] if s["policy"] == p)["success"] * 100 for p in policies]
    los = [next(s for s in result["summary"] if s["policy"] == p)["success_lo"] * 100 for p in policies]
    his = [next(s for s in result["summary"] if s["policy"] == p)["success_hi"] * 100 for p in policies]
    ax.bar(xs, means, color="#4a6fa5", width=0.5)
    ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="none", ecolor="#20304a", capsize=6, lw=1.6)
    for x, m, lo, hi, p in zip(xs, means, los, his, policies):
        n = len(groups[p])
        ax.text(x, min(hi + 4, 96), f"{m:.1f}%\n[{lo:.1f}, {hi:.1f}]\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, policies)
    ax.set_ylabel("survived horizon (%)")
    ax.set_ylim(0, 108)
    ax.set_title("success rate with 95% Wilson interval")

    ax = axes[1]
    base = result["baseline"]
    for i, p in enumerate([q for q in policies if q != base]):
        a, b = _paired(groups[p], groups[base])
        diff = np.array([r["track_err"] for r in a]) - np.array([r["track_err"] for r in b])
        jitter = np.random.default_rng(0).normal(0, 0.04, diff.size)
        ax.scatter(np.full(diff.size, i) + jitter, diff, s=12, alpha=0.6, color="#c77d1a")
        ax.hlines(diff.mean(), i - 0.25, i + 0.25, color="#20304a", lw=2)
    ax.axhline(0.0, ls="--", color="gray", lw=1)
    ax.set_xticks(np.arange(len([q for q in policies if q != base])),
                  [f"{p} − {base}" for p in policies if p != base])
    ax.set_ylabel("paired difference in tracking error (m)")
    ax.set_title("paired differences within condition")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="outcomes CSV written by statistics.evaluate_design")
    ap.add_argument("--baseline", default="random")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pooled", action="store_true",
                    help="fit one pooled model with a policy indicator instead of one model per policy")
    ap.add_argument("--report", help="write the markdown report to this path")
    ap.add_argument("--figure", help="write a summary figure (PNG) to this path")
    args = ap.parse_args()

    rows = read_csv(args.csv)
    result = analyze(rows, baseline=args.baseline, seed=args.seed, n_resamples=args.resamples,
                     model_per_policy=not args.pooled)
    if args.figure:
        write_figure(rows, result, args.figure)
        print(f"wrote {args.figure}")
    md = to_markdown(result, source=str(args.csv))
    print(md)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md)
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
