"""CLI: run the designed evaluation and write the outcomes CSV.

    python -m statistics.evaluate_design --out results/outcomes_design.csv
    python -m statistics.evaluate_design --model models/dg_s3.tflite --out results/outcomes_learner.csv

Without ``--model`` the two baselines (random walk, privileged expert) are run,
which is the reference against which every learned policy is compared.  With
``--model`` the ``.tflite`` policy joins them on exactly the same conditions and
seeds, so the difference is paired and can be tested directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openbot_sim.policies import TFLitePolicy

from .design import ConditionDesign
from .outcomes import baseline_policies, run_design, write_csv
from .power import n_for_paired_effect, n_for_two_proportions


def plot_design(conditions, path: str) -> None:
    """Design-coverage figure: every factor, per replicate, perfectly stratified.

    This is the picture that justifies calling the evaluation *designed*: within
    each replicate every factor occupies each of its ``n`` strata exactly once,
    so no corner of any factor's range is systematically missed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = list(conditions[0].to_row())
    factors = [f for f in factors if f not in {"condition_id", "replicate", "seed"}]
    replicates = sorted({c.replicate for c in conditions})
    fig, axes = plt.subplots(len(factors), len(replicates),
                             figsize=(3.1 * len(replicates), 1.55 * len(factors)),
                             squeeze=False)
    for j, rep in enumerate(replicates):
        sub = sorted([c for c in conditions if c.replicate == rep], key=lambda c: c.condition_id)
        for i, factor in enumerate(factors):
            ax = axes[i][j]
            vals = [getattr(c, factor) for c in sub]
            ax.scatter(range(len(vals)), vals, s=9, color="#4a6fa5")
            if i == 0:
                ax.set_title(f"replicate {rep}", fontsize=9)
            if j == 0:
                ax.set_ylabel(factor, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="optional .tflite policy to evaluate alongside the baselines")
    ap.add_argument("--out", default="results/outcomes_design.csv")
    ap.add_argument("--design-out", default=None, help="also write the condition table as CSV")
    ap.add_argument("--design-figure", default=None, help="write a design-coverage figure (PNG)")
    ap.add_argument("--conditions", type=int, default=24, help="conditions per replicate")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--disturb", type=float, default=0.0, help="extra per-step state disturbance")
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()

    design = ConditionDesign(n_conditions=args.conditions, replicates=args.replicates, seed=args.seed)
    conditions = design.build()
    policies = baseline_policies()
    if args.model:
        policies["learned"] = lambda _rng: TFLitePolicy(args.model)

    print(
        f"design: {args.conditions} conditions x {args.replicates} replicates "
        f"= {len(conditions)} episodes per policy, {len(policies)} policies"
    )
    plan = n_for_two_proportions(0.625, 0.40)
    print(
        f"a-priori plan: {plan} episodes per policy are needed to detect 62.5 % vs 40 % "
        f"at 80 % power / alpha = 0.05 ({n_for_paired_effect(0.5)} pairs for dz = 0.5); "
        f"this run uses {len(conditions)}."
    )
    if args.design_out:
        import csv

        with open(args.design_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(conditions[0].to_row()))
            writer.writeheader()
            for cond in conditions:
                writer.writerow(cond.to_row())
        print(f"wrote condition table -> {args.design_out}")
    if args.design_figure:
        plot_design(conditions, args.design_figure)
        print(f"wrote design figure -> {args.design_figure}")
    rows = run_design(policies, conditions=conditions, max_steps=args.max_steps,
                      disturb=args.disturb, verbose=True)
    path = write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
