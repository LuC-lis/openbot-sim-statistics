"""Input-blindness as a testable null hypothesis: a paired permutation test.

The original diagnostic asked "does the policy ignore its camera?" and answered
it with an ablation — retrain without dropout, retrain without the command
channel, and see which one changes the behaviour.  That is a useful screen but
it answers a different question: *which component of the architecture is
responsible*.  What it does not give you is a test of the claim itself.

This script turns the claim into a null hypothesis:

    H0: the policy's commanded turn does not depend on the camera image.

Operationalised as: across the designed conditions, the span of the commanded
turn (most-left command minus most-right command, in radians of steering) is
the same whether the policy sees the real rendered image or a noise image
substituted for it.  Under H0 the two images within a condition are
exchangeable, so the paired differences can be permuted by sign flips — that is
:func:`statistics.tests.observation_channel_test`.

Because the pairing is by condition, the test is sensitive to a *systematic*
loss of steering span and immune to the condition-to-condition spread in how
large the span happens to be.

Requires TensorFlow only to load the ``.tflite`` policy.

Usage::

    python experiments/probe_permutation.py --model models/dg_s3.tflite
    python experiments/probe_permutation.py --model models/dg_s3.tflite --conditions 24 --json results/permutation_test.json
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from openbot_sim.camera import Camera  # noqa: E402
from openbot_sim.state import State  # noqa: E402
from openbot_sim.world import World  # noqa: E402
from statistics.design import ConditionDesign  # noqa: E402
from statistics.tests import observation_channel_test  # noqa: E402

#: lateral offsets swept to measure the steering span, in metres
SWEEP = (-0.6, -0.3, 0.0, 0.3, 0.6)


def turn_span(policy, world, camera, appearance, rng, noise=False, cmd=0.0):
    """Span of the commanded turn over the lateral sweep, for one condition."""
    turns = []
    for y in SWEEP:
        state = State(0.0, y, 0.0)
        image = camera.render(state, world)
        if noise:
            image = rng.random(image.shape).astype(np.float32)
        obs = {"image": image, "cmd": np.array([cmd], np.float32)}
        info = {"state": (0.0, y, 0.0), "lateral": y, "heading": 0.0, "target_lateral": 0.0}
        action = np.asarray(policy.act(obs, info), dtype=float).reshape(-1)
        turns.append(float(action[1] - action[0]))
    return float(turns[-1] - turns[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a .tflite policy")
    ap.add_argument("--conditions", type=int, default=24)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the test result as JSON to this path")
    args = ap.parse_args()

    from openbot_sim.policies import TFLitePolicy  # imported late: needs TF

    policy = TFLitePolicy(args.model)
    conditions = ConditionDesign(n_conditions=args.conditions, replicates=args.replicates).build()

    span_real, span_noise, rows = [], [], []
    for cond in conditions:
        world = World(road_half_width=cond.road_half_width, obstacles=cond.obstacles)
        camera = Camera((96, 256), appearance=cond.appearance)
        rng = np.random.default_rng(cond.seed)
        real = turn_span(policy, world, camera, cond.appearance, rng, noise=False, cmd=cond.cmd)
        noisy = turn_span(policy, world, camera, cond.appearance, rng, noise=True, cmd=cond.cmd)
        span_real.append(real)
        span_noise.append(noisy)
        rows.append({**cond.to_row(), "span_real": real, "span_noise": noisy})

    result = observation_channel_test(span_real, span_noise, n_permutations=args.permutations, seed=args.seed)
    print(f"{'condition':>9}{'replicate':>10}{'span(real)':>12}{'span(noise)':>13}")
    for row in rows:
        print(f"{row['condition_id']:>9}{row['replicate']:>10}"
              f"{row['span_real']:>+12.3f}{row['span_noise']:>+13.3f}")
    print()
    print(f"mean span, real image  : {np.mean(span_real):+.4f}")
    print(f"mean span, noise image : {np.mean(span_noise):+.4f}")
    print(f"mean paired difference : {result['mean_difference']:+.4f}")
    print(f"effect size (Cohen dz) : {result['effect_size_dz']:+.2f}")
    print(f"permutation p ({args.permutations}) : {result['p_value']:.4f}")
    print("H0: the policy's commanded turn does not depend on the image.")
    if result["p_value"] < 0.05:
        print("-> reject H0: the policy is using its camera.")
    else:
        print("-> do NOT reject H0: consistent with an input-blind policy.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"test": "observation_channel_test", "result": result, "rows": rows}, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
